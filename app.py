from flask import Flask, request, send_file, jsonify
from PIL import Image
import numpy as np
import os
import sys
import platform
import base64
import time
import json
from io import BytesIO

import cv2
import requests

# Optional (but in your requirements): rembg
try:
    from rembg import new_session, remove as rembg_remove
except Exception:
    new_session = None
    rembg_remove = None

# Optional: package listing
try:
    import importlib.metadata as importlib_metadata
except Exception:
    importlib_metadata = None


app = Flask(__name__)

# -----------------------------
# ENV / CONFIG
# -----------------------------
API_KEY = os.environ.get("API_KEY", None)
FAL_KEY = os.environ.get("FAL_KEY", None)

MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE_MB", 10)) * 1024 * 1024
DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

REMBG_MODEL = os.environ.get("REMBG_MODEL", "isnet-general-use")  # good for logos
REMBG_SESSION = None


# -----------------------------
# AUTH / VALIDATION
# -----------------------------
def verify_api_key():
    """Verify API key if set in environment"""
    if API_KEY:
        provided_key = request.headers.get("X-API-Key") or request.form.get("api_key")
        if provided_key != API_KEY:
            return jsonify({"error": "Unauthorized", "message": "Invalid or missing API key"}), 401
    return None


def _enforce_only_image_field():
    """Reject requests that include unexpected file fields (helps n8n / form-data mistakes)."""
    if not request.files:
        return jsonify({"error": "No files provided"}), 400
    allowed = {"image"}
    got = set(request.files.keys())
    if got != allowed:
        return jsonify({
            "error": "Invalid file fields",
            "expected": ["image"],
            "received": sorted(list(got))
        }), 400
    return None


# -----------------------------
# UTIL: IO / TRANSPARENCY
# -----------------------------
def _open_image_bytes(img_bytes: bytes) -> Image.Image:
    img = Image.open(BytesIO(img_bytes))
    # Normalize orientation if EXIF is present (best-effort)
    try:
        exif = getattr(img, "getexif", None)
        if exif:
            orientation = exif().get(274)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img


def _to_png_bytes(img_bytes: bytes) -> bytes:
    img = _open_image_bytes(img_bytes).convert("RGBA")
    out = BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def has_transparency(img_bytes: bytes) -> bool:
    img = _open_image_bytes(img_bytes)
    if img.mode in ("RGBA", "LA"):
        alpha = np.array(img.getchannel("A"))
        return bool(np.any(alpha < 255))
    if "transparency" in img.info:
        return True
    return False


def _img_bytes_has_transparency(img_bytes: bytes) -> bool:
    # Back-compat name (your older code used this)
    return has_transparency(img_bytes)


# -----------------------------
# HEALTH (list installed packages)
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    auth_error = verify_api_key()
    if auth_error:
        return auth_error

    packages = []
    if importlib_metadata is not None:
        try:
            for d in importlib_metadata.distributions():
                name = d.metadata.get("Name") or d.metadata.get("Summary") or "unknown"
                version = d.version or "unknown"
                packages.append({"name": name, "version": version})
            packages.sort(key=lambda x: (x["name"] or "").lower())
        except Exception as e:
            packages = [{"name": "error", "version": str(e)}]
    else:
        packages = [{"name": "importlib.metadata", "version": "unavailable"}]

    return jsonify({
        "status": "healthy",
        "python": sys.version.split(" ")[0],
        "platform": platform.platform(),
        "auth": "required" if bool(API_KEY) else "not_required",
        "fal_configured": bool(FAL_KEY),
        "rembg_available": bool(rembg_remove and new_session),
        "rembg_model": REMBG_MODEL,
        "installed_packages_count": len(packages),
        "installed_packages": packages
    })


# -----------------------------
# fal.ai UPSCALE (optional)
# -----------------------------
def enhance_image_fal(image_bytes: bytes, wait_timeout=120, poll_interval=1.5):
    """
    fal-ai/seedvr/upscale/image (Queue API)
    Returns: (bytes, enhanced_bool, message)
    """
    if not FAL_KEY:
        return image_bytes, False, "FAL_KEY not configured"

    def _first(x):
        return x[0] if isinstance(x, list) and x else x

    def _decode_data_uri(data_uri: str) -> bytes:
        try:
            _, b64 = data_uri.split(",", 1)
            return base64.b64decode(b64)
        except Exception:
            return b""

    try:
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")
        img = _open_image_bytes(image_bytes)
        mime_type = "image/png" if (img.format or "").upper() == "PNG" else "image/jpeg"
        data_uri = f"data:{mime_type};base64,{img_base64}"

        headers = {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}

        submit = requests.post(
            "https://queue.fal.run/fal-ai/seedvr/upscale/image",
            headers=headers,
            json={"image_url": data_uri},
            timeout=60,
        )
        if submit.status_code not in (200, 201, 202):
            return image_bytes, False, f"fal submit failed: HTTP {submit.status_code} {submit.text[:200]}"

        submit_json = _first(submit.json())
        request_id = (submit_json or {}).get("request_id")
        if not request_id:
            return image_bytes, False, "fal submit response missing request_id"

        status_url = f"https://queue.fal.run/fal-ai/seedvr/requests/{request_id}/status"
        result_url = f"https://queue.fal.run/fal-ai/seedvr/requests/{request_id}"

        start = time.monotonic()
        while (time.monotonic() - start) < wait_timeout:
            st = requests.get(status_url, headers={"Authorization": f"Key {FAL_KEY}"}, timeout=30)
            if st.status_code not in (200, 202):
                return image_bytes, False, f"fal status failed: HTTP {st.status_code} {st.text[:200]}"

            st_json = _first(st.json()) or {}
            status = st_json.get("status")

            if status in ("COMPLETED", "SUCCEEDED"):
                break
            if status in ("FAILED", "CANCELED", "CANCELLED"):
                return image_bytes, False, f"fal failed: {st_json}"

            time.sleep(poll_interval)

        res = requests.get(result_url, headers={"Authorization": f"Key {FAL_KEY}"}, timeout=60)
        if res.status_code != 200:
            return image_bytes, False, f"fal result failed: HTTP {res.status_code} {res.text[:200]}"

        res_json = _first(res.json()) or {}
        image_obj = res_json.get("image") or {}
        out_url = image_obj.get("url") or ""

        if out_url.startswith("data:"):
            out_bytes = _decode_data_uri(out_url)
            if out_bytes:
                return out_bytes, True, "fal enhanced (data uri)"
            return image_bytes, False, "fal returned data uri but decode failed"

        if out_url.startswith("http"):
            dl = requests.get(out_url, timeout=60)
            if dl.status_code == 200 and dl.content:
                return dl.content, True, "fal enhanced (downloaded)"
            return image_bytes, False, f"fal output download failed: HTTP {dl.status_code}"

        return image_bytes, False, "fal result missing image.url"

    except Exception as e:
        return image_bytes, False, f"fal exception: {e}"


# -----------------------------
# BG REMOVAL ANALYSIS (auto decision)
# -----------------------------
def analyze_image_for_bg_removal(img: Image.Image):
    """
    Analyze image characteristics to decide best removal strategy.
    """
    img_rgb = img.convert("RGB")
    data = np.array(img_rgb)
    h, w = data.shape[:2]

    analysis = {
        "has_solid_bg": False,
        "bg_color": None,
        "bg_coverage": 0.0,
        "is_graphic": False,
        "color_complexity": 0.0,
        "edge_sharpness": 0.0,
    }

    # Sample border patches
    corner_size = max(6, min(40, h // 15, w // 15))
    corners = [
        data[0:corner_size, 0:corner_size],
        data[0:corner_size, w - corner_size:w],
        data[h - corner_size:h, 0:corner_size],
        data[h - corner_size:h, w - corner_size:w],
    ]
    edge_samples = [
        data[0:corner_size, w // 2 - corner_size // 2:w // 2 + corner_size // 2],
        data[h - corner_size:h, w // 2 - corner_size // 2:w // 2 + corner_size // 2],
        data[h // 2 - corner_size // 2:h // 2 + corner_size // 2, 0:corner_size],
        data[h // 2 - corner_size // 2:h // 2 + corner_size // 2, w - corner_size:w],
    ]
    border_samples = corners + edge_samples

    means = []
    stds = []
    for s in border_samples:
        px = s.reshape(-1, 3).astype(np.float32)
        means.append(px.mean(axis=0))
        stds.append(px.std(axis=0).mean())

    means = np.array(means, dtype=np.float32)
    stds = np.array(stds, dtype=np.float32)

    # If border regions are similar and low variance => solid bg likely
    mean_dist = float(np.mean(np.linalg.norm(means - np.mean(means, axis=0), axis=1)))
    std_mean = float(np.mean(stds))

    if mean_dist < 18 and std_mean < 22:
        analysis["has_solid_bg"] = True
        bg = np.median(means, axis=0)
        analysis["bg_color"] = tuple(int(x) for x in bg)

        # Estimate bg coverage with a tolerant threshold (compression noise-safe)
        bg_rgb = np.array(bg, dtype=np.float32)[None, None, :]
        dist = np.linalg.norm(data.astype(np.float32) - bg_rgb, axis=2)
        analysis["bg_coverage"] = float(np.mean(dist < max(18.0, std_mean * 1.3)))

    # Color complexity (downsample + unique count)
    small = cv2.resize(data, (max(32, w // 10), max(32, h // 10)), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3)
    unique_colors = len(np.unique(pixels, axis=0))
    analysis["color_complexity"] = float(unique_colors / pixels.shape[0])

    # Edge sharpness (Sobel magnitude)
    gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    analysis["edge_sharpness"] = float(np.mean(mag))

    # Heuristic: graphic/logo tends to be low complexity + sharper edges + often solid bg
    analysis["is_graphic"] = bool(
        (analysis["color_complexity"] < 0.35 and analysis["edge_sharpness"] > 20)
        or (analysis["has_solid_bg"] and analysis["color_complexity"] < 0.45)
    )

    return analysis


# -----------------------------
# EDGE HELPERS (trim + defringe)
# -----------------------------
def trim_transparent(img: Image.Image, padding: int = 2) -> Image.Image:
    img = img.convert("RGBA")
    alpha = np.array(img.getchannel("A"))
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(img.width - 1, x1 + padding)
    y1 = min(img.height - 1, y1 + padding)
    return img.crop((x0, y0, x1 + 1, y1 + 1))


def _defringe_rgba(data_rgba: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Remove "white/black halo" by replacing semi-transparent RGB with nearest opaque neighbor RGB.
    """
    if radius <= 0:
        return data_rgba

    rgb = data_rgba[:, :, :3].copy()
    alpha = data_rgba[:, :, 3].copy()

    opaque = (alpha >= 240).astype(np.uint8) * 255
    if opaque.sum() == 0:
        return data_rgba

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    # Expand opaque area outward to borrow colors
    opaque_dil = cv2.dilate(opaque, k, iterations=1)

    # For each channel, dilate using mask of opaque pixels
    # We approximate "nearest opaque" by a few dilations
    borrowed = rgb.copy()
    for _ in range(radius):
        borrowed = cv2.dilate(borrowed, k, iterations=1)

    semi = (alpha > 0) & (alpha < 255)
    # Only replace where we're near an opaque region (avoid painting fully transparent areas)
    near_opaque = (opaque_dil > 0) & semi
    rgb[near_opaque] = borrowed[near_opaque]

    out = data_rgba.copy()
    out[:, :, :3] = rgb
    out[:, :, 3] = alpha
    return out


def refine_edges(img: Image.Image, feather_amount: int = 2, defringe_radius: int = 2) -> Image.Image:
    """
    Light feather + defringe for cleaner logo edges after background removal.
    """
    img = img.convert("RGBA")
    data = np.array(img, dtype=np.uint8)
    alpha = data[:, :, 3].astype(np.uint8)

    # Feather only near edge pixels (where alpha is not fully opaque/transparent)
    edge_mask = (alpha > 0) & (alpha < 255)
    if edge_mask.any() and feather_amount > 0:
        k = feather_amount * 2 + 1
        blurred = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha[edge_mask] = blurred[edge_mask]
        data[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)

    if defringe_radius > 0:
        data = _defringe_rgba(data, radius=defringe_radius)

    return Image.fromarray(data, "RGBA")


# -----------------------------
# BG REMOVAL: COLOR (solid bg) - UPGRADED
# -----------------------------
def _estimate_bg_from_border_rgb(img_rgb: np.ndarray):
    h, w = img_rgb.shape[:2]
    b = max(2, min(20, h // 40, w // 40))  # border thickness
    top = img_rgb[0:b, :, :]
    bottom = img_rgb[h - b:h, :, :]
    left = img_rgb[:, 0:b, :]
    right = img_rgb[:, w - b:w, :]
    border = np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)
    bg_rgb = np.median(border, axis=0).astype(np.float32)
    bg_std = float(np.mean(np.std(border.astype(np.float32), axis=0)))
    return bg_rgb, bg_std, b


def remove_bg_color_method_v2(img: Image.Image, bg_color=None, tolerance=None):
    """
    Robust solid-bg removal:
    - Works better on compressed JPEG logos (border noise)
    - Uses LAB distance + connected-components from border
    - Soft edge alpha + defringe to reduce halo
    Returns: (result_img, meta)
    """
    img = img.convert("RGBA")
    rgba = np.array(img, dtype=np.uint8)
    rgb = rgba[:, :, :3]
    h, w = rgb.shape[:2]

    bg_rgb, bg_std, border_thickness = _estimate_bg_from_border_rgb(rgb)
    if bg_color is not None:
        bg_rgb = np.array(bg_color, dtype=np.float32)

    # Adaptive tolerance (handles noisy JPG backgrounds)
    if tolerance is None:
        # base + noise factor, clamped
        tolerance = int(np.clip(18 + bg_std * 1.6, 18, 48))

    # LAB distance (more perceptual)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
    bg_lab = cv2.cvtColor(np.uint8([[bg_rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.int16)
    d = lab - bg_lab[None, None, :]
    dist = np.sqrt((d[:, :, 0] ** 2) + (d[:, :, 1] ** 2) + (d[:, :, 2] ** 2)).astype(np.float32)

    # Candidate background by distance threshold
    cand = (dist <= float(tolerance)).astype(np.uint8)

    # Connected components: keep components that touch border as "background"
    num, labels = cv2.connectedComponents(cand, connectivity=8)
    if num <= 1:
        # fallback: nothing detected
        out = img.copy()
        meta = {"bg_rgb": tuple(int(x) for x in bg_rgb), "tolerance": tolerance, "reason": "no_components"}
        return out, meta

    border_labels = set()
    # collect labels touching border
    border = np.zeros((h, w), dtype=bool)
    border[0:border_thickness, :] = True
    border[h - border_thickness:h, :] = True
    border[:, 0:border_thickness] = True
    border[:, w - border_thickness:w] = True

    border_touch = labels[border]
    for v in np.unique(border_touch):
        if v != 0:
            border_labels.add(int(v))

    bg_mask = np.isin(labels, list(border_labels))

    # Build alpha: background => 0
    alpha = np.full((h, w), 255, dtype=np.uint8)
    alpha[bg_mask] = 0

    # Soft edges (anti-alias): only near background boundary
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bg_dil = cv2.dilate(bg_mask.astype(np.uint8) * 255, k, iterations=1) > 0
    edge_zone = bg_dil & (~bg_mask)

    feather = max(8.0, float(tolerance) * 0.8)
    soft = edge_zone & (dist < (float(tolerance) + feather))

    # When dist == tolerance => alpha 0; when dist == tolerance+feather => alpha 255
    a_soft = ((dist[soft] - float(tolerance)) / feather) * 255.0
    alpha[soft] = np.clip(a_soft, 0, 255).astype(np.uint8)

    out = rgba.copy()
    out[:, :, 3] = alpha

    # Defringe to avoid white/black halo
    out = _defringe_rgba(out, radius=2)

    meta = {
        "bg_rgb": tuple(int(x) for x in bg_rgb),
        "tolerance": int(tolerance),
        "bg_removed_ratio": float(np.mean(alpha == 0)),
    }
    return Image.fromarray(out, "RGBA"), meta


# -----------------------------
# BG REMOVAL: AI (rembg) - stronger defaults
# -----------------------------
def _get_rembg_session():
    global REMBG_SESSION
    if REMBG_SESSION is None and new_session is not None:
        REMBG_SESSION = new_session(REMBG_MODEL)
    return REMBG_SESSION


def remove_bg_ai_method(img: Image.Image, is_graphic: bool):
    """
    Rembg-based removal:
    - For graphics/logos: keep crisp edges (no alpha matting)
    - For photos: enable alpha matting
    """
    if rembg_remove is None or new_session is None:
        # If rembg not available, just return original
        return img.convert("RGBA"), {"reason": "rembg_unavailable"}

    session = _get_rembg_session()

    pil_in = img.convert("RGBA")

    try:
        if is_graphic:
            out = rembg_remove(
                pil_in,
                session=session,
                post_process_mask=True,
            )
        else:
            out = rembg_remove(
                pil_in,
                session=session,
                post_process_mask=True,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=12,
            )

        out = out.convert("RGBA")
        return out, {"reason": "rembg_ok"}

    except Exception as e:
        return pil_in, {"reason": f"rembg_exception:{e}"}


# -----------------------------
# BG REMOVAL: AUTO (decision + validation)
# -----------------------------
def _corners_transparent_ratio(img_rgba: Image.Image) -> float:
    data = np.array(img_rgba.convert("RGBA"), dtype=np.uint8)
    a = data[:, :, 3]
    h, w = a.shape[:2]
    cs = max(6, min(40, h // 15, w // 15))
    corners = [
        a[0:cs, 0:cs],
        a[0:cs, w - cs:w],
        a[h - cs:h, 0:cs],
        a[h - cs:h, w - cs:w],
    ]
    total = sum(c.size for c in corners)
    trans = sum(int((c <= 5).sum()) for c in corners)
    return float(trans / max(1, total))


def remove_bg_auto_v2(img: Image.Image, analysis: dict):
    """
    Auto strategy:
    1) If solid bg likely -> try COLOR first
    2) Validate: corners must become transparent enough; otherwise fallback to AI
    """
    method_used = "color_v2"
    fallback_used = False

    # Prefer color if border says solid bg (common for logos)
    if analysis.get("has_solid_bg", False):
        out, meta = remove_bg_color_method_v2(img, bg_color=analysis.get("bg_color"))
        out = out.convert("RGBA")

        # Validate: corners should be mostly transparent
        corner_t = _corners_transparent_ratio(out)
        removed_ratio = float(np.mean(np.array(out.getchannel("A")) == 0))

        # If it barely removed anything OR corners still opaque -> fail
        if corner_t < 0.75 or removed_ratio < 0.10:
            fallback_used = True
            method_used = "ai_rembg_fallback"
            out, _ = remove_bg_ai_method(img, is_graphic=analysis.get("is_graphic", False))

        return out.convert("RGBA"), method_used, fallback_used

    # If not solid bg, go AI first
    method_used = "ai_rembg"
    out, _ = remove_bg_ai_method(img, is_graphic=analysis.get("is_graphic", False))
    return out.convert("RGBA"), method_used, fallback_used


# -----------------------------
# /remove-bg (ONLY endpoint kept)
# -----------------------------
@app.route("/remove-bg", methods=["POST"])
def remove_bg_endpoint():
    """
    Only accepts: multipart/form-data with field "image"

    Optional fields:
      - enhance: true/false (default=false)
      - trim: true/false (default=true)
      - output_format: png/webp (default=png)
      - bg_remove: auto/ai/color/skip (default=auto)
    """
    start_time = time.time()
    processing_log = []

    def t_ms():
        return int((time.time() - start_time) * 1000)

    def log(step, success=True, **data):
        entry = {"step": step, "success": bool(success), "t_ms": t_ms()}
        entry.update(data)
        processing_log.append(entry)

    def attach_logs_to_response(resp):
        summary = []
        for e in processing_log:
            summary.append(f"{e['step']}:{'ok' if e.get('success') else 'fail'}@{e.get('t_ms')}ms")
        resp.headers["X-Step-Log"] = " | ".join(summary)

        b = json.dumps(processing_log, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        resp.headers["X-Step-Log-Json"] = base64.b64encode(b).decode("ascii")
        return resp

    def json_error(payload, status=400):
        payload["processing_log"] = processing_log
        resp = jsonify(payload)
        resp.status_code = status
        return attach_logs_to_response(resp)

    # auth
    log("auth_check", success=True)
    auth_error = verify_api_key()
    if auth_error:
        resp, status = auth_error
        resp.status_code = status
        log("auth_check", success=False, reason="invalid_or_missing_api_key")
        return attach_logs_to_response(resp)

    # enforce image-only field
    enforce_error = _enforce_only_image_field()
    if enforce_error:
        resp, status = enforce_error
        resp.status_code = status
        log("validate_input", success=False, reason="invalid_file_fields")
        return attach_logs_to_response(resp)

    try:
        file = request.files["image"]
        img_bytes = file.read() or b""
        if not img_bytes:
            log("read_image", success=False, reason="empty_file")
            return json_error({"error": "Empty image file"}, status=400)

        log("read_image", success=True, bytes=len(img_bytes), filename=getattr(file, "filename", ""))

        # params
        do_enhance = request.form.get("enhance", "false").lower() == "true"
        do_trim = request.form.get("trim", "true").lower() == "true"
        output_format = request.form.get("output_format", "png").strip().lower()

        bg_remove = request.form.get("bg_remove", "auto").strip().lower()
        allowed_bg = {"auto", "ai", "color", "skip"}
        if bg_remove not in allowed_bg:
            log("read_params", success=False, reason="invalid_bg_remove", received=bg_remove)
            return json_error({"error": "Invalid bg_remove value", "allowed": sorted(list(allowed_bg))}, status=400)

        log("read_params", success=True, enhance=do_enhance, trim=do_trim, output_format=output_format, bg_remove=bg_remove)

        # If already transparent, skip removal immediately
        try:
            already_transparent = has_transparency(img_bytes)
            log("check_transparency", success=True, already_transparent=already_transparent)
        except Exception as e:
            already_transparent = False
            log("check_transparency", success=False, error=str(e), already_transparent=False)

        # Optional enhance (fal) BEFORE bg removal
        enhanced = False
        enhance_msg = "not requested"
        if do_enhance:
            enhanced_bytes, enhanced, enhance_msg = enhance_image_fal(img_bytes)
            if enhanced and enhanced_bytes:
                img_bytes = enhanced_bytes
            log("enhance_fal", success=True, applied=bool(enhanced), message=str(enhance_msg))

        # decode image
        img = _open_image_bytes(img_bytes)
        log("decode_image", success=True, mode=img.mode, size=f"{img.width}x{img.height}")

        # analysis
        analysis = analyze_image_for_bg_removal(img)
        log("analyze", success=True, **analysis)

        # bg removal
        method_used = "skip"
        fallback_used = False

        if already_transparent:
            result_img = img.convert("RGBA")
            method_used = "skip_already_transparent"

        else:
            if bg_remove == "skip":
                result_img = img.convert("RGBA")
                method_used = "skip_requested"

            elif bg_remove == "color":
                result_img, meta = remove_bg_color_method_v2(img, bg_color=analysis.get("bg_color"))
                method_used = "color_v2_forced"
                log("bg_remove_color_v2", success=True, **meta)

            elif bg_remove == "ai":
                result_img, meta = remove_bg_ai_method(img, is_graphic=analysis.get("is_graphic", False))
                method_used = "ai_rembg_forced"
                log("bg_remove_ai", success=True, **meta)

            else:
                # auto
                result_img, method_used, fallback_used = remove_bg_auto_v2(img, analysis)

        log("bg_removed", success=True, method_used=method_used, fallback_used=fallback_used)

        # edge refine
        result_img = refine_edges(result_img, feather_amount=2, defringe_radius=2)
        log("refine_edges", success=True)

        # trim
        if do_trim:
            result_img = trim_transparent(result_img, padding=2)
            log("trim", success=True, out_size=f"{result_img.width}x{result_img.height}")
        else:
            log("trim", success=True, skipped=True)

        # encode
        out = BytesIO()
        if output_format == "webp":
            # Keep alpha; use lossless for logos
            result_img.save(out, format="WEBP", lossless=True, quality=100, method=6)
            mimetype = "image/webp"
            ext = "webp"
        else:
            result_img.save(out, format="PNG", optimize=True)
            mimetype = "image/png"
            ext = "png"

        out.seek(0)
        log("encode", success=True, format=ext, out_bytes=out.getbuffer().nbytes)

        processing_time = time.time() - start_time
        log("done", success=True, processing_time_s=f"{processing_time:.2f}")

        resp = send_file(
            out,
            mimetype=mimetype,
            as_attachment=False,
            download_name=f"removed_bg.{ext}",
        )

        resp.headers["X-Bg-Remove"] = bg_remove
        resp.headers["X-Method-Used"] = method_used
        resp.headers["X-Fallback-Used"] = str(bool(fallback_used))

        resp.headers["X-Has-Solid-BG"] = str(analysis.get("has_solid_bg", False))
        resp.headers["X-Is-Graphic"] = str(analysis.get("is_graphic", False))
        resp.headers["X-Already-Transparent"] = str(bool(already_transparent))

        resp.headers["X-Enhanced"] = str(bool(enhanced))
        resp.headers["X-Enhance-Status"] = str(enhance_msg)

        resp.headers["X-Trimmed"] = str(bool(do_trim))
        resp.headers["X-Processing-Time"] = f"{processing_time:.2f}s"
        resp.headers["X-Output-Size"] = f"{result_img.width}x{result_img.height}"

        return attach_logs_to_response(resp)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log("exception", success=False, error=str(e))
        return json_error({"error": "Processing failed", "details": str(e), "traceback": tb}, status=500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=DEBUG)
