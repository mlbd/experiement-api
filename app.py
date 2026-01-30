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
import ftplib
import uuid
import re

# Optional (but likely in your requirements): rembg
try:
    from rembg import new_session, remove as rembg_remove
except Exception:
    new_session = None
    rembg_remove = None

# Optional: installed packages listing
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

# FTP Configuration for /process-logo endpoint
FTP_HOST = os.environ.get('FTP_HOST', None)
FTP_USER = os.environ.get('FTP_USER', None)
FTP_PASS = os.environ.get('FTP_PASS', None)
FTP_DIR = os.environ.get('FTP_DIR', '/logos')
FTP_BASE_URL = os.environ.get('FTP_BASE_URL', None)

DEFAULT_THRESHOLD = int(os.environ.get('DEFAULT_THRESHOLD', 100))

MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE_MB", 10)) * 1024 * 1024
DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

REMBG_MODEL = os.environ.get("REMBG_MODEL", "isnet-general-use")
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
    """Reject requests that include unexpected file fields (helps form-data mistakes)."""
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
    # Normalize orientation (best-effort)
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
                name = d.metadata.get("Name") or "unknown"
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
# SHARPEN (RGB only) + ALPHA EDGE SOFTEN
# -----------------------------
def sharpen_rgb_keep_alpha(img_rgba: Image.Image, amount=1.10, radius=1.2, threshold=3):
    """
    Unsharp mask on RGB only (alpha preserved).
    Safer for logos after upscaling. Keeps transparency clean.
    """
    img = img_rgba.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)

    rgb = arr[:, :, :3].astype(np.float32)
    a = arr[:, :, 3:4]  # keep alpha

    k = int(max(1, round(radius * 2 + 1)))
    if k % 2 == 0:
        k += 1
    blur = cv2.GaussianBlur(rgb, (k, k), 0)

    diff = rgb - blur
    if threshold > 0:
        m = (np.max(np.abs(diff), axis=2, keepdims=True) >= threshold).astype(np.float32)
        diff = diff * m

    sharp = np.clip(rgb + (amount * diff), 0, 255).astype(np.uint8)
    out = np.concatenate([sharp, a], axis=2)
    return Image.fromarray(out, "RGBA")

def remove_inner_bg_holes(alpha_u8: np.ndarray, cand_bg: np.ndarray, max_area_ratio=0.02):
    """
    Remove background-like regions that are NOT border-connected (holes inside letters).
    alpha_u8: current alpha mask (0..255)
    cand_bg: boolean mask of "looks like bg" pixels (same as cand)
    max_area_ratio: don't remove huge areas (protects real fills/highlights)
    """
    h, w = alpha_u8.shape[:2]
    hole = cand_bg & (alpha_u8 > 0)  # bg-like pixels that still remain

    if not np.any(hole):
        return alpha_u8

    num, labels = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
    if num <= 1:
        return alpha_u8

    max_area = int(h * w * max_area_ratio)
    out = alpha_u8.copy()

    # ring check: must be surrounded by opaque pixels (typical letter holes)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for lab in range(1, num):
        comp = (labels == lab)
        area = int(comp.sum())
        if area == 0:
            continue

        # protect big areas (likely real design fill or background chunk)
        if area > max_area:
            continue

        # compute boundary ring and see if it's surrounded by opaque pixels
        dil = cv2.dilate(comp.astype(np.uint8), k, iterations=1).astype(bool)
        ring = dil & (~comp)

        if not np.any(ring):
            continue

        # surrounded if most ring pixels are opaque
        surround_ratio = float(np.mean(out[ring] > 220))
        if surround_ratio >= 0.60:
            out[comp] = 0  # make hole transparent

    return out

def pil_to_png_bytes(img_rgba: Image.Image) -> bytes:
    buf = BytesIO()
    img_rgba.convert("RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def _has_alpha(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA"):
        return True
    return False

def restore_alpha_if_missing(enhanced: Image.Image, alpha_src: Image.Image) -> Image.Image:
    """
    If enhanced output loses alpha, restore alpha from alpha_src (resized).
    """
    enh = enhanced.convert("RGBA")
    # If enhanced already has meaningful alpha, keep it
    a = np.array(enh.getchannel("A"))
    if np.any(a < 255):
        return enh

    # Restore alpha from source
    src = alpha_src.convert("RGBA")
    src_a = src.getchannel("A")

    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = Image.LANCZOS

    src_a_rs = src_a.resize(enh.size, resample=resample)
    out = enh.copy()
    out.putalpha(src_a_rs)
    return out


def cleanup_edge_spill(img_rgba: Image.Image, bg_rgb=(255, 255, 255), band_px=2, dist_thresh=26, gamma=1.6):
    """
    Removes background-colored residue only near the transparency edge (safe).
    - band_px: thickness around transparent area to treat as "edge band"
    - dist_thresh: how close to bg color counts as spill (LAB distance)
    - gamma: stronger falloff (higher = more aggressive)
    """
    img = img_rgba.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    rgb = arr[:, :, :3]
    a   = arr[:, :, 3].astype(np.uint8)

    # Build an "edge band": pixels that are non-transparent but within band_px of transparency
    trans = (a == 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1))
    near_trans = cv2.dilate(trans, k, iterations=1).astype(bool)
    edge_band = near_trans & (a > 0)

    if not np.any(edge_band):
        return img

    # LAB distance to bg
    bg = np.array(bg_rgb, dtype=np.uint8).reshape(1, 1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
    bg_lab = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB)[0, 0].astype(np.int16)
    d = lab - bg_lab[None, None, :]
    dist = np.sqrt((d[:, :, 0] ** 2) + (d[:, :, 1] ** 2) + (d[:, :, 2] ** 2)).astype(np.float32)

    # Only adjust alpha in edge band where color is close to bg
    m = edge_band & (dist < float(dist_thresh))
    if not np.any(m):
        return img

    # Scale alpha down based on distance (closer to bg => more transparent)
    scale = np.clip(dist / float(dist_thresh), 0.0, 1.0) ** gamma
    a2 = a.astype(np.float32)
    a2[m] = a2[m] * scale[m]

    # If extremely close to bg, kill it fully
    a2[edge_band & (dist < float(dist_thresh) * 0.45)] = 0

    arr[:, :, 3] = np.clip(a2, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def soften_alpha_edge(img_rgba: Image.Image, radius_px: int = 1):
    """
    Smooth alpha only where it's partially transparent (edge band).
    Helps reduce tiny jaggies without blurring solid areas.
    """
    img = img_rgba.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    a = arr[:, :, 3]

    edge = (a > 0) & (a < 255)
    if not edge.any() or radius_px <= 0:
        return img

    k = radius_px * 2 + 1
    if k % 2 == 0:
        k += 1

    blurred = cv2.GaussianBlur(a.astype(np.float32), (k, k), 0)
    a2 = a.astype(np.float32)
    a2[edge] = blurred[edge]

    arr[:, :, 3] = np.clip(a2, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


# -----------------------------
# ANALYSIS (auto decision) - CORNERS MODE
# -----------------------------
def _estimate_bg_color_corners_mode(rgb_u8: np.ndarray, corner_px: int):
    h, w = rgb_u8.shape[:2]
    cs = int(max(3, min(corner_px, h // 3, w // 3)))

    c1 = rgb_u8[0:cs, 0:cs, :].reshape(-1, 3)
    c2 = rgb_u8[0:cs, w - cs:w, :].reshape(-1, 3)
    c3 = rgb_u8[h - cs:h, 0:cs, :].reshape(-1, 3)
    c4 = rgb_u8[h - cs:h, w - cs:w, :].reshape(-1, 3)

    px = np.concatenate([c1, c2, c3, c4], axis=0)

    q = (px // 16).astype(np.int16)
    keys = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    mode_key = int(vals[np.argmax(counts)])

    r = (mode_key >> 8) & 0xFF
    g = (mode_key >> 4) & 0x0F
    bq = mode_key & 0x0F

    bg = np.array([r, g, bq], dtype=np.float32) * 16.0 + 8.0
    return bg.astype(np.uint8)


def analyze_image_for_bg_removal(img: Image.Image):
    img_rgb = img.convert("RGB")
    data = np.array(img_rgb, dtype=np.uint8)
    h, w = data.shape[:2]

    corner_px = max(6, min(32, h // 12, w // 12))
    bg = _estimate_bg_color_corners_mode(data, corner_px=corner_px)

    lab = cv2.cvtColor(data, cv2.COLOR_RGB2LAB).astype(np.int16)
    bg_lab = cv2.cvtColor(bg.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(np.int16)
    d = lab - bg_lab[None, None, :]
    dist = np.sqrt((d[:, :, 0] ** 2) + (d[:, :, 1] ** 2) + (d[:, :, 2] ** 2)).astype(np.float32)

    bg_coverage = float(np.mean(dist < 22.0))

    small = cv2.resize(data, (max(32, w // 10), max(32, h // 10)), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3)
    unique_colors = len(np.unique(pixels, axis=0))
    color_complexity = float(unique_colors / max(1, pixels.shape[0]))

    gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    edge_sharpness = float(np.mean(mag))

    has_solid_bg = bg_coverage >= 0.20
    is_graphic = bool(has_solid_bg or (edge_sharpness > 35.0))

    return {
        "has_solid_bg": has_solid_bg,
        "bg_color": tuple(int(x) for x in bg),
        "bg_coverage": bg_coverage,
        "is_graphic": is_graphic,
        "color_complexity": color_complexity,
        "edge_sharpness": edge_sharpness,
    }


# -----------------------------
# TRIM
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


# -----------------------------
# DECONTAMINATION + EDGE FEATHER
# -----------------------------
def _decontaminate_edges(img_rgba: Image.Image, bg_rgb_u8):
    """
    Unmix RGB from background on semi-transparent edge pixels:
      observed = fg*a + bg*(1-a)  ->  fg = (observed - bg*(1-a)) / a
    Removes halos WITHOUT dilating/painting pixels.
    """
    bg = np.array(bg_rgb_u8, dtype=np.float32).reshape(1, 1, 3)

    data = np.array(img_rgba.convert("RGBA"), dtype=np.float32)
    rgb = data[:, :, :3]
    a = data[:, :, 3:4] / 255.0

    mask = (a > 0.0) & (a < 1.0)
    rgb_unmixed = (rgb - (1.0 - a) * bg) / np.clip(a, 1e-3, 1.0)
    rgb_unmixed = np.clip(rgb_unmixed, 0, 255)

    rgb = np.where(mask, rgb_unmixed, rgb)
    rgb = np.where(a == 0.0, 0.0, rgb)

    data[:, :, :3] = rgb
    return Image.fromarray(data.astype(np.uint8), "RGBA")


def refine_edges(img: Image.Image, feather_amount: int = 2) -> Image.Image:
    img = img.convert("RGBA")
    data = np.array(img, dtype=np.uint8)
    alpha = data[:, :, 3].astype(np.uint8)

    edge_mask = (alpha > 0) & (alpha < 255)
    if edge_mask.any() and feather_amount > 0:
        k = feather_amount * 2 + 1
        blurred = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha[edge_mask] = blurred[edge_mask]
        data[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)

    return Image.fromarray(data, "RGBA")


# -----------------------------
# BG COLOR ESTIMATION: BORDER MODE (backup)
# -----------------------------
def _estimate_bg_color_border_mode(rgb_u8: np.ndarray, border_px: int):
    h, w = rgb_u8.shape[:2]
    b = int(max(2, min(border_px, h // 4, w // 4)))

    top = rgb_u8[0:b, :, :].reshape(-1, 3)
    bottom = rgb_u8[h - b:h, :, :].reshape(-1, 3)
    left = rgb_u8[:, 0:b, :].reshape(-1, 3)
    right = rgb_u8[:, w - b:w, :].reshape(-1, 3)

    border = np.concatenate([top, bottom, left, right], axis=0)

    q = (border // 16).astype(np.int16)
    keys = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    mode_key = int(vals[np.argmax(counts)])

    r = (mode_key >> 8) & 0xFF
    g = (mode_key >> 4) & 0x0F
    bq = mode_key & 0x0F

    bg = np.array([r, g, bq], dtype=np.float32) * 16.0 + 8.0
    return bg.astype(np.uint8)


# -----------------------------
# BG REMOVAL: COLOR (solid bg) - V3
# -----------------------------
def _lab_dist(rgb_u8: np.ndarray, bg_rgb_u8: np.ndarray):
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.int16)
    bg_lab = cv2.cvtColor(bg_rgb_u8.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(np.int16)
    d = lab - bg_lab[None, None, :]
    dist = np.sqrt((d[:, :, 0] ** 2) + (d[:, :, 1] ** 2) + (d[:, :, 2] ** 2)).astype(np.float32)
    return lab, dist


def remove_bg_color_method_v3(img: Image.Image, bg_color=None, tolerance=16):
    """
    Solid-bg removal that won't eat dark colored strokes on black backgrounds.
    Background is only border-connected. Includes soft edge + decontamination.
    """
    pil = img.convert("RGBA")
    rgba = np.array(pil, dtype=np.uint8)
    rgb = rgba[:, :, :3]
    h, w = rgb.shape[:2]

    border_px = max(6, min(24, h // 15, w // 15))

    if bg_color is None:
        bg_rgb = _estimate_bg_color_border_mode(rgb, border_px)
    else:
        bg_rgb = np.array(bg_color, dtype=np.uint8)

    tol = int(np.clip(tolerance, 8, 26))

    lab, dist = _lab_dist(rgb, bg_rgb.astype(np.uint8))

    L = lab[:, :, 0].astype(np.int16)
    aa = lab[:, :, 1].astype(np.int16) - 128
    bb = lab[:, :, 2].astype(np.int16) - 128
    chroma = np.sqrt((aa.astype(np.float32) ** 2) + (bb.astype(np.float32) ** 2))

    bg_sum = int(bg_rgb[0]) + int(bg_rgb[1]) + int(bg_rgb[2])
    near_black = bg_sum <= 60
    near_white = bg_sum >= (255 * 3 - 60)

    if near_black:
        cand = (L <= 55) & (chroma <= 16.0)
    elif near_white:
        cand = (L >= 200) & (chroma <= 18.0)
    else:
        cand = dist <= float(tol)

    cand_u8 = cand.astype(np.uint8)

    num, labels = cv2.connectedComponents(cand_u8, connectivity=8)
    if num <= 1:
        out = pil.copy()
        meta = {"bg_rgb": tuple(int(x) for x in bg_rgb), "tolerance": tol, "reason": "no_components"}
        return out, meta

    border = np.zeros((h, w), dtype=bool)
    b = border_px
    border[0:b, :] = True
    border[h - b:h, :] = True
    border[:, 0:b] = True
    border[:, w - b:w] = True

    border_labels = np.unique(labels[border])
    border_labels = border_labels[border_labels != 0]

    bg_mask = np.isin(labels, border_labels)

    alpha = np.full((h, w), 255, dtype=np.uint8)
    alpha[bg_mask] = 0
    # NEW: remove interior holes inside letters (bg-like but not border-connected)
    alpha = remove_inner_bg_holes(alpha, cand, max_area_ratio=0.02)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bg_dil = cv2.dilate(bg_mask.astype(np.uint8), k, iterations=1).astype(bool)
    edge_zone = bg_dil & (~bg_mask)

    feather = 12.0
    if not (near_black or near_white):
        soft = edge_zone & (dist < (float(tol) + feather))
        a_soft = ((dist[soft] - float(tol)) / feather) * 255.0
        alpha[soft] = np.clip(a_soft, 0, 255).astype(np.uint8)
    else:
        if near_black:
            soft = edge_zone & (L < 80)
            a_soft = ((L[soft].astype(np.float32) - 55.0) / 25.0) * 255.0
        else:
            soft = edge_zone & (L > 175)
            a_soft = ((200.0 - L[soft].astype(np.float32)) / 25.0) * 255.0
        alpha[soft] = np.clip(a_soft, 0, 255).astype(np.uint8)

    out = rgba.copy()
    out[:, :, 3] = alpha
    out_img = Image.fromarray(out, "RGBA")

    out_img = _decontaminate_edges(out_img, bg_rgb)

    meta = {
        "bg_rgb": tuple(int(x) for x in bg_rgb),
        "tolerance": tol,
        "near_black": bool(near_black),
        "near_white": bool(near_white),
        "bg_removed_ratio": float(np.mean(alpha == 0)),
    }
    return out_img, meta


# -----------------------------
# BG REMOVAL: AI (rembg)
# -----------------------------
def _get_rembg_session():
    global REMBG_SESSION
    if REMBG_SESSION is None and new_session is not None:
        REMBG_SESSION = new_session(REMBG_MODEL)
    return REMBG_SESSION


def remove_bg_ai_method(img: Image.Image, is_graphic: bool):
    if rembg_remove is None or new_session is None:
        return img.convert("RGBA"), {"reason": "rembg_unavailable"}

    session = _get_rembg_session()
    pil_in = img.convert("RGBA")

    try:
        if is_graphic:
            out = rembg_remove(pil_in, session=session, post_process_mask=True)
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
# AUTO SCORING (stable selection)
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


def _score_result(img_rgba: Image.Image):
    """
    Higher is better:
    - wants corners transparent
    - wants some content kept
    - avoids "almost everything removed" or "nothing removed"
    """
    data = np.array(img_rgba.convert("RGBA"), dtype=np.uint8)
    a = data[:, :, 3]
    content = float(np.mean(a > 20))
    corner_t = _corners_transparent_ratio(img_rgba)

    penalty = 0.0
    if content < 0.02:
        penalty += (0.02 - content) * 4.0
    if content > 0.98:
        penalty += (content - 0.98) * 3.0

    return (corner_t * 2.5) + (content * 0.5) - penalty


def remove_bg_auto_v3(img: Image.Image, analysis: dict):
    """
    AUTO:
    - If bg_coverage looks dominant (>= 0.20) or it's graphic: prefer color remover
    - Otherwise use AI
    """
    is_graphic = bool(analysis.get("is_graphic", False))
    bg_coverage = float(analysis.get("bg_coverage", 0.0))

    if is_graphic or bg_coverage >= 0.20:
        best = None
        best_meta = None
        best_score = -1e9

        for tol in (10, 12, 14, 16, 18, 20, 22):
            out, meta = remove_bg_color_method_v3(
                img,
                bg_color=analysis.get("bg_color"),
                tolerance=tol
            )
            removed_ratio = float(meta.get("bg_removed_ratio", 0.0))
            if removed_ratio < 0.05:
                continue

            sc = _score_result(out)
            if sc > best_score:
                best = out
                best_meta = meta
                best_score = sc

        if best is None:
            best, best_meta = remove_bg_color_method_v3(img, bg_color=analysis.get("bg_color"), tolerance=16)

        corner_t = _corners_transparent_ratio(best)
        if corner_t < 0.70:
            ai, ai_meta = remove_bg_ai_method(img, is_graphic=is_graphic)
            return ai, "ai_rembg_fallback", True, {"picked": "ai", "ai_meta": ai_meta, "color_meta": best_meta}

        return best, "color_v3_auto_pick", False, {"picked": "color", "color_meta": best_meta, "score": best_score}

    ai, ai_meta = remove_bg_ai_method(img, is_graphic=is_graphic)
    return ai, "ai_rembg_auto", False, {"picked": "ai", "ai_meta": ai_meta}


# -----------------------------
# HELPER FUNCTIONS for /process-logo and /smart-print-ready
# -----------------------------

def determine_logo_type(img):
    """
    Analyze logo to determine if it's black-ish or white-ish
    Returns: ("black" or "white", dark_ratio, light_ratio)
    """
    img_rgba = img.convert('RGBA')
    data = np.array(img_rgba)
    
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # Only consider visible pixels (alpha > 10)
    visible_mask = a > 10
    
    if not np.any(visible_mask):
        return "black", 0.0, 0.0
    
    # Calculate luminance for visible pixels
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    visible_lum = luminance[visible_mask]
    
    # Define thresholds
    dark_threshold = 100
    light_threshold = 200
    
    # Count dark and light pixels
    dark_pixels = np.sum(visible_lum < dark_threshold)
    light_pixels = np.sum(visible_lum > light_threshold)
    total_visible = visible_mask.sum()
    
    dark_ratio = dark_pixels / total_visible if total_visible > 0 else 0
    light_ratio = light_pixels / total_visible if total_visible > 0 else 0
    
    # Determine type based on ratios
    if dark_ratio > light_ratio:
        return "black", dark_ratio, light_ratio
    elif light_ratio > dark_ratio:
        return "white", dark_ratio, light_ratio
    else:
        # If equal or both low, check average luminance
        avg_lum = np.mean(visible_lum)
        if avg_lum < 128:
            return "black", dark_ratio, light_ratio
        else:
            return "white", dark_ratio, light_ratio


def generate_variant(img, current_type):
    """
    Generate color variant (opposite of current type)
    """
    img_rgba = img.convert('RGBA')
    data = np.array(img_rgba)
    h, w = data.shape[:2]
    
    r = data[:, :, 0].astype(np.uint8)
    g = data[:, :, 1].astype(np.uint8)
    b = data[:, :, 2].astype(np.uint8)
    a = data[:, :, 3].astype(np.uint8)
    
    # Logo mask (visible pixels)
    alpha_min = 10
    logo_mask = a > alpha_min
    
    if not np.any(logo_mask):
        return img_rgba
    
    # Thresholds
    dark_thr = 100
    white_cut = 220
    
    # Detect dark and white pixels in logo
    blackish = logo_mask & (r < dark_thr) & (g < dark_thr) & (b < dark_thr)
    whiteish = logo_mask & (r > white_cut) & (g > white_cut) & (b > white_cut)
    
    dark_px = int(np.sum(blackish))
    white_px = int(np.sum(whiteish))
    logo_px = int(np.sum(logo_mask))
    
    dark_ratio = dark_px / max(1, logo_px)
    white_ratio = white_px / max(1, logo_px)
    
    changed = False
    
    # Apply transformation based on detected type
    if current_type == "black" and dark_ratio > 0.1:
        # Convert dark to white
        data[blackish, 0] = 255
        data[blackish, 1] = 255
        data[blackish, 2] = 255
        changed = True
    elif current_type == "white" and white_ratio > 0.1:
        # Convert white to black
        data[whiteish, 0] = 0
        data[whiteish, 1] = 0
        data[whiteish, 2] = 0
        changed = True
    
    # If no significant change, invert logo colors
    if not changed:
        data[logo_mask, 0] = 255 - data[logo_mask, 0]
        data[logo_mask, 1] = 255 - data[logo_mask, 1]
        data[logo_mask, 2] = 255 - data[logo_mask, 2]
    
    return Image.fromarray(data, 'RGBA')


def _ensure_ftp_dir(ftp, path):
    """
    Ensure path exists on FTP server and cwd into it
    """
    if not path:
        return
    
    path = path.strip()
    is_abs = path.startswith("/")
    parts = [p for p in path.strip("/").split("/") if p]
    
    if is_abs:
        try:
            ftp.cwd("/")
        except ftplib.error_perm:
            pass
    
    for part in parts:
        try:
            ftp.cwd(part)
        except ftplib.error_perm:
            ftp.mkd(part)
            ftp.cwd(part)


def upload_images_to_ftp(images_dict, folder_id):
    """
    Upload multiple images to FTP
    Returns: (urls_dict, status_message)
    """
    if not all([FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_URL]):
        return None, "FTP not configured (set FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_URL)"
    
    urls = {}
    try:
        ftp = ftplib.FTP(FTP_HOST, timeout=30)
        ftp.login(FTP_USER, FTP_PASS)
        
        _ensure_ftp_dir(ftp, FTP_DIR)
        _ensure_ftp_dir(ftp, folder_id)
        
        base_url = FTP_BASE_URL.rstrip("/")
        
        for filename, pil_img in images_dict.items():
            img_buffer = BytesIO()
            pil_img.save(img_buffer, format="PNG", optimize=True)
            img_buffer.seek(0)
            
            ftp.storbinary(f"STOR {filename}", img_buffer)
            urls[filename] = f"{base_url}/{folder_id}/{filename}"
        
        ftp.quit()
        return urls, "success"
    
    except Exception as e:
        try:
            ftp.quit()
        except Exception:
            pass
        return None, str(e)


def get_folder_id_from_request():
    """
    Get folder_id from request (form, query param, or header)
    Falls back to uuid4 if missing
    """
    raw = (
        (request.form.get("folder_id") if request.form else None)
        or request.args.get("folder_id")
        or request.headers.get("X-Folder-Id")
        or ""
    ).strip()
    
    if not raw:
        return str(uuid.uuid4())
    
    # Sanitize for FTP safety
    safe = raw.replace(" ", "-")
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", safe)
    
    if not safe:
        return str(uuid.uuid4())
    
    return safe[:80]


# -----------------------------
# /remove-bg (ONLY endpoint)
# -----------------------------
@app.route("/remove-bg", methods=["POST"])
def remove_bg_endpoint():
    """
    multipart/form-data:
      - image: file (required)

    optional:
      - enhance: true/false (default false)
      - trim: true/false (default true)
      - output_format: png/webp (default png)
      - bg_remove: auto/ai/color/skip (default auto)
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
    auth_error = verify_api_key()
    if auth_error:
        resp, status = auth_error
        resp.status_code = status
        log("auth_check", success=False, reason="invalid_or_missing_api_key")
        return attach_logs_to_response(resp)
    log("auth_check", success=True)

    # enforce image-only
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

        do_enhance = request.form.get("enhance", "false").lower() == "true"
        do_trim = request.form.get("trim", "true").lower() == "true"
        output_format = request.form.get("output_format", "png").strip().lower()
        do_enhance_second = request.form.get("enhance_second", "false").lower() == "true"
        enhance_second_mode = request.form.get("enhance_second_mode", "fal").strip().lower()
        # allowed: fal | skip_fal | local
        if enhance_second_mode not in ("fal", "skip_fal", "local"):
            enhance_second_mode = "fal"

        bg_remove = request.form.get("bg_remove", "auto").strip().lower()
        allowed_bg = {"auto", "ai", "color", "skip"}
        if bg_remove not in allowed_bg:
            log("read_params", success=False, reason="invalid_bg_remove", received=bg_remove)
            return json_error({"error": "Invalid bg_remove value", "allowed": sorted(list(allowed_bg))}, status=400)

        log("read_params", success=True,
            enhance=do_enhance, enhance_second=do_enhance_second,
            trim=do_trim, output_format=output_format, bg_remove=bg_remove
        )

        # already transparent?
        try:
            already_transparent = has_transparency(img_bytes)
            log("check_transparency", success=True, already_transparent=already_transparent)
        except Exception as e:
            already_transparent = False
            log("check_transparency", success=False, error=str(e), already_transparent=False)

        # optional enhance
        enhanced = False
        enhance_msg = "not requested"
        if do_enhance:
            enhanced_bytes, enhanced, enhance_msg = enhance_image_fal(img_bytes)
            if enhanced and enhanced_bytes:
                img_bytes = enhanced_bytes
            log("enhance_fal", success=True, applied=bool(enhanced), message=str(enhance_msg))

        # decode
        img = _open_image_bytes(img_bytes)
        log("decode_image", success=True, mode=img.mode, size=f"{img.width}x{img.height}")

        analysis = analyze_image_for_bg_removal(img)
        log("analyze", success=True, **analysis)

        method_used = "skip"
        fallback_used = False
        meta = {}

        if already_transparent:
            result_img = img.convert("RGBA")
            method_used = "skip_already_transparent"
        else:
            if bg_remove == "skip":
                result_img = img.convert("RGBA")
                method_used = "skip_requested"

            elif bg_remove == "color":
                result_img, meta = remove_bg_color_method_v3(img, bg_color=analysis.get("bg_color"), tolerance=16)
                method_used = "color_v3_forced"

            elif bg_remove == "ai":
                result_img, meta = remove_bg_ai_method(img, is_graphic=analysis.get("is_graphic", False))
                method_used = "ai_rembg_forced"

            else:
                result_img, method_used, fallback_used, meta = remove_bg_auto_v3(img, analysis)

        log("bg_removed", success=True, method_used=method_used, fallback_used=fallback_used, meta=meta)

        # feather (small)
        result_img = refine_edges(result_img, feather_amount=2)
        log("refine_edges", success=True)

        # sharpen only when enhanced (upscale softens edges)
        if enhanced:
            result_img = sharpen_rgb_keep_alpha(result_img, amount=1.10, radius=1.2, threshold=3)
            log("sharpen", success=True, amount=1.10, radius=1.2, threshold=3)

        # 1) edge spill cleanup (alpha fix near transparency only)
        bg_rgb = analysis.get("bg_color") or (255, 255, 255)
        result_img = cleanup_edge_spill(result_img, bg_rgb=bg_rgb, band_px=2, dist_thresh=26, gamma=1.6)
        log("cleanup_edge_spill", success=True, band_px=2, dist_thresh=26, gamma=1.6)

        # 2) then decontaminate (RGB fix)
        result_img = _decontaminate_edges(result_img, bg_rgb)
        log("decontaminate_final", success=True, bg_rgb=bg_rgb)

        # 3) optional tiny alpha smoothing
        result_img = soften_alpha_edge(result_img, radius_px=1)
        log("soften_alpha_edge", success=True, radius_px=1)

        enhanced_second = False
        enhance_second_msg = "not requested"

        if do_enhance_second:
            before_second = result_img.convert("RGBA")

            if enhance_second_mode == "skip_fal":
                # TEST: don't change image, just run the 2nd-pass cleanup steps
                enhanced_second = False
                enhance_second_msg = "skip_fal_test_mode"
                enh_img = before_second

            elif enhance_second_mode == "local":
                # TEST: local upscale instead of fal (closest offline test)
                enh_img, up_meta = local_upscale_for_logo(before_second, target_max=1200)
                enhanced_second = bool(up_meta.get("applied", False))
                enhance_second_msg = f"local_upscale:{up_meta}"
                log("enhance_local_second", success=True, **up_meta)

            else:
                # REAL: call fal
                tmp_bytes = pil_to_png_bytes(before_second)
                second_bytes, enhanced_second, enhance_second_msg = enhance_image_fal(tmp_bytes)

                if enhanced_second and second_bytes:
                    enh_img = _open_image_bytes(second_bytes)
                else:
                    enh_img = before_second

            # Apply the same helper pipeline (even in skip/local mode)
            result_img = restore_alpha_if_missing(enh_img, before_second)

            bg_rgb = analysis.get("bg_color") or (255, 255, 255)
            result_img = cleanup_edge_spill(result_img, bg_rgb=bg_rgb, band_px=2, dist_thresh=26, gamma=1.6)
            log("cleanup_edge_spill_2", success=True, band_px=2, dist_thresh=26, gamma=1.6)

            result_img = _decontaminate_edges(result_img, bg_rgb)
            log("decontaminate_final_2", success=True, bg_rgb=bg_rgb)

            result_img = soften_alpha_edge(result_img, radius_px=1)
            log("soften_alpha_edge_2", success=True, radius_px=1)

            log("enhance_second", success=True, mode=enhance_second_mode, applied=bool(enhanced_second), message=str(enhance_second_msg),
                out_size=f"{result_img.width}x{result_img.height}")



        # trim
        if do_trim:
            result_img = trim_transparent(result_img, padding=2)
            log("trim", success=True, out_size=f"{result_img.width}x{result_img.height}")
        else:
            log("trim", success=True, skipped=True)

        # encode
        out = BytesIO()
        if output_format == "webp":
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
        resp.headers["X-Enhanced"] = str(bool(enhanced))
        resp.headers["X-Enhance-Status"] = str(enhance_msg)
        resp.headers["X-Enhanced-Second"] = str(bool(enhanced_second))
        resp.headers["X-Enhance-Second-Status"] = str(enhance_second_msg)
        resp.headers["X-Trimmed"] = str(bool(do_trim))
        resp.headers["X-Processing-Time"] = f"{processing_time:.2f}s"
        resp.headers["X-Output-Size"] = f"{result_img.width}x{result_img.height}"

        return attach_logs_to_response(resp)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log("exception", success=False, error=str(e))
        return json_error({"error": "Processing failed", "details": str(e), "traceback": tb}, status=500)


# -----------------------------
# /process-logo endpoint
# -----------------------------
@app.route('/process-logo', methods=['POST'])
def process_logo():
    """
    LOGO PROCESSING PIPELINE - With FTP Upload
    
    Accepts: image (file) - required
    
    Pipeline:
    1. Load image (expects transparent background already)
    2. Determine logo type (black-ish or white-ish)
    3. Generate 2 versions:
       - original_{type}: The original
       - original_{opposite}: Color variant (inverted)
    4. Create unique folder and upload both to FTP
    
    Returns: JSON with folder_id and image URLs
    
    Required Environment Variables for FTP:
        - FTP_HOST: FTP server hostname
        - FTP_USER: FTP username
        - FTP_PASS: FTP password
        - FTP_DIR: Base remote directory (default: /logos)
        - FTP_BASE_URL: Public URL base (e.g., https://cdn.example.com)
    """
    
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Check FTP configuration
        if not all([FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_URL]):
            return jsonify({
                "error": "FTP not configured",
                "message": "Please set FTP_HOST, FTP_USER, FTP_PASS, and FTP_BASE_URL environment variables"
            }), 500
        
        # Read image bytes
        image_bytes = file.read()
        
        # Get folder_id from request
        folder_id = get_folder_id_from_request()
        
        # Track processing steps
        processing_log = []
        
        # STEP 1: Load image
        img = Image.open(BytesIO(image_bytes))
        img = img.convert('RGBA')
        
        processing_log.append({
            "step": "load_image",
            "success": True,
            "size": f"{img.width}x{img.height}"
        })
        
        # STEP 2: Determine logo type
        logo_type, dark_ratio, light_ratio = determine_logo_type(img)
        opposite_type = "white" if logo_type == "black" else "black"
        
        processing_log.append({
            "step": "analyze",
            "detected_type": logo_type,
            "dark_ratio": f"{dark_ratio:.4f}",
            "light_ratio": f"{light_ratio:.4f}"
        })
        
        # STEP 3: Generate 2 versions
        original_key = f"original_{logo_type}"
        original_img = img.copy()
        
        variant_key = f"original_{opposite_type}"
        variant_img = generate_variant(img, logo_type)
        
        processing_log.append({
            "step": "generate_versions",
            "success": True,
            "versions": [original_key, variant_key]
        })
        
        # STEP 4: Upload to FTP
        images_to_upload = {
            f"{original_key}.png": original_img,
            f"{variant_key}.png": variant_img
        }
        
        urls, ftp_status = upload_images_to_ftp(images_to_upload, folder_id)
        
        if urls is None:
            return jsonify({
                "error": "FTP upload failed",
                "message": ftp_status,
                "processing_log": processing_log
            }), 500
        
        processing_log.append({
            "step": "ftp_upload",
            "success": True,
            "folder_id": folder_id,
            "files_uploaded": len(urls)
        })
        
        # OUTPUT: JSON with URLs
        return jsonify({
            "success": True,
            "folder_id": folder_id,
            "detected_type": logo_type,
            "processing_log": processing_log,
            "images": {
                original_key: {
                    "description": f"Original logo ({logo_type})",
                    "url": urls[f"{original_key}.png"]
                },
                variant_key: {
                    "description": f"Color variant ({opposite_type})",
                    "url": urls[f"{variant_key}.png"]
                }
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


# -----------------------------
# /smart-print-ready endpoint
# -----------------------------
@app.route('/smart-print-ready', methods=['POST'])
def smart_print_ready():
    """
    SMART PRINT-READY (Hybrid v7)
    - Converts logo to print-ready grayscale (white or black scheme) with transparency preserved.
    - Handles mixed logos: per connected-component decide Gradient vs Stepped Layers.
    - Keeps your “group/layer” behavior for solid regions while avoiding ugly banding for gradients.

    Params (form-data):
    - image: file (required)
    - print_color: 'white' | 'black' (required)
    - layers: 'auto' | 2..6 (default: 'auto')         # used for STEPPED components
    - white_step: 5..30 (default: 10)                 # used for STEPPED components
    - black_step: 15..50 (default: 33)                # used for STEPPED components
    - gradient_mode: 'auto' | 'smooth' | 'stepped' (default: 'auto')
        auto   = detect per component
        smooth = force all components smooth
        stepped= force all components stepped
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        print_color = request.form.get('print_color', '').lower().strip()
        num_layers = request.form.get('layers', 'auto')
        white_step = int(request.form.get('white_step', 10))
        black_step = int(request.form.get('black_step', 33))
        gradient_mode = request.form.get('gradient_mode', 'auto').lower().strip()

        if print_color not in ['white', 'black']:
            return jsonify({
                "error": "print_color is required",
                "usage": "print_color=white (for dark shirts) or print_color=black (for light shirts)"
            }), 400

        white_step = max(5, min(30, white_step))
        black_step = max(15, min(50, black_step))
        if gradient_mode not in ['auto', 'smooth', 'stepped']:
            gradient_mode = 'auto'

        # ---------------------------
        # Load image
        # ---------------------------
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        height, width = data.shape[:2]

        r = data[:, :, 0].astype(np.uint8)
        g = data[:, :, 1].astype(np.uint8)
        b = data[:, :, 2].astype(np.uint8)
        a = data[:, :, 3].astype(np.uint8)
        original_alpha = a.copy()

        # ============================================================
        # STEP 1: BACKGROUND DETECTION (your logic, kept)
        # ============================================================
        corners = [
            data[0, 0],
            data[0, width - 1],
            data[height - 1, 0],
            data[height - 1, width - 1]
        ]

        corner_colors = np.array([c[:3] for c in corners], dtype=np.float32)
        corner_alphas = np.array([c[3] for c in corners], dtype=np.float32)

        avg_corner = np.mean(corner_colors, axis=0)
        avg_alpha = float(np.mean(corner_alphas))
        corner_std = float(np.std(corner_colors))

        corners_consistent = corner_std < 30

        if avg_alpha < 128:
            bg_type = "transparent"
            bg_mask = original_alpha < 10
        elif corners_consistent and float(np.mean(avg_corner)) > 240:
            bg_type = "white"
            bg_mask = (r > 250) & (g > 250) & (b > 250) & (original_alpha > 200)
        elif corners_consistent and float(np.mean(avg_corner)) < 15:
            bg_type = "black"
            bg_mask = (r < 5) & (g < 5) & (b < 5) & (original_alpha > 200)
        elif corners_consistent:
            bg_type = "colored"
            tolerance = 20
            bg_mask = (
                (np.abs(r.astype(np.int16) - int(avg_corner[0])) < tolerance) &
                (np.abs(g.astype(np.int16) - int(avg_corner[1])) < tolerance) &
                (np.abs(b.astype(np.int16) - int(avg_corner[2])) < tolerance) &
                (original_alpha > 200)
            )
        else:
            bg_type = "mixed/none"
            bg_mask = original_alpha < 10

        # Always treat near-transparent as background
        bg_mask = bg_mask | (original_alpha < 10)

        # Flood fill from corners for non-transparent backgrounds
        if bg_type not in ["transparent", "mixed/none"]:
            potential_bg = (bg_mask.astype(np.uint8) * 255)
            connected_bg = np.zeros_like(potential_bg)

            for start_y, start_x in [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]:
                if potential_bg[start_y, start_x] > 0:
                    temp = potential_bg.copy()
                    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
                    cv2.floodFill(temp, flood_mask, (start_x, start_y), 128)
                    connected_bg[temp == 128] = 255

            bg_mask = connected_bg > 0

        # Logo mask (non-background, but keep alpha relevance)
        logo_mask = (~bg_mask) & (original_alpha > 10)

        if int(np.sum(logo_mask)) == 0:
            # fallback: alpha-only
            logo_mask = original_alpha > 10
            bg_mask = ~logo_mask
            bg_type = "fallback-transparent-only"

        # ============================================================
        # STEP 2: LUMINANCE
        # ============================================================
        luminance = (0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32))

        logo_luminance = luminance[logo_mask]
        if logo_luminance.size == 0:
            return jsonify({"error": "No logo pixels found"}), 400

        lum_min = float(np.min(logo_luminance))
        lum_max = float(np.max(logo_luminance))
        lum_range = float(lum_max - lum_min)

        # ============================================================
        # STEP 3+4: HYBRID PER CONNECTED COMPONENT
        # ============================================================
        result = np.zeros((height, width, 4), dtype=np.uint8)
        result[bg_mask] = [0, 0, 0, 0]

        # Label connected components on logo pixels
        mask_u8 = (logo_mask.astype(np.uint8) * 255)
        num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        def _parse_layers(val):
            if isinstance(val, str) and val.lower() == 'auto':
                return 'auto'
            try:
                return max(2, min(6, int(val)))
            except:
                return 'auto'

        layers_setting = _parse_layers(num_layers)

        # --- gradient detection per component ---
        def is_gradient_component(comp_bool: np.ndarray) -> bool:
            # Respect explicit mode
            if gradient_mode == 'smooth':
                return True
            if gradient_mode == 'stepped':
                return False

            # Remove edges (anti-alias) to avoid false detection
            k = np.ones((3, 3), np.uint8)
            interior = cv2.erode(comp_bool.astype(np.uint8), k, iterations=1).astype(bool)
            if interior.sum() < 80:
                interior = comp_bool  # fallback if too thin

            lum_vals = luminance[interior].astype(np.float32)
            if lum_vals.size < 140:
                return False

            # Histogram entropy (smooth gradients -> higher entropy, less dominance)
            hist = np.histogram(lum_vals, bins=32)[0].astype(np.float32)
            p = hist / (hist.sum() + 1e-9)
            entropy = float(-np.sum(p * np.log(p + 1e-9)))
            norm_entropy = entropy / float(np.log(len(p)))
            dom_mass = float(np.sort(p)[-3:].sum())     # top-3 bin mass
            occupancy = int(np.sum(hist > 0))

            # Linear fit: lum ≈ ax + by + c (gradients often fit well)
            ys, xs = np.where(interior)
            X = np.column_stack([xs.astype(np.float32), ys.astype(np.float32), np.ones(xs.size, np.float32)])
            yv = lum_vals
            coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
            pred = X @ coef
            ss_res = float(np.sum((yv - pred) ** 2))
            ss_tot = float(np.sum((yv - float(yv.mean())) ** 2)) + 1e-9
            r2 = 1.0 - (ss_res / ss_tot)

            lum_std = float(lum_vals.std())
            lum_rng = float(lum_vals.max() - lum_vals.min())

            # Decision rules:
            entropy_gradient = (norm_entropy > 0.70 and dom_mass < 0.55 and occupancy > 10 and lum_rng > 20)
            linear_gradient = (r2 > 0.60 and lum_std > 8 and lum_rng > 20)

            return bool(entropy_gradient or linear_gradient)

        # Smooth gradient output range (more “solid” by default)
        # If you want even MORE solid gradients: white 255..235, black 0..25
        if print_color == 'white':
            grad_out_max, grad_out_min = 255, 200
        else:
            grad_out_min, grad_out_max = 0, 70

        stepped_components = 0
        gradient_components = 0
        used_gray_values = set()

        # Process each component
        for lab in range(1, num_cc):
            area = int(cc_stats[lab, cv2.CC_STAT_AREA])
            if area < 25:
                continue

            comp = (cc_labels == lab) & logo_mask
            if int(np.sum(comp)) == 0:
                continue

            component_is_grad = is_gradient_component(comp)
            if component_is_grad:
                gradient_components += 1
            else:
                stepped_components += 1

            ys, xs = np.where(comp)
            comp_lum = luminance[ys, xs].astype(np.float32)
            comp_alpha = original_alpha[ys, xs].astype(np.uint8)

            if component_is_grad:
                # -------- Smooth mapping (vectorized) --------
                if lum_range > 1e-6:
                    norm = (comp_lum - lum_min) / lum_range
                else:
                    norm = np.zeros_like(comp_lum)

                # Optional gentle gamma to reduce harsh contrast
                norm = np.clip(norm, 0.0, 1.0) ** 1.1

                if print_color == 'white':
                    out = grad_out_max - norm * (grad_out_max - grad_out_min)
                else:
                    out = grad_out_min + norm * (grad_out_max - grad_out_min)

                out = np.clip(out, 0, 255).astype(np.uint8)

                result[ys, xs, 0] = out
                result[ys, xs, 1] = out
                result[ys, xs, 2] = out
                result[ys, xs, 3] = comp_alpha

                # track used grays (sampled, to avoid huge sets)
                if out.size > 0:
                    used_gray_values.update(np.unique(out[::max(1, out.size // 5000)]).tolist())

            else:
                # -------- Stepped layers (per component kmeans) --------
                comp_min = float(comp_lum.min())
                comp_max = float(comp_lum.max())
                comp_rng = float(comp_max - comp_min)

                # Decide layers for this component
                if layers_setting == 'auto':
                    if comp_rng < 30:
                        L = 2
                    elif comp_rng < 80:
                        L = 3
                    elif comp_rng < 150:
                        L = 4
                    else:
                        L = 5
                else:
                    L = int(layers_setting)

                # If too flat, keep as single tone
                if comp_rng < 10 or comp_lum.size < L:
                    L = 1
                    ranks = np.zeros(comp_lum.size, dtype=np.int32)
                else:
                    Z = comp_lum.reshape(-1, 1).astype(np.float32)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                    _, labels_k, centers = cv2.kmeans(Z, L, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

                    order = np.argsort(centers.flatten())  # darkest -> lightest
                    lut = np.empty(L, dtype=np.int32)
                    for rank, old in enumerate(order):
                        lut[int(old)] = rank
                    ranks = lut[labels_k.flatten()]

                # Build output gray palette for this component
                if print_color == 'white':
                    grays = np.empty(L, dtype=np.uint8)
                    for rank in range(L):
                        if rank == 0:
                            grays[rank] = 255
                        else:
                            darkness = min(rank * white_step, 50)
                            v = int(255 * (100 - darkness) / 100)
                            grays[rank] = max(128, v)
                else:
                    grays = np.empty(L, dtype=np.uint8)
                    for rank in range(L):
                        if rank == 0:
                            grays[rank] = 0
                        else:
                            lightness = min(rank * black_step, 85)
                            v = int(255 * lightness / 100)
                            grays[rank] = min(220, v)

                out = grays[np.clip(ranks, 0, L - 1)].astype(np.uint8)

                result[ys, xs, 0] = out
                result[ys, xs, 1] = out
                result[ys, xs, 2] = out
                result[ys, xs, 3] = comp_alpha

                used_gray_values.update(grays.tolist())

        # Extra safety fallback
        if int(np.sum(result[:, :, 3] > 0)) == 0:
            ys, xs = np.where(original_alpha > 10)
            base = 255 if print_color == 'white' else 0
            result[ys, xs, 0] = base
            result[ys, xs, 1] = base
            result[ys, xs, 2] = base
            result[ys, xs, 3] = original_alpha[ys, xs]
            bg_type = bg_type + "|fallback-filled"

        # ============================================================
        # STEP 5: OUTPUT
        # ============================================================
        result_img = Image.fromarray(result, 'RGBA')
        output = BytesIO()
        result_img.save(output, format='PNG', optimize=True)
        output.seek(0)

        used_step = white_step if print_color == 'white' else black_step

        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'print_ready_{print_color}.png'
        )

        # Headers for debugging/telemetry
        response.headers['X-Print-Color'] = print_color.upper()
        response.headers['X-Background-Type'] = bg_type
        response.headers['X-Processing-Method'] = "hybrid-components"
        response.headers['X-Gradient-Mode'] = gradient_mode
        response.headers['X-Gradient-Components'] = f"{gradient_components}/{max(1, (gradient_components + stepped_components))}"
        response.headers['X-Stepped-Components'] = str(stepped_components)
        response.headers['X-Step'] = f"{used_step}%"
        response.headers['X-Luminance-Range'] = f"{lum_min:.0f}-{lum_max:.0f}"

        # Keep this short so headers don’t explode
        if used_gray_values:
            sample = sorted(list(used_gray_values))
            if len(sample) > 40:
                sample = sample[:20] + ["..."] + sample[-20:]
            response.headers['X-Used-Grays'] = str(sample)

        return response

    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=DEBUG)