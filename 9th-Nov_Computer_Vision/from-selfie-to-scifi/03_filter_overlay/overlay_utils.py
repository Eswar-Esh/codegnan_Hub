import cv2
import numpy as np

def _clip_roi(x, y, w, h, img_w, img_h):
    """Clip a rectangle (x,y,w,h) to image bounds and return clipped coords and offsets into overlay."""
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img_w, x + w), min(img_h, y + h)
    if x0 >= x1 or y0 >= y1:
        return None  # fully out of frame
    # offsets inside the overlay image
    dx0, dy0 = x0 - x, y0 - y
    dx1, dy1 = dx0 + (x1 - x0), dy0 + (y1 - y0)
    return (x0, y0, x1, y1, dx0, dy0, dx1, dy1)


def overlay_image_alpha(dst_bgr, overlay_bgr, alpha_mask, x, y):
    """
    Alpha blend overlay_bgr (H,W,3) onto dst_bgr (H,W,3) at top-left (x,y)
    using alpha_mask (H,W) in range [0..255]. Safe to be partially out of bounds.
    """
    H, W = dst_bgr.shape[:2]
    h, w = overlay_bgr.shape[:2]

    roi = _clip_roi(x, y, w, h, W, H)
    if roi is None:
        return False
    x0, y0, x1, y1, dx0, dy0, dx1, dy1 = roi

    # Crop source and mask to the visible region only
    src = overlay_bgr[dy0:dy1, dx0:dx1]
    mask = alpha_mask[dy0:dy1, dx0:dx1].astype(np.float32) / 255.0
    inv = 1.0 - mask

    dst_slice = dst_bgr[y0:y1, x0:x1].astype(np.float32)
    blended = (dst_slice * inv[..., None]) + (src.astype(np.float32) * mask[..., None])
    dst_bgr[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return True


def add_filter(frame_bgr, filter_img_bgra, pos_xy, size_wh, debug=False):
    """
    Resizes filter_img_bgra to size_wh (w,h) and overlays at pos_xy (x,y).
    Accepts BGRA (with alpha). If no alpha, treats as opaque.
    """
    x, y = pos_xy
    w, h = size_wh

    if w <= 0 or h <= 0:
        if debug:
            print("[overlay] Skipped overlay due to non-positive size:", size_wh)
        return frame_bgr

    # Ensure 4 channels
    if filter_img_bgra is None or filter_img_bgra.ndim != 3:
        if debug:
            print("[overlay] Invalid filter image.")
        return frame_bgr

    if filter_img_bgra.shape[2] == 4:
        b, g, r, a = cv2.split(filter_img_bgra)
        overlay_bgr = cv2.merge([b, g, r])
        alpha = a
    else:
        overlay_bgr = filter_img_bgra
        alpha = np.full(filter_img_bgra.shape[:2], 255, dtype=np.uint8)

    # Resize
    overlay_bgr = cv2.resize(overlay_bgr, (w, h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)

    ok = overlay_image_alpha(frame_bgr, overlay_bgr, alpha, x, y)
    if debug and not ok:
        print("[overlay] Overlay out of bounds (fully). Pos/Size:", pos_xy, size_wh)
    return frame_bgr
