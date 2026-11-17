import math

# approximate real-world heights in metres
REAL_HEIGHT = {
    "cone": 0.70,
    "barrier": 1.00,
    "stop_sign": 0.75,
}

REF_WIDTH = 1280
REF_FOCAL = 1000.0


def focal_from_fov(img_w, hfov_deg):
    return img_w / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def focal_for_image(img_w, focal_px=None, hfov_deg=None):
    if focal_px is not None:
        return focal_px
    if hfov_deg is not None:
        return focal_from_fov(img_w, hfov_deg)
    return REF_FOCAL * img_w / REF_WIDTH


def estimate_distance(cls_name, box_h_px, focal_px):
    # pinhole model: Z = f * H / h
    if box_h_px <= 1 or cls_name not in REAL_HEIGHT:
        return None
    return focal_px * REAL_HEIGHT[cls_name] / box_h_px
