import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from distance import estimate_distance, focal_for_image

COLORS = {"cone": (0, 140, 255), "barrier": (0, 200, 0), "stop_sign": (0, 0, 255)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def pretty(name):
    return name.replace("_", " ").title()


def annotate(frame, result, focal):
    names = result.names
    for b in result.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        name = names[int(b.cls)]
        d = estimate_distance(name, y2 - y1, focal)
        label = pretty(name) + (f", {d:.1f}m" if d else "")
        color = COLORS.get(name, (255, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    return frame


def run_images(model, paths, out_dir, args):
    for p in paths:
        img = cv2.imread(str(p))
        focal = focal_for_image(img.shape[1], args.focal, args.hfov)
        res = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        cv2.imwrite(str(out_dir / p.name), annotate(img, res, focal))


def run_video(model, path, out_dir, args):
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(str(out_dir / (path.stem + "_out.mp4")),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    focal = focal_for_image(w, args.focal, args.hfov)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        writer.write(annotate(frame, res, focal))
    cap.release()
    writer.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--focal", type=float, default=None, help="focal length in px")
    p.add_argument("--hfov", type=float, default=None, help="horizontal fov in deg")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    src = Path(args.source)

    if src.is_dir():
        run_images(model, sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXT),
                   out_dir, args)
    elif src.suffix.lower() in IMG_EXT:
        run_images(model, [src], out_dir, args)
    else:
        run_video(model, src, out_dir, args)


if __name__ == "__main__":
    main()
