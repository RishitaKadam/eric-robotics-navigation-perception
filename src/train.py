import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--name", default="nav_n")
    a = p.parse_args()

    model = YOLO(a.weights)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        name=a.name,
        freeze=10,
        patience=10,
        project="runs",
    )
    model.val()


if __name__ == "__main__":
    main()
