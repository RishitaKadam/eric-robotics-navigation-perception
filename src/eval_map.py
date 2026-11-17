import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="data/data.yaml")
    a = p.parse_args()
    m = YOLO(a.weights).val(data=a.data)
    print("mAP50:", m.box.map50, "mAP50-95:", m.box.map)


if __name__ == "__main__":
    main()
