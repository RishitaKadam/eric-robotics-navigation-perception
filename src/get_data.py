import argparse
from roboflow import Roboflow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--out", default="datasets/navobjects")
    a = p.parse_args()

    rf = Roboflow(api_key=a.api_key)
    proj = rf.workspace(a.workspace).project(a.project)
    proj.version(a.version).download("yolov8", location=a.out)


if __name__ == "__main__":
    main()
