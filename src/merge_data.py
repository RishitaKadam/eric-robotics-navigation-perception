import argparse
import shutil
from pathlib import Path

import yaml

CLASSES = ["cone", "barrier", "stop_sign"]
SPLITS = ["train", "valid", "test"]


def parse_map(text):
    pairs = {}
    for item in text.split(","):
        src, dst = item.split("=")
        pairs[src.strip().lower()] = CLASSES.index(dst.strip())
    return pairs


def merge_one(src_dir, mapping, out_dir, tag):
    names = yaml.safe_load(open(src_dir / "data.yaml"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    id_map = {i: mapping[n.lower()] for i, n in enumerate(names) if n.lower() in mapping}
    kept = 0
    for split in SPLITS:
        img_dir = src_dir / split / "images"
        lbl_dir = src_dir / split / "labels"
        if not img_dir.exists():
            continue
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in img_dir.iterdir():
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            rows = []
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if parts and int(parts[0]) in id_map:
                    rows.append(" ".join([str(id_map[int(parts[0])])] + parts[1:]))
            if not rows:
                continue
            stem = f"{tag}_{img.stem}"
            shutil.copy(img, out_dir / split / "images" / (stem + img.suffix))
            (out_dir / split / "labels" / (stem + ".txt")).write_text("\n".join(rows) + "\n")
            kept += 1
    print(tag, "images kept:", kept, "class map:", id_map)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", action="append", required=True, help="dataset folder")
    p.add_argument("--map", action="append", required=True,
                   help="e.g. cones=cone,barrier=barrier (same order as --src)")
    p.add_argument("--out", default="datasets/navobjects")
    a = p.parse_args()

    out = Path(a.out)
    for i, (s, m) in enumerate(zip(a.src, a.map)):
        merge_one(Path(s), parse_map(m), out, f"d{i}")

    cfg = {"path": str(out), "train": "train/images", "val": "valid/images",
           "test": "test/images", "names": {i: n for i, n in enumerate(CLASSES)}}
    yaml.safe_dump(cfg, open(out / "data.yaml", "w"), sort_keys=False)


if __name__ == "__main__":
    main()