import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.utils.prune as prune
from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics import YOLO


def size_mb(path):
    return os.path.getsize(path) / 1e6


def bench_ort(path, providers, imgsz, runs=100, warmup=10):
    sess = ort.InferenceSession(path, providers=providers)
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    t = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {name: x})
    return runs / (time.perf_counter() - t)


def bench_torch(model, device, imgsz, runs=100, warmup=10):
    net = model.model.to(device).eval()
    x = torch.rand(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(runs):
            net(x)
        if device == "cuda":
            torch.cuda.synchronize()
    return runs / (time.perf_counter() - t)


def prune_model(weights, amount, out_path):
    model = YOLO(weights)
    for m in model.model.modules():
        if isinstance(m, torch.nn.Conv2d):
            prune.l1_unstructured(m, "weight", amount=amount)
            prune.remove(m, "weight")
    model.save(out_path)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--prune", type=float, default=0.3)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    gpu = torch.cuda.is_available()

    base = YOLO(a.weights)
    rows.append(("pytorch fp32", size_mb(a.weights),
                 bench_torch(base, "cpu", a.imgsz),
                 bench_torch(base, "cuda", a.imgsz) if gpu else None))

    onnx_path = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, simplify=True)
    gpu_prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    rows.append(("onnx fp32", size_mb(onnx_path),
                 bench_ort(onnx_path, ["CPUExecutionProvider"], a.imgsz),
                 bench_ort(onnx_path, gpu_prov, a.imgsz) if gpu else None))

    int8_path = onnx_path.replace(".onnx", "_int8.onnx")
    try:
        quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QUInt8)
        rows.append(("onnx int8 (dynamic)", size_mb(int8_path),
                     bench_ort(int8_path, ["CPUExecutionProvider"], a.imgsz), None))
    except Exception as e:
        print("int8 quantization failed:", e)

    pruned_pt = prune_model(a.weights, a.prune, os.path.join(a.out, "pruned.pt"))
    pruned = YOLO(pruned_pt)
    rows.append((f"pytorch pruned {int(a.prune * 100)}%", size_mb(pruned_pt),
                 bench_torch(pruned, "cpu", a.imgsz),
                 bench_torch(pruned, "cuda", a.imgsz) if gpu else None))

    lines = ["| variant | size (MB) | CPU FPS | GPU FPS |", "|---|---|---|---|"]
    for n, s, c, g in rows:
        lines.append(f"| {n} | {s:.1f} | {c:.1f} | {'-' if g is None else f'{g:.1f}'} |")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(a.out, "benchmark.md"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
