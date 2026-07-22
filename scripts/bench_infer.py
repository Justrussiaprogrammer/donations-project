#!/usr/bin/env python3
"""Чистый инференс-бенчмарк детектора: fps модели без видео и декода.

Гоняет model.predict по случайным кадрам 1920x1080 (без диска и ffmpeg) —
это потолок скорости самой модели на данном устройстве. Сравнивать с ним
end-to-end прогон пайплайна: если end-to-end сильно ниже потолка, узкое
место — декод/оверхед, а не модель.

Файл самодостаточен (не импортирует donsearcher) — можно копировать на
сервер/в ноутбук отдельно от репозитория.

Примеры:
  python3 scripts/bench_infer.py --model models/best_1024x1024.pt --device cpu --img-size 576,1024
  python3 scripts/bench_infer.py --model models/best_1024x1024.pt --device 0 --img-size 576,1024 --half
  python3 scripts/bench_infer.py --model models/best_1024x1024.engine --device 0 --img-size 576,1024
  python3 scripts/bench_infer.py --model models/best_1024x1024_openvino_model --device intel:cpu --img-size 576,1024
"""

from __future__ import annotations

import argparse
import re
import time

import numpy as np


def parse_img_size(value: str):
    parts = [p for p in re.split(r"[x,×]", str(value).strip()) if p]
    dims = [int(p) for p in parts]
    return dims[0] if len(dims) == 1 else dims


def fp16_kwargs(enable: bool):
    """ultralytics 8.4.90+ переименовал half -> quantize."""
    if not enable:
        return {}
    from ultralytics.utils import DEFAULT_CFG_DICT
    return {"quantize": "fp16"} if "quantize" in DEFAULT_CFG_DICT else {"half": True}


def main() -> None:
    p = argparse.ArgumentParser(description="Чистый инференс-бенчмарк (fps модели)")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cpu",
                   help="cpu / CUDA-индекс ('0') / intel:cpu / intel:gpu")
    p.add_argument("--img-size", default="640", help="'640' или '576,1024' (в,ш)")
    p.add_argument("--half", action="store_true", help="FP16 (CUDA)")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--n", type=int, default=100, help="замеряемых итераций")
    args = p.parse_args()

    from ultralytics import YOLO

    imgsz = parse_img_size(args.img_size)
    model = YOLO(args.model, task="detect")

    rng = np.random.default_rng(0)
    # 8 разных кадров по кругу — исключаем кэш-эффекты на одном буфере
    frames = [rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(8)]

    fp16 = fp16_kwargs(args.half and not str(args.model).endswith(".engine"))

    def infer(img):
        return model.predict(source=img, imgsz=imgsz, device=args.device,
                             conf=args.conf, verbose=False, **fp16)

    for i in range(args.warmup):
        infer(frames[i % len(frames)])

    t0 = time.perf_counter()
    for i in range(args.n):
        infer(frames[i % len(frames)])
    dt = time.perf_counter() - t0

    fps = args.n / dt
    print(f"\nmodel={args.model} device={args.device} imgsz={imgsz} "
          f"half={args.half} n={args.n}")
    print(f"infer: {dt / args.n * 1000:.1f} ms/кадр = {fps:.1f} fps")
    print(f"ориентир: при frame-step 10 на 60fps-видео нужно 6 сэмпл-кадров/с "
          f"реального времени -> потолок {fps / 6:.1f}x реального времени")


if __name__ == "__main__":
    main()
