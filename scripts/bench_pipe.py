#!/usr/bin/env python3
"""Честный e2e-бенчмарк одного видео на NVIDIA: ffmpeg(NVDEC) -> pipe -> батчевый TensorRT.

Архитектура повторяет cpp/fast_detector и уровень 5 bench-gpu.ipynb: ffmpeg
декодирует видео и отбирает каждый N-й кадр в ОТДЕЛЬНОМ процессе, сырые
BGR-кадры уходят по pipe, читатель — отдельный поток, инференс — батчами;
декод и инференс перекрываются. Только замер скорости: детекции считаются,
события не группируются.

Файл самодостаточен (не импортирует donsearcher) — можно копировать на
арендованный сервер отдельно от репозитория.

Несколько копий безопасно запускать ПАРАЛЛЕЛЬНО на одной карте — так меряется,
сколько видео тянет GPU (см. server_bench_nvidia.sh --parallel N).

Примеры:
  python3 scripts/bench_pipe.py --model models/best_1024x1024_b8.engine \
      --video test/video/test_fragment.mp4 --img-size 576,1024 --batch 8
  python3 scripts/bench_pipe.py --model models/best.pt --half \
      --video test/video/test_fragment.mp4 --img-size 384,640   # без TRT (медленнее)
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

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


def decode_cmd(video, step, mode, probe=False):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if mode.startswith("nvdec"):
        cmd += ["-hwaccel", "cuda"]
        if mode == "nvdec-gpusel":
            cmd += ["-hwaccel_output_format", "cuda"]
    vf = rf"select='not(mod(n\,{step}))'"
    if mode == "nvdec-gpusel":
        # select на GPU: по PCIe скачиваются только сэмплы, а не все кадры
        vf += ",hwdownload,format=nv12"
    cmd += ["-i", str(video), "-vf", vf, "-vsync", "0"]
    if probe:
        cmd += ["-frames:v", "2", "-f", "null", "-"]
    else:
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    return cmd


def pick_decode_mode(video, step, want):
    if want != "auto":
        return want
    hw = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"],
                        capture_output=True, text=True).stdout
    candidates = (["nvdec-gpusel", "nvdec"] if "cuda" in hw else []) + ["cpu"]
    for mode in candidates:
        r = subprocess.run(decode_cmd(video, step, mode, probe=True),
                           capture_output=True)
        if r.returncode == 0:
            return mode
    raise SystemExit(f"ffmpeg не смог декодировать {video} ни одним способом")


def main() -> None:
    p = argparse.ArgumentParser(description="pipe-бенчмарк: ffmpeg -> батчевый TRT")
    p.add_argument("--model", required=True, help=".engine (или .pt с --half)")
    p.add_argument("--video", required=True)
    p.add_argument("--img-size", default="576,1024", help="'в,ш', напр. 576,1024")
    p.add_argument("--frame-step", type=int, default=10)
    p.add_argument("--batch", type=int, default=8,
                   help="для .engine должен совпадать с batch экспорта")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--device", default="0")
    p.add_argument("--half", action="store_true",
                   help="FP16 для .pt (engine уже собран в FP16)")
    p.add_argument("--decode", default="auto",
                   choices=["auto", "nvdec-gpusel", "nvdec", "cpu"])
    p.add_argument("--json", help="куда написать JSON с результатами")
    args = p.parse_args()

    import cv2
    from ultralytics import YOLO

    imgsz = parse_img_size(args.img_size)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"не открылось видео: {args.video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    frame_bytes = w * h * 3

    mode = pick_decode_mode(args.video, args.frame_step, args.decode)

    model = YOLO(args.model, task="detect")
    fp16 = fp16_kwargs(args.half and not args.model.endswith(".engine"))
    dummy = [np.zeros((h, w, 3), np.uint8)] * args.batch
    for _ in range(2):  # прогрев ДО таймера (загрузка engine, аллокации)
        model.predict(dummy, imgsz=imgsz, device=args.device, conf=args.conf,
                      verbose=False, **fp16)

    def flush(batch_frames):
        real = len(batch_frames)
        if real < args.batch:  # статичный batch-engine: добиваем повтором последнего
            batch_frames = batch_frames + [batch_frames[-1]] * (args.batch - real)
        rs = model.predict(batch_frames, imgsz=imgsz, device=args.device,
                           conf=args.conf, verbose=False, **fp16)
        return sum(len(r.boxes) for r in rs[:real])

    # читатель pipe — отдельный поток: ffmpeg декодирует следующие кадры,
    # пока GPU инференсит текущий батч (иначе wall = сумма, а не max)
    q: queue.Queue = queue.Queue(maxsize=args.batch * 4)

    def _reader(proc):
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            q.put(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
        q.put(None)

    t0_epoch = time.time()
    t0 = time.perf_counter()
    proc = subprocess.Popen(decode_cmd(args.video, args.frame_step, mode),
                            stdout=subprocess.PIPE, bufsize=frame_bytes * 8)
    threading.Thread(target=_reader, args=(proc,), daemon=True).start()
    n_frames = n_det = 0
    batch = []
    while True:
        frame = q.get()
        if frame is None:
            break
        batch.append(frame)
        n_frames += 1
        if len(batch) == args.batch:
            n_det += flush(batch)
            batch = []
    if batch:
        n_det += flush(batch)
    proc.wait()
    wall = time.perf_counter() - t0

    res = {
        "model": args.model, "video": args.video, "imgsz": imgsz,
        "frame_step": args.frame_step, "batch": args.batch, "decode": mode,
        "video_dur_s": round(dur, 1), "samples": n_frames, "detections": n_det,
        "wall_s": round(wall, 2), "sample_fps": round(n_frames / wall, 1),
        "rt_factor": round(dur / wall, 1),
        "t0_epoch": t0_epoch, "t1_epoch": time.time(),
    }
    print(f"декод: {mode}; видео {dur:.0f}s, сэмплов {n_frames}, детекций {n_det}")
    print(f"pipe e2e: {wall:.1f}s = {res['sample_fps']} сэмпл-fps = "
          f"{res['rt_factor']}x реального времени")
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
