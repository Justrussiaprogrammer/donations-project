#!/usr/bin/env python3
"""
Бенчмарк YOLO-стадии: эталонный Python-пайплайн против нативного cpp-детектора.

VLM-стадия не запускается (--skip-vlm): она одинакова для обоих движков
(HTTP-вызов llama.cpp) и зависит только от числа событий.

Движки:
  py-torch    python -m donsearcher + models/best.pt (ultralytics, torch CPU)
  py-openvino python -m donsearcher + models/best_openvino_model
              (OpenVINO на CPU, запускается с --device intel:cpu)
  cpp-cpu     cpp/fast_detector, OpenVINO CPU
  cpp-gpu     cpp/fast_detector, OpenVINO GPU (Intel iGPU)

Примеры:
  python3 scripts/benchmark_detector.py --video test/video/test_fragment.mp4 --conf 0.5
  python3 scripts/benchmark_detector.py --video ... --engines all --repeats 3
  python3 scripts/benchmark_detector.py --video ... --no-save-images   # чистая детекция
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
ALL_ENGINES = ["py-torch", "py-openvino", "cpp-cpu", "cpp-gpu"]
DEFAULT_ENGINES = ["py-torch", "cpp-cpu"]


def run_py(engine: str, args: argparse.Namespace) -> dict:
    if engine == "py-torch":
        model, device = args.model_pt, "cpu"
    else:  # py-openvino — OpenVINO backend ожидает device в форме intel:*
        model, device = args.model_openvino, "intel:cpu"
    run_name = f"_bench_{engine.replace('-', '_')}"
    cmd = [
        sys.executable, "-m", "donsearcher",  # py-движок; cwd=PROJECT_DIR (см. ниже)
        "--model", model,
        "--video", args.video,
        "--device", device,
        "--frame-step", str(args.frame_step),
        "--conf", str(args.conf),
        "--img-size", str(args.img_size),
        "--skip-vlm", "--sequential", "--overwrite",
        "--run-name", run_name,
    ]
    if args.max_processed_frames:
        cmd += ["--max-processed-frames", str(args.max_processed_frames)]
    if args.no_save_images:
        cmd += ["--images-schema", "0"]  # py-движок: битовая маска вместо флага

    t0 = time.time()
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True,
                   stdout=subprocess.DEVNULL if args.quiet else None)
    wall = time.time() - t0

    run_dir = PROJECT_DIR / "vlm_runs" / run_name
    with (run_dir / "run_metadata.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    if not args.keep_runs:
        shutil.rmtree(run_dir, ignore_errors=True)
    return {
        "yolo_sec": meta["yolo_elapsed_sec"],
        "wall_sec": round(wall, 2),
        "detections": meta["raw_detections"],
        "events": meta["events"],
    }


def run_cpp(engine: str, args: argparse.Namespace) -> dict:
    device = "GPU" if engine == "cpp-gpu" else "CPU"
    binary = PROJECT_DIR / "cpp" / "fast_detector"
    if not binary.exists():
        raise FileNotFoundError(f"{binary} не найден — соберите: ./cpp/build.sh")

    work = Path(tempfile.mkdtemp(prefix=f"bench_{engine}_"))
    tmp_dir = work / "tmp"
    tmp_dir.mkdir()
    out_json = work / "events.json"
    cmd = [
        str(binary),
        "--video", args.video,
        "--model", args.model_openvino,
        "--device", device,
        "--frame-step", str(args.frame_step),
        "--conf", str(args.conf),
        "--tmp-dir", str(tmp_dir),
        "--out-json", str(out_json),
        "--max-processed-frames", str(args.max_processed_frames),
    ]
    if args.no_save_images:
        cmd.append("--no-save-images")
    if args.quiet:
        cmd.append("--quiet")

    t0 = time.time()
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)
    wall = time.time() - t0

    with out_json.open(encoding="utf-8") as f:
        det = json.load(f)
    shutil.rmtree(work, ignore_errors=True)
    return {
        "yolo_sec": round(det["yolo_elapsed_sec"], 2),
        "wall_sec": round(wall, 2),
        "detections": det["raw_detections"],
        "events": len(det["events"]),
        "infer_sec": round(det["infer_sec"], 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Бенчмарк детекторной стадии py vs cpp")
    p.add_argument("--video", required=True)
    p.add_argument("--model-pt", default="models/best.pt")
    p.add_argument("--model-openvino", default="models/best_openvino_model")
    p.add_argument("--frame-step", type=int, default=10)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--img-size", default="640",
                   help="imgsz py-движка: '640' или '576,1024' (высота,ширина); "
                        "cpp берёт форму входа из самой IR")
    p.add_argument("--max-processed-frames", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1, help="Берётся лучший результат")
    p.add_argument("--engines", default=",".join(DEFAULT_ENGINES),
                   help=f"через запятую из {ALL_ENGINES} или 'all'")
    p.add_argument("--no-save-images", action="store_true",
                   help="Чистая детекция без записи кропов/кадров")
    p.add_argument("--keep-runs", action="store_true")
    p.add_argument("--quiet", action="store_true", default=True)
    p.add_argument("--verbose", dest="quiet", action="store_false")
    args = p.parse_args()

    engines = ALL_ENGINES if args.engines == "all" else [
        e.strip() for e in args.engines.split(",") if e.strip()
    ]
    for e in engines:
        if e not in ALL_ENGINES:
            raise SystemExit(f"Неизвестный движок: {e} (доступны: {ALL_ENGINES})")

    results: dict[str, dict] = {}
    for engine in engines:
        print(f"\n=== {engine} ({args.repeats} прогон(а)) ===")
        best = None
        for i in range(args.repeats):
            try:
                r = (run_py if engine.startswith("py") else run_cpp)(engine, args)
            except Exception as exc:
                print(f"  {engine} прогон {i + 1} упал: {exc}")
                break
            print(f"  прогон {i + 1}: YOLO-стадия {r['yolo_sec']}s "
                  f"(детекций {r['detections']}, событий {r['events']})")
            if best is None or r["yolo_sec"] < best["yolo_sec"]:
                best = r
        if best:
            results[engine] = best

    if not results:
        raise SystemExit("Ни один движок не отработал")

    baseline = results.get("py-torch") or results[next(iter(results))]
    base_name = "py-torch" if "py-torch" in results else next(iter(results))

    print(f"\n{'=' * 72}")
    print(f"Видео: {args.video}  frame-step={args.frame_step} conf={args.conf} "
          f"{'(no images)' if args.no_save_images else '(с записью кропов/кадров)'}")
    print(f"{'движок':<14}{'YOLO-стадия':>14}{'ускорение':>12}{'детекций':>10}{'событий':>9}")
    print("-" * 72)
    for engine, r in results.items():
        speedup = baseline["yolo_sec"] / r["yolo_sec"] if r["yolo_sec"] else 0
        print(f"{engine:<14}{r['yolo_sec']:>12.2f}s"
              f"{speedup:>10.2f}x{r['detections']:>10}{r['events']:>9}")
    print(f"(ускорение относительно {base_name})")

    out = PROJECT_DIR / "benchmarks"
    out.mkdir(exist_ok=True)
    out_file = out / f"detector_bench_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump({
            "video": args.video,
            "frame_step": args.frame_step,
            "conf": args.conf,
            "img_size": args.img_size,
            "no_save_images": args.no_save_images,
            "repeats": args.repeats,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены: {out_file}")


if __name__ == "__main__":
    main()
