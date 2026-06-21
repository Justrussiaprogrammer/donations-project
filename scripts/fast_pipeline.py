#!/usr/bin/env python3
"""
Единая точка запуска пайплайна с выбором движка детекции и режима VLM.

  --engine cpp (по умолчанию) — нативный детектор cpp/fast_detector
      (OpenVINO C++, ffmpeg-декодирование). Выходные файлы идентичны
      vlm_pipeline.py: events_summary.csv, totals_by_currency.csv,
      donations.jsonl, run_metadata.json, events/.

  --engine py — просто запускает scripts/vlm_pipeline.py с теми же аргументами
      (эталонная реализация).

Режим VLM (как в vlm_pipeline.py):
  по умолчанию — ПАРАЛЛЕЛЬНО: cpp-детектор стримит закрытые события на stdout
      по мере их закрытия, Python-воркер тут же гонит их через VLM, пока
      детекция ещё идёт. Для длинных стримов, где VLM — узкое место, это
      прячет почти всё время VLM за временем детекции.
  --sequential — ПОСЛЕДОВАТЕЛЬНО: сначала вся детекция (events.json), затем
      весь VLM. Даёт чистый раздельный тайминг стадий для сравнения.

Примеры:
  python3 scripts/fast_pipeline.py --video test/video/test_fragment.mp4 \
      --conf 0.25 --overwrite                       # cpp + параллельный VLM
  python3 scripts/fast_pipeline.py --video ... --sequential --overwrite
  python3 scripts/fast_pipeline.py --engine py --video ... --overwrite
  python3 scripts/fast_pipeline.py --video ... --skip-vlm   # только детекция

Сборка бинарника (один раз): ./cpp/build.sh
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import vlm_pipeline as vp  # noqa: E402


def resolve_openvino_model(model_path: Path, project_dir: Path) -> Path:
    """cpp-движку нужен OpenVINO IR. Если передан best.pt — ищем экспорт рядом."""
    if model_path.suffix == ".xml" or model_path.is_dir():
        return model_path
    candidate = model_path.parent / f"{model_path.stem}_openvino_model"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Для --engine cpp нужен OpenVINO-экспорт модели, не найден: {candidate}\n"
        "Сделайте экспорт:\n"
        f'  python3 -c "from ultralytics import YOLO; '
        f"YOLO('{model_path}').export(format='openvino', half=True, dynamic=False, imgsz=640)\""
    )


def _event_from_dict(e: dict[str, Any], video_name: str) -> vp.DonationEvent:
    """Собрать DonationEvent из JSON-объекта детектора (общий код для обоих режимов)."""
    ev = vp.DonationEvent(
        event_id=e["event_id"],
        video_name=video_name,
        start_sec=e["start_sec"],
        end_sec=e["end_sec"],
        first_frame=e["first_frame"],
        last_frame=e["last_frame"],
        last_box=(0, 0, 0, 0),
        detections_count=e["detections_count"],
        best_confidence=e["best_confidence"],
        best_timestamp_sec=e["best_timestamp_sec"],
        best_frame_idx=e["best_frame_idx"],
    )
    for c in e["candidates"]:
        ev.candidates.append(vp.CandidateCrop(
            event_id=ev.event_id,
            video_name=video_name,
            frame_idx=c["frame_idx"],
            timestamp_sec=c["timestamp_sec"],
            confidence=c["confidence"],
            class_id=c["class_id"],
            base_box=tuple(c["base_box"]),
            padded_box=tuple(c["padded_box"]),
            crop_path=Path(c["crop_path"]),
            annotated_frame_path=Path(c["annotated_frame_path"]) if c["annotated_frame_path"] else None,
            original_frame_path=Path(c["original_frame_path"]) if c["original_frame_path"] else None,
            score=c["score"],
        ))
    return ev


def _wait_for_file(path: Optional[Path], timeout: float = 30.0) -> None:
    """В параллельном режиме событие может прийти на stdout раньше, чем фоновый
    писатель cpp успел сбросить лучший кроп на диск. Дожидаемся файла."""
    if path is None:
        return
    deadline = time.time() + timeout
    while not path.exists() and time.time() < deadline:
        time.sleep(0.01)


def run_py_engine() -> None:
    """Делегируем эталонному пайплайну, убрав свои флаги из argv."""
    passthrough = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a == "--engine" or a == "--cpp-binary" or a == "--cpp-device":
            skip_next = True
            continue
        if a.startswith(("--engine=", "--cpp-binary=", "--cpp-device=")):
            continue
        passthrough.append(a)
    cmd = [sys.executable, str(SCRIPTS_DIR / "vlm_pipeline.py")] + passthrough
    raise SystemExit(subprocess.call(cmd))


def run_cpp_engine(args: argparse.Namespace) -> None:
    if getattr(args, "train_select", "").strip():
        raise SystemExit(
            "Сбор обучающих данных (--train-select) поддерживается только движком py "
            "(он держит кандидатов в памяти). Запустите с --engine py."
        )

    project_dir = Path(args.project_dir).expanduser().resolve()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_dir / model_path
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = project_dir / video_path

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    ov_model = resolve_openvino_model(model_path, project_dir)

    binary = Path(args.cpp_binary)
    if not binary.is_absolute():
        binary = project_dir / binary
    if not binary.exists() and binary.with_suffix(".exe").exists():
        binary = binary.with_suffix(".exe")  # Windows
    if not binary.exists():
        raise FileNotFoundError(
            f"Бинарник не найден: {binary}\nСоберите его: ./cpp/build.sh (Windows: cmake, см. README)"
        )

    run_name = args.run_name or f"{vp.safe_filename(video_path.stem)}_vlm_v5_run"
    output_dir = (project_dir / args.output_dir / run_name).resolve()
    events_dir = output_dir / "events"
    crops_dir = events_dir / "best_crops"
    frames_dir = events_dir / "annotated_frames"
    original_frames_dir = events_dir / "original_frames"
    tmp_dir = output_dir / "_tmp_candidates"

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output dir already exists and is not empty: {output_dir}\n"
                f"Use --overwrite to replace it or --run-name to pick another name."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_images:
        for d in (events_dir, crops_dir, frames_dir, original_frames_dir):
            d.mkdir(parents=True, exist_ok=True)

    mode = "sequential" if args.sequential else "parallel"
    print(f"Project:      {project_dir}")
    print(f"Engine:       cpp ({binary.name}, device {args.cpp_device})")
    print(f"VLM mode:     {mode}")
    print(f"YOLO model:   {ov_model}")
    print(f"Video:        {video_path}")
    print(f"Output:       {output_dir}")
    print(f"VLM server:   {args.vlm_server_url}")
    print(f"VLM model:    {args.vlm_model}\n")

    cmd = [
        str(binary),
        "--video", str(video_path),
        "--model", str(ov_model),
        "--device", args.cpp_device,
        "--frame-step", str(args.frame_step),
        "--conf", str(args.conf),
        "--padding-x", str(args.padding_x),
        "--padding-y", str(args.padding_y),
        "--event-gap-sec", str(args.event_gap_sec),
        "--event-iou-thr", str(args.event_iou_thr),
        "--event-center-thr", str(args.event_center_thr),
        "--keep-top-candidates", str(args.keep_top_candidates),
        "--max-processed-frames", str(args.max_processed_frames),
        "--tmp-dir", str(tmp_dir),
    ]
    if args.no_save_images:
        cmd.append("--no-save-images")

    vlm_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    wall_start = time.time()

    if args.sequential:
        det, vlm_results = _run_sequential(
            cmd, output_dir, crops_dir, frames_dir, original_frames_dir, args
        )
        vlm_elapsed = det.pop("_vlm_elapsed")
    else:
        det, vlm_results = _run_parallel(
            cmd, video_path.name, crops_dir, frames_dir, original_frames_dir, args
        )
        vlm_elapsed = None  # перекрыт детекцией, отдельного времени нет

    wall_elapsed = round(time.time() - wall_start, 2)
    n_events = det["events_count"]

    vlm_results.sort(key=lambda r: r[0]["event_id"])
    event_rows = [r[0] for r in vlm_results]
    jsonl_rows = [r[1] for r in vlm_results]

    shutil.rmtree(tmp_dir, ignore_errors=True)

    events_csv = output_dir / "events_summary.csv"
    totals_csv = output_dir / "totals_by_currency.csv"
    jsonl_path = output_dir / "donations.jsonl"
    metadata_json = output_dir / "run_metadata.json"

    totals_rows, totals_skipped_events = vp.build_totals_rows(event_rows)
    vp.write_csv(events_csv, event_rows)
    vp.write_csv(totals_csv, totals_rows)
    vp.write_jsonl(jsonl_path, jsonl_rows)

    yolo_elapsed = round(det["yolo_elapsed_sec"], 2)

    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "project_dir": str(project_dir),
                "run_name": run_name,
                "engine": "cpp",
                "cpp_binary": str(binary),
                "cpp_device": args.cpp_device,
                "vlm_mode": mode,
                "yolo_model": str(ov_model),
                "video": str(video_path),
                "fps": det["fps"],
                "total_frames": det["total_frames"],
                "frame_step": args.frame_step,
                "sampled_frames_processed": det["sampled_frames_processed"],
                "raw_detections": det["raw_detections"],
                "events": n_events,
                "yolo_conf": args.conf,
                "yolo_img_size": args.img_size,
                "padding_x": args.padding_x,
                "padding_y": args.padding_y,
                "event_gap_sec": args.event_gap_sec,
                "event_iou_thr": args.event_iou_thr,
                "event_center_thr": args.event_center_thr,
                "vlm_server_url": args.vlm_server_url,
                "vlm_model": args.vlm_model,
                "vlm_prompt_version": vp.VLM_PROMPT_VERSION,
                "vlm_prompt": vp.VLM_PROMPT,
                "vlm_retries": args.vlm_retries,
                "vlm_timeout": args.vlm_timeout,
                "vlm_max_tokens": args.vlm_max_tokens,
                "vlm_temperature": args.vlm_temperature,
                "skip_vlm": args.skip_vlm,
                "sequential": args.sequential,
                "no_save_images": args.no_save_images,
                "yolo_elapsed_sec": yolo_elapsed,
                "yolo_infer_sec": round(det["infer_sec"], 2),
                "yolo_decode_wait_sec": round(det["decode_wait_sec"], 2),
                "vlm_elapsed_sec": vlm_elapsed,
                "wall_elapsed_sec": wall_elapsed,
                "totals_skipped_events": totals_skipped_events,
                "outputs": {
                    "events_summary_csv": str(events_csv),
                    "totals_by_currency_csv": str(totals_csv),
                    "donations_jsonl": str(jsonl_path),
                    "events_dir": "" if args.no_save_images else str(events_dir),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDone.")
    print(f"  Mode:              {mode}")
    print(f"  YOLO detection:    {vp.seconds_to_timestamp(yolo_elapsed)} (compute)")
    if vlm_elapsed is not None:
        print(f"  VLM stage:         {vp.seconds_to_timestamp(vlm_elapsed)}")
    else:
        print(f"  VLM stage:         перекрыт детекцией (отдельного времени нет)")
    print(f"  Total wall clock:  {vp.seconds_to_timestamp(wall_elapsed)}")
    print(f"  Sampled frames:    {det['sampled_frames_processed']}")
    print(f"  Raw detections:    {det['raw_detections']}")
    print(f"  Donation events:   {n_events}")
    if totals_skipped_events:
        print(f"  Excluded from totals: {totals_skipped_events} event(s)")
    print(f"Events summary:      {events_csv}")
    print(f"Totals by currency:  {totals_csv}")
    print(f"Raw JSONL:           {jsonl_path}")
    if not args.no_save_images:
        print(f"Best crops/frames:   {events_dir}")


def _run_sequential(
    cmd: list[str],
    output_dir: Path,
    crops_dir: Path,
    frames_dir: Path,
    original_frames_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Детекция целиком (events.json), затем весь VLM. Чистый раздельный тайминг."""
    detector_json = output_dir / "_detector_events.json"
    rc = subprocess.call(cmd + ["--out-json", str(detector_json)])
    if rc != 0:
        raise RuntimeError(f"fast_detector завершился с кодом {rc}")

    with detector_json.open(encoding="utf-8") as f:
        det = json.load(f)
    detector_json.unlink(missing_ok=True)

    events = [_event_from_dict(e, det["video_name"]) for e in det["events"]]
    det["events_count"] = len(events)

    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if events:
        print(f"\nStarting VLM stage ({len(events)} events)...")
    vlm_start = time.time()
    for ev in events:
        try:
            event_row, jsonl_row = vp._process_event_vlm(
                ev, crops_dir, frames_dir, original_frames_dir, args
            )
            results.append((event_row, jsonl_row))
            print(f"[VLM] event {ev.event_id}/{len(events)}: "
                  f"donor={event_row.get('donor') or '?'} amount={event_row.get('amount') or '?'}")
        except Exception as exc:
            print(f"[VLM] event {ev.event_id} failed: {exc}")
            results.append(vp._make_error_rows(
                ev, f"worker_exception: {type(exc).__name__}: {exc}"))
    det["_vlm_elapsed"] = round(time.time() - vlm_start, 2)
    return det, results


def _run_parallel(
    cmd: list[str],
    video_name: str,
    crops_dir: Path,
    frames_dir: Path,
    original_frames_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Детектор стримит закрытые события на stdout; VLM-воркер обрабатывает их
    в отдельном потоке, пока детекция продолжается."""
    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    results_lock = threading.Lock()
    vlm_queue: queue.Queue = queue.Queue()
    worker = threading.Thread(
        target=vp._vlm_worker,
        args=(vlm_queue, results, results_lock, crops_dir, frames_dir,
              original_frames_dir, args),
        daemon=True,
        name="vlm-worker",
    )
    worker.start()
    print("VLM worker thread started — события идут в VLM по мере закрытия.\n")

    # stderr наследуется (прогресс детектора виден), stdout — это протокол JSONL.
    proc = subprocess.Popen(
        cmd + ["--stream-events"],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    det: dict[str, Any] = {}
    n_events = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print(f"[detector] не-JSON строка на stdout: {line[:120]}")
            continue
        if msg.get("type") == "event":
            ev = _event_from_dict(msg["event"], video_name)
            # Лучшие кропы/кадры события могли ещё не долететь на диск из фонового
            # писателя cpp. Писатель — пул потоков с маршрутизацией по хешу пути,
            # поэтому crop/ann/orig одной детекции пишутся разными потоками без
            # взаимного порядка: маленький crop.png может уже лежать на диске,
            # пока тяжёлый ann.jpg ещё в очереди. Ждём все три файла, иначе
            # process_event падает на copy2 ещё не записанного кадра.
            if ev.candidates:
                best = ev.candidates[0]
                _wait_for_file(best.crop_path)
                _wait_for_file(best.annotated_frame_path)
                _wait_for_file(best.original_frame_path)
            vlm_queue.put(ev)
            n_events += 1
        elif msg.get("type") == "summary":
            det = msg

    rc = proc.wait()
    if rc != 0:
        vlm_queue.put(None)
        worker.join()
        raise RuntimeError(f"fast_detector завершился с кодом {rc}")

    vlm_queue.put(None)  # sentinel
    print("\nWaiting for VLM worker to finish...")
    worker.join()

    if not det:
        raise RuntimeError("Детектор не прислал summary — протокол нарушен")
    det["events_count"] = n_events
    return det, results


def main() -> None:
    parser = vp.build_parser()
    parser.description = "Donation pipeline c выбором движка детекции (cpp/py)"
    parser.add_argument("--engine", choices=["cpp", "py"], default="cpp",
                        help="cpp — нативный детектор (быстрый), py — vlm_pipeline.py")
    parser.add_argument("--cpp-binary", default="cpp/fast_detector",
                        help="Путь к собранному fast_detector")
    parser.add_argument("--cpp-device", default="CPU",
                        help="OpenVINO device для cpp-движка: CPU или GPU")
    args = parser.parse_args()

    if args.frame_step <= 0:
        raise ValueError("--frame-step must be >= 1")
    if args.keep_top_candidates <= 0:
        raise ValueError("--keep-top-candidates must be >= 1")

    if args.engine == "py":
        run_py_engine()
    else:
        run_cpp_engine(args)


if __name__ == "__main__":
    main()
