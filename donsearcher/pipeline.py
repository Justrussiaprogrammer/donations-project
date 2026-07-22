#!/usr/bin/env python3
"""
Donation extraction pipeline v6 (memory-optimised) — the reference Python engine.

video -> YOLO donation detector -> event grouping -> best crop ->
local Qwen3-VL via llama.cpp server -> JSON -> CSV/JSONL.

This is the orchestrator (run_pipeline) plus the CLI (build_parser/main). It lives
inside the `donsearcher` package alongside the focused logic modules (geometry,
textutil, vlm_client, events, vlm_events, dedup, reports, training); the package
__init__ re-exports their public names, so callers use `import donsearcher` and
`donsearcher.<name>`. Run it via `python3 -m donsearcher` or the installed
console command `donsearcher`.

Memory model:
  1. CandidateCrop stores file paths instead of numpy arrays.
     Images are written to a tmp dir immediately during YOLO stage,
     evicted candidates' files are deleted right away.
     This keeps RAM proportional to keep_top_candidates * events_alive_at_once,
     not to the whole video.
  2. find_matching_event only searches active events (end_sec within gap).
     Old events are moved to a closed list once per frame, keeping the
     search O(active) instead of O(all).

The ultralytics/torch import is deferred into run_pipeline(), so importing the
package for its helpers does not pay the torch import cost.

The previous versions lives in archive.
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2

# Only what the orchestrator itself uses; the package's __init__ is the single
# place that re-exports the full public surface (donsearcher.<name>).
from .geometry import box_area, box_height, clamp, expand_box, push_recent_height
from .textutil import parse_img_size, safe_filename, seconds_to_timestamp, unpack_images_schema
from .vlm_client import DEFAULT_PROMPT_VERSION, load_prompt
from .events import (
    CandidateCrop,
    DonationEvent,
    _delete_candidate_files,
    add_candidate,
    find_matching_event,
)
from .vlm_events import _make_error_rows, _process_event_vlm, _vlm_worker
from .dedup import GENERIC_DONORS, GENERIC_MESSAGES, build_totals_rows, dedup_events
from .reports import build_run_record, write_csv, write_events_meta, write_jsonl
from .training import TRAIN_STRATEGIES, TrainingCollector


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    project_dir = Path(args.project_dir).expanduser().resolve()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_dir / model_path

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = project_dir / video_path

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Промпт загружается один раз до старта VLM-воркера (fail fast при опечатке
    # в --vlm-prompt); текст кладётся в args — его читает _process_event_vlm.
    vlm_prompt_text, vlm_prompt_version = load_prompt(args.vlm_prompt)
    args.vlm_prompt_text = vlm_prompt_text

    # Training-data collection turns the run into a pure dataset-collection mode:
    # no VLM, no event/totals reports, no events/ crop tree — just the dataset files
    # written straight into the run dir.
    train_strategies = {s.strip() for s in args.train_select.split(",") if s.strip()}
    training_mode = bool(train_strategies)

    # Битовая маска сохраняемых изображений (fail fast при значении вне 0..7).
    save_crops, save_annotated, save_original = unpack_images_schema(args.images_schema)
    save_any_images = save_crops or save_annotated or save_original
    if args.events_meta and not save_crops:
        print("ВНИМАНИЕ: --events-meta задан, но бит crops в --images-schema выключен — "
              "поле crop в метаданных будет пустым (VLM-серверу нечего будет обрабатывать).")

    run_name = args.run_name or f"{safe_filename(video_path.stem)}_vlm_v6_run"
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
    if save_any_images and not training_mode:
        events_dir.mkdir(parents=True, exist_ok=True)
        if save_crops:
            crops_dir.mkdir(parents=True, exist_ok=True)
        if save_annotated:
            frames_dir.mkdir(parents=True, exist_ok=True)
        if save_original:
            original_frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project:      {project_dir}")
    print(f"YOLO model:   {model_path}")
    print(f"YOLO device:  {args.device}")
    print(f"Video:        {video_path}")
    print(f"Output:       {output_dir}")
    print(f"VLM server:   {args.vlm_server_url}")
    print(f"VLM model:    {args.vlm_model}")

    # The collector object needs fps and is created once the video is open (below).
    train_dir: Optional[Path] = None
    collector: Optional[TrainingCollector] = None
    if training_mode:
        if not args.skip_vlm:
            args.skip_vlm = True
        print("Training-data collection is on -> VLM stage skipped, only the dataset is written.")
        # Default: dataset files (images/labels/previews/manifest.csv) go straight into
        # the run dir — no extra training_data/ wrapper.
        train_dir = Path(args.train_dir).expanduser() if args.train_dir else output_dir
        if not train_dir.is_absolute():
            train_dir = project_dir / train_dir
        print(f"Training data: {sorted(train_strategies)} -> {train_dir}")
    # 'uncertain' needs to see weak fires below --conf, so run detection at the lower
    # floor; boxes >= --conf are still the only ones that feed events (so grouping is
    # unchanged), boxes in [floor, --conf) are captured for training only.
    uncertain_on = "uncertain" in train_strategies
    detect_conf = args.train_uncertain_min if uncertain_on else args.conf
    if uncertain_on:
        print(f"Detection threshold lowered to {detect_conf} for 'uncertain' capture "
              f"(donations still require conf >= {args.conf}).")
    # result.plot() нужен только если сохраняются annotated-кадры или включён
    # training-режим (превью датасета рисуются из annotated).
    need_annotated = save_annotated or training_mode

    vlm_queue: queue.Queue = queue.Queue()
    vlm_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if not args.sequential and not training_mode:
        vlm_results_lock = threading.Lock()
        vlm_thread = threading.Thread(
            target=_vlm_worker,
            args=(vlm_queue, vlm_results, vlm_results_lock, crops_dir, frames_dir, original_frames_dir, args),
            daemon=True,
            name="vlm-worker",
        )
        vlm_thread.start()
        print("VLM worker thread started - will process events as they close.")

    # Deferred so importing this module (for its helpers) does not pay the torch
    # import cost; only an actual detection run pulls in ultralytics.
    from ultralytics import YOLO
    model = YOLO(str(model_path), task="detect")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"FPS: {fps:.3f}, frames: {total_frames}, frame_step: {args.frame_step}\n")

    if training_mode:
        collector = TrainingCollector(
            train_dir=train_dir,
            strategies=train_strategies,
            budget=args.train_budget,
            worst_per_event=args.train_worst_per_event,
            conf_thr=args.conf,
            uncertain_min=args.train_uncertain_min,
            video_name=video_path.name,
            fps=fps,
            total_frames=total_frames,
        )

    all_events: list[DonationEvent] = []
    active_events: list[DonationEvent] = []

    frame_idx = 0
    processed_frames = 0
    raw_detections_count = 0
    detection_counter = 0  # unique ID for tmp file naming

    print("Starting YOLO detection stage...")
    stage_start_time = time.time()
    batch_start_time = stage_start_time

    while True:
        # Skipped frames: only advance the decoder (grab), don't pay for the
        # full decode-to-numpy (retrieve) of a frame we won't process.
        if frame_idx % args.frame_step != 0:
            if not cap.grab():
                break
            frame_idx += 1
            continue

        ok, frame = cap.read()
        if not ok:
            break

        processed_frames += 1
        timestamp_sec = frame_idx / fps
        h, w = frame.shape[:2]

        still_active = []
        for ev in active_events:
            if timestamp_sec - ev.end_sec <= args.event_gap_sec:
                still_active.append(ev)
            else:
                for evicted in ev.candidates[1:]:
                    _delete_candidate_files(evicted)
                ev.candidates = ev.candidates[:1]
                if not args.sequential and not training_mode:
                    vlm_queue.put(ev)
        active_events = still_active

        result = model.predict(
            source=frame,
            conf=detect_conf,
            imgsz=args.img_size,
            device=args.device,
            verbose=False,
        )[0]

        boxes = result.boxes
        # Per-frame accumulators for training-data collection (see collector calls below).
        # confident_boxes: conf >= args.conf (real donations -> events + labels).
        # uncertain_boxes: conf in [detect_conf, args.conf) (weak fires, only when detection
        # ran at the lower floor for the 'uncertain' strategy).
        confident_boxes: list[tuple[tuple[int, int, int, int], float, int]] = []
        uncertain_boxes: list[tuple[tuple[int, int, int, int], float, int]] = []
        # (event_id, candidate_score, confidence, base_box) for per-event train strategies
        frame_dets: list[tuple[int, float, float, tuple[int, int, int, int]]] = []
        annotated = None
        if boxes is not None and len(boxes) > 0:
            annotated = result.plot() if need_annotated else None

            for box in boxes:
                cls_id = int(box.cls[0].cpu().item())
                conf = float(box.conf[0].cpu().item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()

                base_box = (
                    clamp(x1, 0, w - 1),
                    clamp(y1, 0, h - 1),
                    clamp(x2, 0, w - 1),
                    clamp(y2, 0, h - 1),
                )
                if conf < args.conf:
                    # below the donation threshold -> uncertain (training capture only);
                    # does NOT take part in event grouping/candidates.
                    uncertain_boxes.append((base_box, conf, cls_id))
                    continue
                confident_boxes.append((base_box, conf, cls_id))
                padded_box = expand_box(base_box, w, h, args.padding_x, args.padding_y)
                px1, py1, px2, py2 = padded_box
                crop = frame[py1:py2, px1:px2].copy()

                bx1, by1, bx2, by2 = base_box

                # Area: larger donation = more pixels for VLM to read (0–0.3)
                area_norm = box_area(base_box) / max(1, w * h)
                area_score = min(area_norm * 10.0, 0.3)

                # Sharpness: blurry crops give bad VLM results (0–0.4).
                # Laplacian variance: ~20 for motion-blurred, ~500+ for sharp text.
                # Threshold of 500 may need tuning for your stream resolution.
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness_score = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 0.4)

                # Clipping penalty: padding was cut off by the frame edge (-0.15)
                clipped = (
                    (px1 == 0 and bx1 > args.padding_x) or
                    (py1 == 0 and by1 > args.padding_y) or
                    (px2 == w - 1 and bx2 < w - 1 - args.padding_x) or
                    (py2 == h - 1 and by2 < h - 1 - args.padding_y)
                )
                clip_penalty = -0.15 if clipped else 0.0

                candidate_score = conf + area_score + sharpness_score + clip_penalty

                det_id = detection_counter
                detection_counter += 1
                # Кроп пишется всегда: он нужен VLM (даже при выключенном бите crops
                # OCR читает его из tmp) и копируется в best_crops при закрытии события.
                tmp_crop_path = tmp_dir / f"det{det_id:07d}_crop.png"
                cv2.imwrite(str(tmp_crop_path), crop)
                tmp_ann_path = None
                tmp_orig_path = None
                if save_annotated and annotated is not None:
                    tmp_ann_path = tmp_dir / f"det{det_id:07d}_ann.jpg"
                    cv2.imwrite(str(tmp_ann_path), annotated)
                if save_original:
                    tmp_orig_path = tmp_dir / f"det{det_id:07d}_orig.png"
                    cv2.imwrite(str(tmp_orig_path), frame)

                matched = find_matching_event(
                    events=active_events,
                    box=base_box,
                    frame_w=w,
                    frame_h=h,
                    iou_thr=args.event_iou_thr,
                    center_thr=args.event_center_thr,
                    current_frame_idx=frame_idx,
                    split_height_frac=args.event_split_height_frac,
                )

                if matched is None:
                    matched = DonationEvent(
                        event_id=len(all_events) + 1,
                        video_name=video_path.name,
                        start_sec=timestamp_sec,
                        end_sec=timestamp_sec,
                        first_frame=frame_idx,
                        last_frame=frame_idx,
                        last_box=base_box,
                    )
                    all_events.append(matched)
                    active_events.append(matched)

                matched.end_sec = timestamp_sec
                matched.last_frame = frame_idx
                matched.last_box = base_box
                matched.detections_count += 1
                push_recent_height(matched.recent_heights, box_height(base_box))

                if conf >= matched.best_confidence:
                    matched.best_confidence = conf
                    matched.best_timestamp_sec = timestamp_sec
                    matched.best_frame_idx = frame_idx

                candidate = CandidateCrop(
                    event_id=matched.event_id,
                    video_name=video_path.name,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    confidence=conf,
                    class_id=cls_id,
                    base_box=base_box,
                    padded_box=padded_box,
                    crop_path=tmp_crop_path,
                    annotated_frame_path=tmp_ann_path,
                    original_frame_path=tmp_orig_path,
                    score=candidate_score,
                )
                add_candidate(matched, candidate, args.keep_top_candidates)
                frame_dets.append((matched.event_id, candidate_score, conf, base_box))
                raw_detections_count += 1

        if collector is not None:
            # Per-event strategies (best/worst/random): group this frame's confident
            # detections by event, deferred to here so the label captures ALL confident
            # boxes in the frame.
            if frame_dets:
                by_event: dict[int, list[tuple[float, float, tuple[int, int, int, int]]]] = {}
                for eid, sc, cf, bx in frame_dets:
                    by_event.setdefault(eid, []).append((sc, cf, bx))
                for eid, recs in by_event.items():
                    collector.observe_event(eid, recs, frame, confident_boxes, frame_idx, w, h, annotated)
            # Frame-level strategies (uncertain/negatives).
            collector.observe_frame(frame, confident_boxes, uncertain_boxes, frame_idx, w, h, annotated)

        if args.max_processed_frames and processed_frames >= args.max_processed_frames:
            break

        if processed_frames % 100 == 0:
            batch_elapsed = time.time() - batch_start_time
            print(
                f"Processed sampled frames: {processed_frames}, "
                f"source frame: {frame_idx}, detections: {raw_detections_count}, events: {len(all_events)}, "
                f"batch time: {round(batch_elapsed, 2)}с"
            )
            batch_start_time = time.time()

        frame_idx += 1

    cap.release()
    yolo_elapsed = round(time.time() - stage_start_time, 2)
    print(f"\nYOLO stage done in {seconds_to_timestamp(yolo_elapsed)}.")
    print(f"Sampled frames: {processed_frames}, detections: {raw_detections_count}, events: {len(all_events)}")

    # Clean up non-best candidates for events still active at end of video.
    for ev in active_events:
        for evicted in ev.candidates[1:]:
            _delete_candidate_files(evicted)
        ev.candidates = ev.candidates[:1]

    training_summary: Optional[dict[str, int]] = None
    if collector is not None:
        training_summary = collector.finalize()
        print("Training data saved to " + str(train_dir) + ": "
              + ", ".join(f"{k}={training_summary[k]}" for k in sorted(train_strategies)))

    vlm_elapsed: Optional[float] = None

    # Training mode is a pure dataset-collection run: no VLM, no event/totals reports
    # (the training manifest replaces them).
    if not training_mode:
        if args.sequential:
            print(f"\nStarting VLM stage (sequential)...")
            vlm_start = time.time()
            for ev in all_events:
                try:
                    event_row, jsonl_row = _process_event_vlm(
                        ev, crops_dir, frames_dir, original_frames_dir, args
                    )
                    vlm_results.append((event_row, jsonl_row))
                    print(f"[VLM] event {ev.event_id}/{len(all_events)}: "
                          f"donor={event_row.get('donor') or '?'} amount={event_row.get('amount') or '?'}")
                except Exception as exc:
                    print(f"[VLM] event {ev.event_id} failed: {exc}")
                    vlm_results.append(_make_error_rows(ev, f"worker_exception: {type(exc).__name__}: {exc}"))
            vlm_elapsed = round(time.time() - vlm_start, 2)
            print(f"VLM stage done in {seconds_to_timestamp(vlm_elapsed)}.")
        else:
            # Flush remaining active events to worker thread, then wait for it to finish.
            for ev in active_events:
                vlm_queue.put(ev)
            vlm_queue.put(None)
            print("Waiting for VLM worker to finish...")
            vlm_thread.join()
            vlm_results.sort(key=lambda r: r[0]["event_id"])

    mode = "sequential" if args.sequential else "parallel"
    wall_elapsed = round(time.time() - stage_start_time, 2)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    metadata_json = output_dir / "run_metadata.json"
    events_csv: Optional[Path] = None
    totals_csv: Optional[Path] = None
    jsonl_path: Optional[Path] = None
    duplicate_events = 0
    totals_skipped_events = 0

    events_meta_path: Optional[Path] = None
    if not training_mode:
        event_rows = [r[0] for r in vlm_results]
        jsonl_rows = [r[1] for r in vlm_results]
        duplicate_events = dedup_events(event_rows, jsonl_rows)

        events_csv = output_dir / "events_summary.csv"
        totals_csv = output_dir / "totals_by_currency.csv"
        jsonl_path = output_dir / "donations.jsonl"
        totals_rows, totals_skipped_events = build_totals_rows(event_rows)

        write_csv(events_csv, event_rows)
        write_csv(totals_csv, totals_rows)
        write_jsonl(jsonl_path, jsonl_rows)

        # Сайдкар для раздельного запуска стадий (сервер A -> сервер B):
        # уезжает вместе с кропами, VLM-сервер по нему связывает OCR с событиями.
        if args.events_meta:
            events_meta_path = output_dir / "events_meta.jsonl"
            run_record = build_run_record(
                args.events_meta, video_path.name,
                args.streamer, args.platform, args.stream_date,
                run_name=run_name, fps=fps, frame_step=args.frame_step,
                conf=args.conf, images_schema=args.images_schema,
            )
            write_events_meta(events_meta_path, run_record, event_rows, args.events_meta)

    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "project_dir": str(project_dir),
                "run_name": run_name,
                "yolo_model": str(model_path),
                "video": str(video_path),
                "fps": fps,
                "total_frames": total_frames,
                "frame_step": args.frame_step,
                "sampled_frames_processed": processed_frames,
                "raw_detections": raw_detections_count,
                "events": len(all_events),
                "yolo_conf": args.conf,
                "yolo_img_size": args.img_size,
                "padding_x": args.padding_x,
                "padding_y": args.padding_y,
                "event_gap_sec": args.event_gap_sec,
                "event_iou_thr": args.event_iou_thr,
                "event_center_thr": args.event_center_thr,
                "event_split_height_frac": args.event_split_height_frac,
                "vlm_server_url": args.vlm_server_url,
                "vlm_model": args.vlm_model,
                "vlm_prompt_version": vlm_prompt_version,
                "vlm_prompt": vlm_prompt_text,
                "vlm_retries": args.vlm_retries,
                "vlm_timeout": args.vlm_timeout,
                "vlm_max_tokens": args.vlm_max_tokens,
                "vlm_temperature": args.vlm_temperature,
                "skip_vlm": args.skip_vlm,
                "sequential": args.sequential,
                "images_schema": args.images_schema,
                "events_meta": args.events_meta,
                "streamer": args.streamer,
                "platform": args.platform,
                "stream_date": args.stream_date,
                "vlm_mode": mode,
                "yolo_elapsed_sec": yolo_elapsed,
                "vlm_elapsed_sec": vlm_elapsed,
                "wall_elapsed_sec": wall_elapsed,
                "totals_skipped_events": totals_skipped_events,
                "dedup": {
                    "duplicates_excluded": duplicate_events,
                    "key": "donor+amount+currency+message (normalized, exact)",
                    "generic_donors": sorted(GENERIC_DONORS),
                    "generic_messages": sorted(GENERIC_MESSAGES),
                },
                "training_data": None if collector is None else {
                    "dir": str(train_dir),
                    "strategies": sorted(train_strategies),
                    "budget": args.train_budget,
                    "budget_applies_to": ["uncertain", "negatives"],
                    "worst_per_event": args.train_worst_per_event,
                    "uncertain_conf_band": [args.train_uncertain_min, args.conf],
                    "image_format": "png",
                    "preview_format": "jpg",
                    "manifest_csv": str(train_dir / "manifest.csv"),
                    "saved_counts": training_summary,
                },
                "outputs": ({
                    "training_data_dir": str(train_dir),
                    "manifest_csv": str(train_dir / "manifest.csv"),
                } if training_mode else {
                    "events_summary_csv": str(events_csv),
                    "totals_by_currency_csv": str(totals_csv),
                    "donations_jsonl": str(jsonl_path),
                    "events_dir": str(events_dir) if save_any_images else "",
                    "events_meta_jsonl": str(events_meta_path) if events_meta_path else "",
                })
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDone.")
    if training_mode:
        print(f"  YOLO detection:    {seconds_to_timestamp(yolo_elapsed)} (compute)")
        print(f"  Total wall clock:  {seconds_to_timestamp(wall_elapsed)}")
        print(f"  Sampled frames:    {processed_frames}")
        print(f"  Raw detections:    {raw_detections_count}")
        print(f"  Donation events:   {len(all_events)}")
        print(f"Training manifest:   {train_dir / 'manifest.csv'}")
        print(f"Training data dir:   {train_dir}")
        return
    print(f"  Mode:              {mode}")
    print(f"  YOLO detection:    {seconds_to_timestamp(yolo_elapsed)} (compute)")
    if vlm_elapsed is not None:
        print(f"  VLM stage:         {seconds_to_timestamp(vlm_elapsed)}")
    else:
        print(f"  VLM stage:         перекрыт детекцией (отдельного времени нет)")
    print(f"  Total wall clock:  {seconds_to_timestamp(wall_elapsed)}")
    print(f"  Sampled frames:    {processed_frames}")
    print(f"  Raw detections:    {raw_detections_count}")
    print(f"  Donation events:   {len(all_events)}")
    if duplicate_events:
        print(f"  Duplicates excluded: {duplicate_events} re-shown donation(s)")
    if totals_skipped_events:
        print(f"  Excluded from totals: {totals_skipped_events} event(s)")
    print(f"Events summary:      {events_csv}")
    print(f"Totals by currency:  {totals_csv}")
    print(f"Raw JSONL:           {jsonl_path}")
    if save_any_images:
        print(f"Best crops/frames:   {events_dir}")
    if events_meta_path:
        print(f"Events meta:         {events_meta_path}")


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YOLO + local Qwen3-VL donation extraction pipeline v6 (memory-optimised)"
    )

    p.add_argument("--project-dir", default=".")
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--video", default="test/video/test_fragment.mp4")
    p.add_argument("--output-dir", default="vlm_runs")
    p.add_argument("--run-name", default="")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--device", default="cpu",
                   help="Передаётся в ultralytics как есть. Для .pt: 'cpu' или CUDA-индекс "
                        "('0', 'cuda:0'). Для OpenVINO-модели: 'intel:cpu' / 'intel:gpu' / "
                        "'intel:npu'. Значение по умолчанию 'cpu'")
    p.add_argument("--img-size", type=parse_img_size, default=640,
                   help="imgsz ultralytics: '640' (квадратный леттербокс) или "
                        "'576,1024' (высота,ширина — для прямоугольной статичной "
                        "OpenVINO-модели)")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--frame-step", type=int, default=10)
    p.add_argument("--padding-x", type=int, default=20)
    p.add_argument("--padding-y", type=int, default=12)

    p.add_argument("--event-gap-sec", type=float, default=3)
    p.add_argument("--event-iou-thr", type=float, default=0.25)
    p.add_argument("--event-center-thr", type=float, default=0.05)
    p.add_argument("--event-split-height-frac", type=float, default=0.10,
                   help="Плашка-своп: если высота бокса меняется больше чем на эту "
                        "долю от опорной (медиана последних детекций события), "
                        "детекция считается НОВЫМ донатом в том же слоте и не "
                        "сливается со старым событием. 0 — выключить. По умолчанию 0.10")
    p.add_argument("--keep-top-candidates", type=int, default=1)
    p.add_argument("--max-processed-frames", type=int, default=0)

    p.add_argument("--vlm-server-url", default="http://127.0.0.1:8081/v1/chat/completions")
    p.add_argument("--vlm-model", default="Qwen3-VL")
    p.add_argument("--vlm-prompt", default=DEFAULT_PROMPT_VERSION,
                   help="Версия промпта из prompts/ (например 'v7') или путь к "
                        f".txt-файлу с промптом. По умолчанию {DEFAULT_PROMPT_VERSION} "
                        "(лучший по бенчмарку на test/gt)")
    p.add_argument("--vlm-timeout", type=int, default=300)
    p.add_argument("--vlm-retries", type=int, default=2,
                   help="Retries for failed VLM requests (network/server errors only)")
    p.add_argument("--vlm-max-tokens", type=int, default=1024)
    p.add_argument("--vlm-temperature", type=float, default=0.0)
    p.add_argument("--skip-vlm", action="store_true",
                   help="Run YOLO/event grouping only, no VLM calls")
    p.add_argument("--sequential", action="store_true",
                   help="Run YOLO fully, then VLM fully (no overlap). "
                        "Gives clean per-stage timing for benchmarking")
    p.add_argument("--images-schema", type=int, default=7,
                   help="Битовая маска сохраняемых изображений: 1=best_crops, "
                        "2=annotated_frames, 4=original_frames (сумма степеней двойки). "
                        "7 — всё (по умолчанию), 0 — ничего (только CSV/JSONL, бывший "
                        "--no-save-images), 1 — только кропы (YOLO-стадия для "
                        "отдельного VLM-сервера), 5 — кропы + оригинальные кадры")

    # Раздельный запуск стадий: сервер A (YOLO) пишет кропы + events_meta.jsonl,
    # сервер B (VLM) обрабатывает их отдельно. Провенанс стрима (дата/платформа/
    # стример) в видео отсутствует — передаётся флагами и попадает в метаданные.
    p.add_argument("--events-meta", choices=["minimal", "full"], default="",
                   help="Писать events_meta.jsonl рядом с отчётами: minimal — "
                        "id/crop/время доната + дата/платформа/стример прогона "
                        "(экономный рабочий вариант для передачи на VLM-сервер); "
                        "full — все детекторные поля событий (полный отчёт для "
                        "проверки/заказчика). По умолчанию выключено")
    p.add_argument("--streamer", default="", help="Имя стримера (в events_meta.jsonl)")
    p.add_argument("--platform", default="", help="Платформа стрима (в events_meta.jsonl)")
    p.add_argument("--stream-date", default="", help="Дата стрима (в events_meta.jsonl)")

    # Training-data collection (for improving the YOLO detector). Independent of
    # the event-crop saving above; pairs naturally with --skip-vlm.
    p.add_argument("--train-select", default="",
                   help="Сбор датасета для дообучения детектора: список через запятую из "
                        f"{','.join(TRAIN_STRATEGIES)} (пусто = выкл). Пишет PNG full-frame + "
                        "YOLO-разметку + превью + manifest.csv. VLM/отчёты не запускаются")
    p.add_argument("--train-dir", default="",
                   help="Куда складывать датасет (по умолчанию <output_dir>/training_data)")
    p.add_argument("--train-budget", type=int, default=200,
                   help="Лимит кадров для uncertain и negatives (каждая стратегия отдельно). "
                        "Сэмплинг — стратифицированный по времени (стрим режется на budget "
                        "корзин, ≤1 кадр/корзина), примеры разнесены по стриму. best/worst/"
                        "random — на событие, бюджетом не ограничены")
    p.add_argument("--train-worst-per-event", type=int, default=1,
                   help="Сколько худших кадров сохранять на событие (стратегия worst)")
    p.add_argument("--train-uncertain-min", type=float, default=0.25,
                   help="Нижняя граница confidence для 'uncertain' (верхняя = --conf). "
                        "При включённой 'uncertain' детекция запускается на этом пороге, "
                        "чтобы увидеть слабые срабатывания в [min, conf)")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.frame_step <= 0:
        raise ValueError("--frame-step must be >= 1")
    if args.keep_top_candidates <= 0:
        raise ValueError("--keep-top-candidates must be >= 1")
    if args.vlm_max_tokens <= 0:
        raise ValueError("--vlm-max-tokens must be >= 1")
    if args.vlm_timeout <= 0:
        raise ValueError("--vlm-timeout must be >= 1")

    if args.train_select.strip():
        sel = {s.strip() for s in args.train_select.split(",") if s.strip()}
        unknown = sel - set(TRAIN_STRATEGIES)
        if unknown:
            raise ValueError(
                f"--train-select: неизвестные стратегии {sorted(unknown)}; "
                f"доступны {list(TRAIN_STRATEGIES)}"
            )
        if args.train_budget <= 0:
            raise ValueError("--train-budget must be >= 1")
        if args.train_worst_per_event <= 0:
            raise ValueError("--train-worst-per-event must be >= 1")
        if "uncertain" in sel and not (0.0 < args.train_uncertain_min < args.conf):
            raise ValueError("--train-uncertain-min должен быть в диапазоне (0, --conf)")

    run_pipeline(args)


if __name__ == "__main__":
    main()
