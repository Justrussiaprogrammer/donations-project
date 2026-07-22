#!/usr/bin/env python3
"""
VLM/OCR-стадия по готовым кропам — сервер B при раздельном запуске стадий.

Сервер A гонит YOLO-стадию и отдаёт кропы + метаданные:
  python3 scripts/fast_pipeline.py --video ... --skip-vlm --images-schema 1 \
      --events-meta minimal --streamer Ник --platform twitch --stream-date 2026-07-06

Кропы (events/best_crops/*.png) и events_meta.jsonl переносятся сюда (rsync,
сетевой диск и т.п.), после чего этот скрипт собирает ПОЛНЫЕ отчёты прогона —
те же, что у обычного пайплайна: events_summary.csv, totals_by_currency.csv,
donations.jsonl (с дедупом повторно показанных донатов), run_metadata.json.
Формат строк общий с пайплайном (donsearcher.build_event_rows): при full-мета
детекторные поля заполнены целиком, при minimal — пустые (totals/дедуп работают,
им нужны только VLM-поля).

Запросы к VLM идут ПАРАЛЛЕЛЬНО (--concurrency N): главный рычаг пропускной
способности на больших объёмах. llama-server должен быть поднят с -np N
(слотов не меньше, чем concurrency), иначе запросы просто встанут в очередь
на сервере.

Примеры:
  python3 scripts/vlm_stage.py --crops incoming/best_crops --overwrite
  python3 scripts/vlm_stage.py --crops incoming/best_crops \
      --meta incoming/events_meta.jsonl --concurrency 4 --overwrite

events_meta.jsonl ищется автоматически: рядом с папкой кропов, в ней самой
или на уровень выше (раскладка run-каталога сервера A). Без метаданных скрипт
тоже работает: event_id берётся из имени файла (event_0001_...), времена пустые.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import cv2

import donsearcher as vp

# Имя кропа от сервера A: event_0001_best_detector_crop.png
CROP_ID_RE = re.compile(r"event_(\d+)_")


def find_meta(crops_dir: Path, meta_arg: str) -> Optional[Path]:
    if meta_arg:
        path = Path(meta_arg).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"events_meta.jsonl не найден: {path}")
        return path
    for candidate in (crops_dir / "events_meta.jsonl",
                      crops_dir.parent / "events_meta.jsonl",
                      crops_dir.parent.parent / "events_meta.jsonl"):
        if candidate.is_file():
            return candidate
    return None


def load_meta(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """events_meta.jsonl -> (run-запись, метаданные событий по имени кропа)."""
    run_record: dict[str, Any] = {}
    by_crop: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "run":
                run_record = rec
                continue
            meta = vp.meta_from_record(rec)
            if meta["crop_path"]:
                by_crop[meta["crop_path"]] = meta
            else:
                print(f"⚠ Событие {meta['event_id']} в метаданных без кропа — "
                      f"попадёт в отчёт с ошибкой crop_missing.")
                by_crop[f"__no_crop_{meta['event_id']}"] = meta
    return run_record, by_crop


def fallback_meta(crop_name: str, next_id: int) -> dict[str, Any]:
    """Метаданные для кропа, которого нет в events_meta.jsonl (или мета нет вовсе)."""
    m = CROP_ID_RE.search(crop_name)
    meta = {k: "" for k in vp.FULL_META_FIELDS}
    meta["event_id"] = int(m.group(1)) if m else next_id
    meta["crop_path"] = crop_name
    return meta


def build_jobs(crops_dir: Path, by_crop: dict[str, dict[str, Any]]
               ) -> list[tuple[dict[str, Any], Optional[Path]]]:
    """Список заданий (meta, путь_к_кропу|None). None -> кроп потерялся при переносе."""
    jobs: list[tuple[dict[str, Any], Optional[Path]]] = []
    matched: set[str] = set()
    crop_files = sorted(crops_dir.glob("*.png"))
    if not crop_files and not by_crop:
        raise FileNotFoundError(f"В {crops_dir} нет *.png и метаданные пусты — нечего обрабатывать.")
    # 10^9 + позиция: заведомо не пересекается с настоящими event_id
    for i, p in enumerate(crop_files):
        meta = by_crop.get(p.name)
        if meta is not None:
            matched.add(p.name)
        else:
            if by_crop:
                print(f"⚠ Кроп {p.name} отсутствует в метаданных — обрабатываю без них.")
            meta = fallback_meta(p.name, next_id=10**9 + i)
        jobs.append((meta, p))
    for name, meta in by_crop.items():
        if name not in matched:
            print(f"⚠ Кроп {meta['crop_path'] or name} из метаданных не найден в {crops_dir}.")
            jobs.append((meta, None))
    return jobs


def _id_key(row: dict[str, Any]) -> tuple[int, Any]:
    try:
        return (0, int(row["event_id"]))
    except (ValueError, TypeError):
        return (1, str(row["event_id"]))


def process_jobs(jobs: list[tuple[dict[str, Any], Optional[Path]]],
                 args: argparse.Namespace, prompt_text: str
                 ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Прогнать задания через VLM в --concurrency потоков."""
    done_count = 0
    count_lock = threading.Lock()
    total = len(jobs)

    def process_one(job: tuple[dict[str, Any], Optional[Path]]
                    ) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal done_count
        meta, crop_path = job
        try:
            if crop_path is None:
                rows = vp.build_event_rows(meta, "", None, "crop_missing")
            else:
                crop_bgr = cv2.imread(str(crop_path))
                if crop_bgr is None:
                    rows = vp.build_event_rows(meta, "", None, "crop_load_failed")
                else:
                    raw_text, parsed, model_error = vp.call_vlm_for_image(
                        crop_bgr=crop_bgr,
                        server_url=args.vlm_server_url,
                        model_name=args.vlm_model,
                        timeout_sec=args.vlm_timeout,
                        max_tokens=args.vlm_max_tokens,
                        temperature=args.vlm_temperature,
                        prompt=prompt_text,
                        retries=args.vlm_retries,
                    )
                    rows = vp.build_event_rows(meta, raw_text, parsed, model_error)
        except Exception as exc:  # как в _vlm_worker: событие не теряется
            rows = vp.build_event_rows(meta, "", None,
                                       f"worker_exception: {type(exc).__name__}: {exc}")
        with count_lock:
            done_count += 1
            n = done_count
        event_row = rows[0]
        print(f"[VLM] {n}/{total} event {event_row['event_id']}: "
              f"donor={event_row.get('donor') or '?'} amount={event_row.get('amount') or '?'}"
              + (f" [{event_row['model_error']}]" if event_row["model_error"] else ""))
        return rows

    if args.concurrency <= 1:
        return [process_one(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        return list(ex.map(process_one, jobs))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VLM/OCR-стадия по готовым кропам (сервер B) -> полные отчёты прогона"
    )
    p.add_argument("--crops", required=True,
                   help="Папка с кропами *.png от YOLO-стадии (events/best_crops)")
    p.add_argument("--meta", default="",
                   help="Путь к events_meta.jsonl (по умолчанию ищется в папке кропов, "
                        "рядом с ней и на уровень выше)")
    p.add_argument("--output-dir", default="vlm_runs")
    p.add_argument("--run-name", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Параллельные запросы к VLM. llama-server должен быть поднят "
                        "с -np N (N >= concurrency). По умолчанию 1")
    # Те же имена/дефолты, что у пайплайна — команды переносимы между стадиями.
    p.add_argument("--vlm-server-url", default="http://127.0.0.1:8081/v1/chat/completions")
    p.add_argument("--vlm-model", default="Qwen3-VL")
    p.add_argument("--vlm-prompt", default=vp.DEFAULT_PROMPT_VERSION,
                   help="Версия промпта из prompts/ или путь к .txt")
    p.add_argument("--vlm-timeout", type=int, default=300)
    p.add_argument("--vlm-retries", type=int, default=2)
    p.add_argument("--vlm-max-tokens", type=int, default=1024)
    p.add_argument("--vlm-temperature", type=float, default=0.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be >= 1")

    crops_dir = Path(args.crops).expanduser().resolve()
    if not crops_dir.is_dir():
        raise FileNotFoundError(f"Папка кропов не найдена: {crops_dir}")

    # Fail fast при опечатке в версии промпта — до каких-либо VLM-вызовов.
    prompt_text, prompt_version = vp.load_prompt(args.vlm_prompt)

    meta_path = find_meta(crops_dir, args.meta)
    run_record: dict[str, Any] = {}
    by_crop: dict[str, dict[str, Any]] = {}
    if meta_path is not None:
        run_record, by_crop = load_meta(meta_path)
        print(f"Метаданные:   {meta_path} ({run_record.get('meta', '?')}, "
              f"{len(by_crop)} событий)")
    else:
        print("Метаданные:   не найдены — event_id из имён файлов, времена пустые.")

    run_name = args.run_name or (
        f"{vp.safe_filename(Path(run_record['video']).stem)}_vlm_stage"
        if run_record.get("video") else f"{vp.safe_filename(crops_dir.name)}_vlm_stage")
    output_dir = (Path(args.output_dir) / run_name).resolve()
    if output_dir.exists():
        if args.overwrite:
            import shutil
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output dir already exists and is not empty: {output_dir}\n"
                f"Use --overwrite to replace it or --run-name to pick another name."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Кропы:        {crops_dir}")
    print(f"Output:       {output_dir}")
    print(f"VLM server:   {args.vlm_server_url} (concurrency {args.concurrency})")
    print(f"VLM model:    {args.vlm_model} (промпт {prompt_version})\n")

    jobs = build_jobs(crops_dir, by_crop)
    print(f"Starting VLM stage ({len(jobs)} events)...")
    t0 = time.time()
    results = process_jobs(jobs, args, prompt_text)
    vlm_elapsed = round(time.time() - t0, 2)

    results.sort(key=lambda r: _id_key(r[0]))
    event_rows = [r[0] for r in results]
    jsonl_rows = [r[1] for r in results]
    duplicate_events = vp.dedup_events(event_rows, jsonl_rows)

    events_csv = output_dir / "events_summary.csv"
    totals_csv = output_dir / "totals_by_currency.csv"
    jsonl_path = output_dir / "donations.jsonl"
    totals_rows, totals_skipped_events = vp.build_totals_rows(event_rows)
    vp.write_csv(events_csv, event_rows)
    vp.write_csv(totals_csv, totals_rows)
    vp.write_jsonl(jsonl_path, jsonl_rows)

    error_events = sum(1 for r in event_rows if r["model_error"])
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stage": "vlm_only",
                "run_name": run_name,
                "crops_dir": str(crops_dir),
                "events_meta": str(meta_path) if meta_path else "",
                # Провенанс стрима с сервера A (streamer/platform/stream_date/video).
                "source_run": run_record,
                "events": len(event_rows),
                "error_events": error_events,
                "vlm_server_url": args.vlm_server_url,
                "vlm_model": args.vlm_model,
                "vlm_prompt_version": prompt_version,
                "vlm_prompt": prompt_text,
                "vlm_retries": args.vlm_retries,
                "vlm_timeout": args.vlm_timeout,
                "vlm_max_tokens": args.vlm_max_tokens,
                "vlm_temperature": args.vlm_temperature,
                "concurrency": args.concurrency,
                "vlm_elapsed_sec": vlm_elapsed,
                "totals_skipped_events": totals_skipped_events,
                "dedup": {
                    "duplicates_excluded": duplicate_events,
                    "key": "donor+amount+currency+message (normalized, exact)",
                    "generic_donors": sorted(vp.GENERIC_DONORS),
                    "generic_messages": sorted(vp.GENERIC_MESSAGES),
                },
                "outputs": {
                    "events_summary_csv": str(events_csv),
                    "totals_by_currency_csv": str(totals_csv),
                    "donations_jsonl": str(jsonl_path),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDone.")
    print(f"  VLM stage:         {vp.seconds_to_timestamp(vlm_elapsed)} "
          f"(concurrency {args.concurrency})")
    print(f"  Donation events:   {len(event_rows)}")
    if error_events:
        print(f"  Events with errors: {error_events}")
    if duplicate_events:
        print(f"  Duplicates excluded: {duplicate_events} re-shown donation(s)")
    if totals_skipped_events:
        print(f"  Excluded from totals: {totals_skipped_events} event(s)")
    print(f"Events summary:      {events_csv}")
    print(f"Totals by currency:  {totals_csv}")
    print(f"Raw JSONL:           {jsonl_path}")


if __name__ == "__main__":
    main()
