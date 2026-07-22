#!/usr/bin/env python3
"""Быстрый стенд для подбора VLM-промпта на трудных кропах.

Гоняет ПРОИЗВОЛЬНЫЙ промпт (из файла) по папке кропов, сравнивает с GT и
печатает агрегаты + остаточные ошибки message. В отличие от эталонного
eval_vlm.py:
  * промпт берётся из --prompt-file (не зашит в модуль) — можно сравнивать варианты;
  * при сравнении message переносы строк нормализуются в пробел (--normalize-newlines,
    по умолчанию вкл.) — текущий GT склеен пробелами, а новый промпт может писать \\n;
  * ничего не пишет в test_ocr/ — только stdout, опционально --dump JSONL.

Не трогает продакшн-промпт (prompts/, дефолт donsearcher.DEFAULT_PROMPT_VERSION):
промпт передаётся в call_vlm_for_image явно.
Это рабочий инструмент для итераций, не часть продакшн-пайплайна.
"""
import argparse
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import cv2

import donsearcher

FIELDS = ["donor", "amount", "currency", "message", "fee_covered"]


def norm_msg(v, collapse_newlines=True):
    if v is None:
        return None
    s = str(v).strip()
    if collapse_newlines:
        s = re.sub(r"\s*\n\s*", " ", s)
        s = re.sub(r" {2,}", " ", s)
    return s


def field_similarity(field, pred, gt, collapse_newlines=True):
    if field == "fee_covered":
        return 1.0 if bool(pred) == bool(gt) else 0.0
    if field == "amount":
        if pred is None and gt is None:
            return 1.0
        if pred is None or gt is None:
            return 0.0
        try:
            return 1.0 if float(pred) == float(gt) else 0.0
        except (TypeError, ValueError):
            return 0.0
    if field == "message":
        pred, gt = norm_msg(pred, collapse_newlines), norm_msg(gt, collapse_newlines)
    if pred is None and gt is None:
        return 1.0
    if pred is None or gt is None:
        return 0.0
    return SequenceMatcher(None, str(pred).strip(), str(gt).strip(), autojunk=False).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="test/gt/core_cases")
    ap.add_argument("--truth", default="test/gt/true_ocr.json")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--server-url", default="http://127.0.0.1:8082/v1/chat/completions")
    ap.add_argument("--model-name", default="Qwen3-VL-8B")
    ap.add_argument("--no-normalize-newlines", action="store_true",
                    help="не схлопывать \\n в пробел при сравнении message")
    ap.add_argument("--dump", help="сохранить предсказания в JSONL")
    ap.add_argument("--tag", default="", help="метка прогона в выводе")
    ap.add_argument("--retries", type=int, default=1,
                    help="ретраи на сетевую ошибку (мало — чтобы не копить зависшие при обрыве)")
    ap.add_argument("--timeout", type=int, default=600, help="клиентский таймаут на запрос, с")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="лимит генерации; самое длинное сообщение в GT ~420 ток., 512 с запасом, "
                         "режет вырожденные простыни (модель иногда зацикливается до лимита)")
    a = ap.parse_args()

    collapse = not a.no_normalize_newlines
    prompt = Path(a.prompt_file).read_text()
    gt = json.load(open(a.truth, encoding="utf-8"))
    gt_examples = gt.get("donations") or gt.get("examples") or []
    gt_by_name = {ex["file_name"]: ex for ex in gt_examples}
    fields = list(FIELDS) if any("fee_covered" in ex for ex in gt_examples) else FIELDS[:4]

    images = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() == ".png")
    rows = []
    dump = []
    print(f"\n=== {a.tag or a.prompt_file} | {a.model_name} | {len(images)} кропов | newline→space={collapse} ===")
    for p in images:
        ex = gt_by_name.get(p.name)
        if ex is None:
            print(f"  ⚠ нет GT для {p.name}")
            continue
        t0 = time.perf_counter()
        raw, parsed, err = donsearcher.call_vlm_for_image(
            crop_bgr=cv2.imread(str(p)), server_url=a.server_url,
            model_name=a.model_name, prompt=prompt,
            retries=a.retries, timeout_sec=a.timeout, max_tokens=a.max_tokens)
        dt = time.perf_counter() - t0
        if err or parsed is None:
            parsed = {f: None for f in fields}
            parsed["error"] = err or "parse_fail"
        sims = {f: field_similarity(f, parsed.get(f), ex.get(f), collapse) for f in fields}
        all_ok = all(s == 1.0 for s in sims.values())
        print(f"  [{len(rows)+1:2}/{len(images)}] {p.name.replace('_best_detector_crop.png','')}"
              f"  {dt:5.1f}s  msg={sims['message']:.2f}{'  ✓' if all_ok else ''}", flush=True)
        rows.append((p.name, sims, all_ok, dt, parsed.get(f"message"), ex.get("message")))
        dump.append({"file_name": p.name, **{f: parsed.get(f) for f in fields}})

    n = len(rows)
    if not n:
        print("нет данных")
        return
    print(f"\nПолностью верно: {sum(r[2] for r in rows)}/{n} ({sum(r[2] for r in rows)/n*100:.1f}%)")
    for f in fields:
        exact = sum(1 for r in rows if r[1][f] == 1.0) / n
        mean = sum(r[1][f] for r in rows) / n
        print(f"  {f:12} exact={exact*100:5.1f}%  mean={mean:.3f}")
    print(f"  latency mean={sum(r[3] for r in rows)/n:.1f}s")

    print("\n--- остаточные ошибки message (sim<0.97) ---")
    for name, sims, _, _, pm, gm in sorted(rows, key=lambda r: r[1]["message"]):
        if sims["message"] < 0.97:
            print(f"[{sims['message']:.2f}] {name.replace('_best_detector_crop.png','')}")
            print(f"   PRED: {norm_msg(pm, collapse)!r}")
            print(f"   GT  : {norm_msg(gm, collapse)!r}")

    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as fh:
            for d in dump:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"\ndump → {a.dump}")


if __name__ == "__main__":
    main()
