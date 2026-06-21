#!/usr/bin/env python3
"""Достать структурированные донаты из events_summary.csv готового прогона.

Небольшой утилитарный скрипт для особых случаев: берёт raw_model_response из
CSV-прогона и собирает JSON {"donations": [...]} (например, как заготовку
эталона). Путь к CSV задаётся аргументом — по умолчанию пишет в
extracted_examples.json рядом.

  python3 scripts/json_from_done_ocr.py vlm_runs/<run>/events_summary.csv
  python3 scripts/json_from_done_ocr.py vlm_runs/<run>/events_summary.csv out.json
"""

import argparse
import csv
import json
from pathlib import Path


def extract(input_csv: Path, output_json: Path) -> None:
    examples = []
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop_path = row.get("crop_path", "").strip()
            if not crop_path:
                continue

            raw_response = row.get("raw_model_response", "").strip()
            if not raw_response:
                print(f"⚠️  Пустой raw_model_response для {crop_path} (ошибка VLM в прогоне), пропускаем.")
                continue

            try:
                raw_text = json.loads(raw_response)
            except json.JSONDecodeError:
                print(f"⚠️  Не удалось распарсить JSON для {crop_path}, пропускаем.")
                continue

            if not isinstance(raw_text, dict):
                print(f"⚠️  Ответ модели для {crop_path} не является JSON-объектом, пропускаем.")
                continue

            examples.append({
                "file_name": Path(crop_path).name,
                "donor": raw_text.get("donor"),
                "amount": raw_text.get("amount"),
                "currency": raw_text.get("currency"),
                "message": raw_text.get("message"),
                "needs_review": raw_text.get("needs_review", True),
            })

    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"donations": examples}, f, ensure_ascii=False, indent=4)
    print(f"✅ Готово! {len(examples)} примеров сохранено в {output_json}")


def main() -> None:
    p = argparse.ArgumentParser(description="CSV прогона -> JSON со структурированными донатами")
    p.add_argument("input_csv", type=Path, help="Путь к events_summary.csv прогона")
    p.add_argument("output_json", type=Path, nargs="?", default=Path("extracted_examples.json"),
                   help="Куда сохранить (по умолчанию extracted_examples.json)")
    args = p.parse_args()

    if not args.input_csv.exists():
        raise SystemExit(f"CSV не найден: {args.input_csv}")
    extract(args.input_csv, args.output_json)


if __name__ == "__main__":
    main()
