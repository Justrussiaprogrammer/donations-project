import json
import csv
from pathlib import Path
import cv2

from difflib import SequenceMatcher

import vlm_pipeline

IMAGE_FOLDER = "some_crops"
SERVER_URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL_NAME = "Qwen3-VL-8B-Q4_K_M"

TRUTH_OCR_FILE = Path("true_ocr.json")
TEST_OCR_FILE = Path("test_ocr.jsonl")


def save_jsons_from_model(folder, out_folder, server_url=SERVER_URL, model_name=MODEL_NAME):
    image_extensions = ('.png', '.PNG')
    image_files = sorted([f for f in Path(folder).iterdir() if f.suffix in image_extensions])

    summary_data = []
    for img_path in image_files:
        print(f"Обрабатываю: {img_path.name}")

        raw_text, parsed, error = vlm_pipeline.call_vlm_for_image(
            crop_bgr=cv2.imread(str(img_path)),
            server_url=server_url,
            model_name=model_name
        )

        if error:
            print(f"  ❌ Ошибка вызова: {error}")
            parsed = {}
        elif parsed is None:
            print(f"  ❌ Не удалось извлечь JSON из ответа:\n{raw_text[:200]}")
            parsed = {}

        summary_data.append({
            "file_name": img_path.name,
            "donor": parsed.get("donor", ""),
            "amount": parsed.get("amount", ""),
            "currency": parsed.get("currency", ""),
            "message": parsed.get("message", ""),
            "needs_review": parsed.get("needs_review", True),
            "error": error,
        })
    
    print("Модель проанализировала все изображения")
    vlm_pipeline.write_jsonl(out_folder, summary_data)
    print("Все json донатов сохранены")


def compare_fields(pred, gt):
    """
    Сравнивает предсказанные и эталонные поля с оценкой степени сходства.
    Возвращает:
      - comp: словарь с деталями по каждому полю
      - all_match: bool, True если все поля совпали полностью (similarity == 1.0)
    """
    fields = ['donor', 'amount', 'currency', 'message']
    comp = {}
    all_match = True

    for f in fields:
        pred_val = pred.get(f)
        gt_val = gt.get(f)

        if pred_val is None and gt_val is None:
            similarity = 1.0
        elif pred_val is None or gt_val is None:
            similarity = 0.0
        elif f == 'amount':
            # Числовое сравнение: 1500 == 1500.0 == "1500"
            try:
                similarity = 1.0 if float(pred_val) == float(gt_val) else 0.0
            except (TypeError, ValueError):
                similarity = 0.0
        else:
            p_str = str(pred_val).strip()
            g_str = str(gt_val).strip()
            similarity = SequenceMatcher(None, p_str, g_str).ratio()

        comp[f] = {
            "predicted": pred_val,
            "ground_truth": gt_val,
            "similarity": similarity
        }

        if similarity != 1.0:
            all_match = False

    return comp, all_match


def main():
    results_dir = Path("test_ocr") / MODEL_NAME
    results_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = ('.png', '.PNG')
    image_files = sorted([f for f in Path(IMAGE_FOLDER).iterdir() if f.suffix in image_extensions])

    summary_data = []

    with open(TRUTH_OCR_FILE, 'r', encoding='utf-8') as f:
        gt = json.load(f)
        if gt is None:
            print("  ⚠ Нет эталона, пропускаю.")
            return

    # Сопоставление эталона по file_name, а не по порядку файлов:
    # пропуск/добавление одного кропа не сдвигает всю разметку.
    gt_examples = gt.get("donations") or gt.get("examples") or []
    gt_by_name = {ex.get("file_name"): ex for ex in gt_examples}

    for img_path in image_files:
        true_json = gt_by_name.get(img_path.name)
        if true_json is None:
            print(f"⚠ Нет эталона для {img_path.name}, пропускаю.")
            continue
        print(f"Обрабатываю: {img_path.name}")

        raw_text, parsed, error = vlm_pipeline.call_vlm_for_image(
            crop_bgr=cv2.imread(str(img_path)),
            server_url=SERVER_URL,
            model_name=MODEL_NAME
        )
        if error:
            print(f"  ❌ Ошибка вызова: {error}")
            parsed = None
        if parsed is None:
            print(f"  ❌ Не удалось извлечь JSON из ответа:\n{raw_text[:200]}")
            debug_file = results_dir / f"{img_path.stem}_raw.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            print(f"  → сырой ответ сохранён в {debug_file}")
            result = {
                "image": img_path.name,
                "all_fields_correct": False,
                "fields": {f: {"predicted": None, "ground_truth": true_json.get(f), "similarity": 0.0} for f in ['donor','amount','currency','message']},
                "error": error or "json_parse_failed"
            }
        else:
            field_comparison, all_correct = compare_fields(parsed, true_json)
            result = {
                "image": img_path.name,
                "all_fields_correct": all_correct,
                "fields": field_comparison,
                "raw_response": raw_text
            }

        res_file = results_dir / f"{img_path.stem}_result.json"
        with open(res_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  → результат сохранён: {res_file}")

        row = {"image": img_path.name}
        for f in ['donor','amount','currency','message']:
            row[f"{f}_similarity"] = result["fields"][f]["similarity"]
        row["all_correct"] = result["all_fields_correct"]
        summary_data.append(row)

    if summary_data:
        csv_file = results_dir / "summary.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"\nСводка сохранена в {csv_file}")

        total = len(summary_data)
        correct = sum(1 for r in summary_data if r["all_correct"])
        print(f"Итого: {correct}/{total} изображений распознаны полностью правильно (все поля совпали).")
    else:
        print("Нет обработанных изображений.")

if __name__ == "__main__":
    # main()
    save_jsons_from_model(IMAGE_FOLDER, TEST_OCR_FILE)
