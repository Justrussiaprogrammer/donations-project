import json
import csv
import time
import statistics
from pathlib import Path
import cv2

from difflib import SequenceMatcher

import vlm_pipeline

IMAGE_FOLDER = "test/gt/donations"
SERVER_URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL_NAME = "Qwen3-VL-8B-Q4_K_M"

# Промпт для оценки. По умолчанию — встроенный в пайплайн (продакшн), но через
# --prompt-file можно прогнать любой файл-вариант (prompts/v10.txt и т.п.), чтобы
# сравнивать промпты тем же каноническим eval'ом. ВАЖНО: раньше скрипт молча брал
# только VLM_PROMPT, и прогон под другим --model-name всё равно использовал продакшн-
# промпт — отсюда путаница «v10 == v7». Теперь промпт явный.
PROMPT = vlm_pipeline.VLM_PROMPT

TRUTH_OCR_FILE = Path("test/gt/true_ocr.json")
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
            model_name=model_name,
            prompt=PROMPT
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


def eval_fields_for_gt(gt_examples):
    """
    Список сравниваемых полей. fee_covered добавляется, только если он реально
    размечен в эталоне — пока GT без него, поле не штрафует модели и не ломает
    all_correct (обратная совместимость со старой разметкой).
    """
    fields = ['donor', 'amount', 'currency', 'message']
    if any('fee_covered' in ex for ex in gt_examples):
        fields = fields + ['fee_covered']
    return fields


def compare_fields(pred, gt, fields=('donor', 'amount', 'currency', 'message')):
    """
    Сравнивает предсказанные и эталонные поля с оценкой степени сходства.
    Возвращает:
      - comp: словарь с деталями по каждому полю
      - all_match: bool, True если все поля совпали полностью (similarity == 1.0)
    """
    comp = {}
    all_match = True

    for f in fields:
        pred_val = pred.get(f)
        gt_val = gt.get(f)

        if f == 'fee_covered':
            # Булев флаг: точное равенство (None трактуем как False).
            similarity = 1.0 if bool(pred_val) == bool(gt_val) else 0.0
        elif pred_val is None and gt_val is None:
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
            # autojunk=False: на строках >200 символов дефолтная эвристика
            # SequenceMatcher помечает частые символы как «мусор» и выкидывает их
            # из матчинга, занижая ratio (длинные message получали ~0.4 при 4
            # отличиях из 295). Для верной оценки длинных сообщений отключаем.
            similarity = SequenceMatcher(None, p_str, g_str, autojunk=False).ratio()

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
    # Пер-донатные JSON-результаты (и сырые ответы при сбое парсинга) — в отдельной
    # подпапке, чтобы её можно было свернуть и смотреть на сводные summary.csv /
    # timing.json в корне прогона.
    per_image_dir = results_dir / "donations"
    per_image_dir.mkdir(parents=True, exist_ok=True)

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
    eval_fields = eval_fields_for_gt(gt_examples)

    run_t0 = time.perf_counter()
    for img_path in image_files:
        true_json = gt_by_name.get(img_path.name)
        if true_json is None:
            print(f"⚠ Нет эталона для {img_path.name}, пропускаю.")
            continue
        print(f"Обрабатываю: {img_path.name}")

        _t0 = time.perf_counter()
        raw_text, parsed, error = vlm_pipeline.call_vlm_for_image(
            crop_bgr=cv2.imread(str(img_path)),
            server_url=SERVER_URL,
            model_name=MODEL_NAME,
            prompt=PROMPT
        )
        latency_sec = time.perf_counter() - _t0
        print(f"  ⏱ {latency_sec:.2f} с")
        if error:
            print(f"  ❌ Ошибка вызова: {error}")
            parsed = None
        if parsed is None:
            print(f"  ❌ Не удалось извлечь JSON из ответа:\n{raw_text[:200]}")
            debug_file = per_image_dir / f"{img_path.stem}_raw.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            print(f"  → сырой ответ сохранён в {debug_file}")
            result = {
                "image": img_path.name,
                "all_fields_correct": False,
                "latency_sec": latency_sec,
                "fields": {f: {"predicted": None, "ground_truth": true_json.get(f), "similarity": 0.0} for f in eval_fields},
                "error": error or "json_parse_failed"
            }
        else:
            field_comparison, all_correct = compare_fields(parsed, true_json, eval_fields)
            result = {
                "image": img_path.name,
                "all_fields_correct": all_correct,
                "latency_sec": latency_sec,
                "fields": field_comparison,
                "raw_response": raw_text
            }

        res_file = per_image_dir / f"{img_path.stem}_result.json"
        with open(res_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  → результат сохранён: {res_file}")

        row = {"image": img_path.name}
        for f in eval_fields:
            row[f"{f}_similarity"] = result["fields"][f]["similarity"]
        row["all_correct"] = result["all_fields_correct"]
        row["latency_sec"] = round(latency_sec, 3)
        summary_data.append(row)

    wall_elapsed_sec = time.perf_counter() - run_t0

    if summary_data:
        # Пер-изображенные строки (детализация) — в per_image.csv. Раньше это
        # лежало в summary.csv, но по таблице из 178 строк нельзя было понять
        # ситуацию по прогону; summary.csv теперь — агрегаты (см. ниже).
        per_image_file = results_dir / "per_image.csv"
        with open(per_image_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"\nДетализация по изображениям сохранена в {per_image_file}")

        fields = eval_fields
        total = len(summary_data)
        correct = sum(1 for r in summary_data if r["all_correct"])

        # Агрегаты точности: по каждому полю — среднее сходство и доля точных
        # совпадений (similarity == 1.0).
        field_mean_sim = {
            f: statistics.mean(r[f"{f}_similarity"] for r in summary_data)
            for f in fields
        }
        field_exact_rate = {
            f: sum(1 for r in summary_data if r[f"{f}_similarity"] == 1.0) / total
            for f in fields
        }

        # Статистика по времени вызовов модели.
        latencies = sorted(r["latency_sec"] for r in summary_data)
        n = len(latencies)
        p95 = latencies[min(n - 1, int(round(0.95 * (n - 1))))]
        timing = {
            "model_name": MODEL_NAME,
            "server_url": SERVER_URL,
            "images": n,
            "wall_elapsed_sec": round(wall_elapsed_sec, 3),
            "vlm_total_sec": round(sum(latencies), 3),
            "latency_mean_sec": round(statistics.mean(latencies), 3),
            "latency_median_sec": round(statistics.median(latencies), 3),
            "latency_p95_sec": round(p95, 3),
            "latency_min_sec": round(latencies[0], 3),
            "latency_max_sec": round(latencies[-1], 3),
        }
        timing_file = results_dir / "timing.json"
        with open(timing_file, 'w', encoding='utf-8') as f:
            json.dump(timing, f, ensure_ascii=False, indent=2)

        # summary.csv — одна сводная таблица «метрика, значение» по всему прогону:
        # точность (полные совпадения + по полям) и тайминг в одном месте.
        summary_rows = [
            ("model_name", MODEL_NAME),
            ("images", total),
            ("all_fields_correct", correct),
            ("all_fields_correct_rate", round(correct / total, 4)),
        ]
        for f in fields:
            summary_rows.append((f"{f}_exact_rate", round(field_exact_rate[f], 4)))
        for f in fields:
            summary_rows.append((f"{f}_mean_similarity", round(field_mean_sim[f], 4)))
        summary_rows += [
            ("wall_elapsed_sec", timing["wall_elapsed_sec"]),
            ("vlm_total_sec", timing["vlm_total_sec"]),
            ("latency_mean_sec", timing["latency_mean_sec"]),
            ("latency_median_sec", timing["latency_median_sec"]),
            ("latency_p95_sec", timing["latency_p95_sec"]),
            ("latency_min_sec", timing["latency_min_sec"]),
            ("latency_max_sec", timing["latency_max_sec"]),
        ]
        csv_file = results_dir / "summary.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerows(summary_rows)
        print(f"Сводка по прогону сохранена в {csv_file}")

        print(f"Итого: {correct}/{total} изображений распознаны полностью правильно (все поля совпали).")
        print(
            "Точность по полям (точные совпадения): "
            + ", ".join(f"{f} {field_exact_rate[f] * 100:.1f}%" for f in fields)
        )
        print(
            "Время: "
            f"всего прогона {timing['wall_elapsed_sec']:.1f} с, "
            f"в VLM {timing['vlm_total_sec']:.1f} с на {n} вызовов | "
            f"на вызов: среднее {timing['latency_mean_sec']:.2f} с, "
            f"медиана {timing['latency_median_sec']:.2f} с, "
            f"p95 {timing['latency_p95_sec']:.2f} с, "
            f"min {timing['latency_min_sec']:.2f} с, "
            f"max {timing['latency_max_sec']:.2f} с"
        )
        print(f"Тайминг сохранён в {timing_file}")
    else:
        print("Нет обработанных изображений.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Оценка VLM на кропах против эталона (eval) или выгрузка предсказаний (dump)"
    )
    parser.add_argument("--mode", choices=["eval", "dump"], default="eval",
                        help="eval — сравнить с true_ocr.json; dump — выгрузить предсказания в JSONL")
    parser.add_argument("--images", default=IMAGE_FOLDER, help="папка с кропами .png")
    parser.add_argument("--truth", default=str(TRUTH_OCR_FILE),
                        help="JSON-эталон (для --mode eval), матчинг по file_name")
    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--prompt-file", default=None,
                        help="файл с промптом (по умолчанию — встроенный vlm_pipeline.VLM_PROMPT)")
    parser.add_argument("--out", default=str(TEST_OCR_FILE), help="выходной JSONL (для --mode dump)")
    a = parser.parse_args()

    # Оба режима читают эти имена как глобали — переопределяем под аргументы.
    IMAGE_FOLDER = a.images
    TRUTH_OCR_FILE = Path(a.truth)
    SERVER_URL = a.server_url
    MODEL_NAME = a.model_name
    if a.prompt_file:
        PROMPT = Path(a.prompt_file).read_text()
        print(f"Промпт из файла: {a.prompt_file} ({len(PROMPT)} симв.)")

    if a.mode == "dump":
        save_jsons_from_model(a.images, Path(a.out), server_url=a.server_url, model_name=a.model_name)
    else:
        main()
