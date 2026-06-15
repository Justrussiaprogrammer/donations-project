import csv
import json

INPUT_CSV = "vlm_runs/2025-01-08_vlm_v5_run/events_summary.csv"
OUTPUT_JSON = "extracted_examples.json"

examples = []

with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        crop_path = row.get('crop_path', '').strip()

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
            'file_name': crop_path.split('/')[-1],
            "donor": raw_text.get('donor'),
            "amount": raw_text.get('amount'),
            "currency": raw_text.get('currency'),
            "message": raw_text.get('message'),
            "needs_review": raw_text.get('needs_review', True)
        })
    

output = {'donations': examples}
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print(f"✅ Готово! {len(examples)} примеров сохранено в {OUTPUT_JSON}")
