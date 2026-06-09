import csv
import json

INPUT_CSV = "vlm_runs/2025-01-08_vlm_v4_run/events_summary.csv"
OUTPUT_JSON = "extracted_examples.json"

examples = []

with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        crop_path = row.get('crop_path', '').strip()

        if not crop_path:
            continue
        else:
            try:
                raw_text = json.loads(row.get("raw_model_response"))

                example = {
                    'file_name': crop_path.split('/')[-1],
                    "donor": raw_text['donor'],
                    "amount": raw_text['amount'],
                    "currency": raw_text['currency'],
                    "message": raw_text['message'],
                    "needs_review": raw_text['needs_review']
                }
            except json.JSONDecodeError:
                print(f"⚠️  Не удалось распарсить JSON для {crop_path}, пропускаем.")
                continue

        examples.append(example)
    

output = {'donations': examples}
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print(f"✅ Готово! {len(examples)} примеров сохранено в {OUTPUT_JSON}")
