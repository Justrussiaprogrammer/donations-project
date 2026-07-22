#!/usr/bin/env bash
# Бенчмарк детектора на серверном CPU (EPYC/Xeon и т.п.).
#
# Что нужно на сервере: клон репозитория + models/ (best.pt, best_1024x1024.pt
# и оба *_openvino_model) + тестовое видео. GT/эталоны не нужны.
#
# Первый запуск на голом Ubuntu:
#   ./scripts/server_bench_cpu.sh --setup          # apt + venv + pip + сборка cpp
# Сам бенчмарк:
#   ./scripts/server_bench_cpu.sh [путь_к_видео]   # дефолт test/video/test_fragment.mp4
#
# Выход: три уровня цифр, по которым сразу видно, стоит ли арендовать:
#   1) benchmark_app  — потолок инференса OpenVINO (fps, без видео);
#   2) bench_infer.py — потолок через ultralytics (как в py-движке);
#   3) benchmark_detector.py — честный end-to-end (декод+инференс+события).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--setup" ]]; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq ffmpeg cmake g++ python3-venv python3-dev
    python3 -m venv donate_env
    # shellcheck disable=SC1091
    source donate_env/bin/activate
    pip install -q -r requirements.txt
    ./cpp/build.sh
    echo "=== setup готов; запускайте: $0 [видео] ==="
    exit 0
fi

VIDEO="${1:-test/video/test_fragment.mp4}"
[[ -f "$VIDEO" ]] || { echo "нет видео: $VIDEO"; exit 1; }
[[ -x cpp/fast_detector ]] || { echo "cpp/fast_detector не собран — ./cpp/build.sh (или --setup)"; exit 1; }
# shellcheck disable=SC1091
[[ -f donate_env/bin/activate ]] && source donate_env/bin/activate

OUT="benchmarks/server_cpu_$(hostname)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")

{
echo "=== Железо ==="
lscpu | grep -E "Model name|^CPU\(s\)|Thread|MHz|avx512" || true
grep -o 'avx512[a-z_]*' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
echo "Видео: $VIDEO (${DUR}s)"

echo
echo "=== 1. Потолок инференса OpenVINO (benchmark_app, THROUGHPUT, 15s) ==="
for dir in models/best_openvino_model models/best_1024x1024_openvino_model; do
    xml=$(ls "$dir"/*.xml)
    echo "--- $dir ---"
    benchmark_app -m "$xml" -d CPU -hint throughput -t 15 2>/dev/null \
        | grep -E "Throughput|Median|count" || echo "  benchmark_app недоступен (pip install openvino)"
done

echo
echo "=== 2. Потолок через ultralytics (bench_infer.py) ==="
python3 scripts/bench_infer.py --model models/best_1024x1024_openvino_model \
    --device intel:cpu --img-size 576,1024 --n 50

echo
echo "=== 3. End-to-end: детекция видео (py-openvino + cpp-cpu, 2 повтора) ==="
python3 scripts/benchmark_detector.py --video "$VIDEO" --conf 0.5 \
    --engines py-openvino,cpp-cpu --no-save-images --repeats 2 \
    --img-size 384,640 \
    --model-pt models/best.pt --model-openvino models/best_openvino_model
python3 scripts/benchmark_detector.py --video "$VIDEO" --conf 0.5 \
    --engines py-openvino,cpp-cpu --no-save-images --repeats 2 \
    --img-size 576,1024 \
    --model-pt models/best_1024x1024.pt --model-openvino models/best_1024x1024_openvino_model

echo
echo "=== Ориентиры (ноутбук Core Ultra 9 285H, 16 ядер, на зарядке) ==="
echo "  cpp-cpu best.pt(384x640):        10.1s на 148s клипа (~15x реального времени)"
echo "  cpp-cpu yolo26s(576x1024):       54.6s на 148s клипа (~2.7x реального времени)"
echo "  Вердикт по аренде: делите ${DUR%.*}s на cpp-cpu-время выше — это множитель"
echo "  реального времени на этом CPU. Для потока стримов нужен запас >=2x на стрим."
} 2>&1 | tee "$OUT/report.txt"

echo
echo "Отчёт: $OUT/report.txt"
