#!/usr/bin/env bash
# Бенчмарк детектора на NVIDIA-GPU сервере (T4 / L4 / A100 / RTX 6000 Pro / H100 / H200).
#
# ВАЖНО: cpp-движок (OpenVINO) на NVIDIA не работает — здесь меряются пути,
# реальные для NVIDIA: torch CUDA, TensorRT batch=1 и pipe-конвейер
# ffmpeg(NVDEC) -> батчевый TensorRT (scripts/bench_pipe.py — та же архитектура,
# что уровень 5 bench-gpu.ipynb и cpp/fast_detector).
#
# Что нужно: клон репозитория + models/best_1024x1024.pt + тестовое видео.
# NVIDIA-драйвер на хосте.
#
# Первый запуск:
#   ./scripts/server_bench_nvidia.sh --setup      # venv + pip (torch CUDA, TensorRT)
# Полный одиночный бенчмарк (уровни 1-5):
#   ./scripts/server_bench_nvidia.sh [путь_к_видео]
# N параллельных видео на ОДНОЙ карте (агрегатная пропускная способность):
#   ./scripts/server_bench_nvidia.sh --parallel 4 [путь_к_видео]
#   for n in 1 2 4 8 16; do ./scripts/server_bench_nvidia.sh --parallel $n; done
#
# Настройки через окружение (в скобках — дефолты):
#   MODEL_PT (models/best_1024x1024.pt)  IMGSZ (576,1024)  FRAME_STEP (10)  BATCH (8)
#
# Скрипт параллелебезопасен: экспорт engine — под flock, имена прогонов
# уникальны ($$), несколько копий на одной карте не мешают друг другу.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/bench_hwinfo.sh   # hwinfo: CPU-модель/RAM/VRAM в каждый отчёт

MODEL_PT="${MODEL_PT:-models/best_1024x1024.pt}"
IMGSZ="${IMGSZ:-576,1024}"
FRAME_STEP="${FRAME_STEP:-10}"
BATCH="${BATCH:-8}"
ENGINE="${MODEL_PT%.pt}.engine"
ENGINE_B="${MODEL_PT%.pt}_b${BATCH}.engine"

if [[ "${1:-}" == "--setup" ]]; then
    command -v nvidia-smi >/dev/null || { echo "нет nvidia-smi — поставьте драйвер"; exit 1; }
    sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg python3-venv python3-dev
    python3 -m venv gpu_env
    # shellcheck disable=SC1091
    source gpu_env/bin/activate
    pip install -q -U pip
    pip install -q -e ".[dev]" ultralytics requests   # torch с CUDA приедет зависимостью
    # для экспорта TensorRT (иначе ultralytics доставляет на лету при первом экспорте)
    pip install -q "tensorrt-cu12>=10.0,!=10.2.0" onnxruntime-gpu onnxslim
    echo "=== setup готов; запускайте: $0 [видео] или $0 --parallel N [видео] ==="
    exit 0
fi

# shellcheck disable=SC1091
[[ -f gpu_env/bin/activate ]] && source gpu_env/bin/activate
[[ -f "$MODEL_PT" ]] || { echo "нет модели: $MODEL_PT"; exit 1; }

# Экспорт engine под flock: параллельные копии скрипта не устроят гонку сборки.
# Батчевый экспорт идёт из КОПИИ .pt с суффиксом _bN, т.к. ultralytics всегда
# пишет <имя_pt>.engine — иначе затёрся бы batch-1 engine.
# Первый экспорт долгий (сборка под конкретный GPU, минуты); engine привязан
# к модели GPU — на другой карте пересоберётся заново.
export_engine() {   # $1=.pt  $2=.engine  $3=batch
    [[ -f "$2" ]] && return 0
    (
        flock 9
        [[ -f "$2" ]] && exit 0
        echo "--- экспорт $2 (batch=$3, imgsz=$IMGSZ; минуты, один раз на карту) ---"
        local src="$1"
        if [[ "$3" != 1 ]]; then
            src="${1%.pt}_b$3.pt"
            [[ -f "$src" ]] || cp "$1" "$src"
        fi
        yolo export model="$src" format=engine half=True imgsz="$IMGSZ" batch="$3" device=0
    ) 9>"$2.lock"
}

# ---------------------------------------------------------------------------
# Режим --parallel N: N одновременных копий pipe-конвейера на одной карте.
# «Разных» видео не нужно — тот же файл, открытый N раз, эквивалентен по
# стоимости декода и инференса. Агрегат считается по перекрытию таймеров
# процессов (t0/t1 в JSON), стартовый прогрев каждого процесса не входит.
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--parallel" ]]; then
    N="${2:?использование: $0 --parallel N [видео]}"
    VIDEO="${3:-test/video/test_fragment.mp4}"
    [[ -f "$VIDEO" ]] || { echo "нет видео: $VIDEO"; exit 1; }
    export_engine "$MODEL_PT" "$ENGINE_B" "$BATCH"
    OUT="benchmarks/server_gpu_$(hostname)_par${N}_$(date +%Y%m%d_%H%M%S)_$$"
    mkdir -p "$OUT"
    DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")
    { hwinfo
      echo
      echo "=== $N параллельных видео: $ENGINE_B (batch=$BATCH), $VIDEO (${DUR}s) ==="
    } | tee "$OUT/report.txt"
    for i in $(seq 1 "$N"); do
        python3 scripts/bench_pipe.py --model "$ENGINE_B" --video "$VIDEO" \
            --img-size "$IMGSZ" --frame-step "$FRAME_STEP" --batch "$BATCH" \
            --json "$OUT/pipe_$i.json" >"$OUT/pipe_$i.log" 2>&1 &
    done
    wait
    python3 - "$OUT" "$N" "$DUR" <<'PY' | tee -a "$OUT/report.txt"
import json, sys
from pathlib import Path
out, n, dur = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
rr = [json.loads(p.read_text()) for p in sorted(out.glob("pipe_*.json"))]
if len(rr) < n:
    print(f"!! завершилось только {len(rr)}/{n} процессов — смотри {out}/pipe_*.log")
for i, r in enumerate(rr, 1):
    print(f"  #{i}: {r['wall_s']:6.1f}s  {r['sample_fps']:6.1f} сэмпл-fps  "
          f"{r['rt_factor']:5.1f}x rt  декод={r['decode']}")
if rr:
    span = max(r["t1_epoch"] for r in rr) - min(r["t0_epoch"] for r in rr)
    agg_fps = sum(r["samples"] for r in rr) / span
    agg_rt = len(rr) * dur / span
    print("-" * 60)
    print(f"АГРЕГАТ: {len(rr)} видео x {dur:.0f}s за {span:.1f}s = "
          f"{agg_fps:.1f} сэмпл-fps = {agg_rt:.1f}x реального времени")
    print(f"(= карта пережёвывает ~{agg_rt:.0f} часов стрима в час)")
PY
    echo "Логи и JSON: $OUT/"
    exit 0
fi

# ---------------------------------------------------------------------------
# Полный одиночный бенчмарк (уровни 1-5)
# ---------------------------------------------------------------------------
VIDEO="${1:-test/video/test_fragment.mp4}"
[[ -f "$VIDEO" ]] || { echo "нет видео: $VIDEO"; exit 1; }

OUT="benchmarks/server_gpu_$(hostname)_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")
RUN="_bench_trt_$$"

{
hwinfo
echo "Видео: $VIDEO (${DUR}s)  модель: $MODEL_PT  imgsz: $IMGSZ  step: $FRAME_STEP"

echo
echo "=== 1. Чистый инференс .pt FP16 (torch CUDA, batch=1) ==="
python3 scripts/bench_infer.py --model "$MODEL_PT" --device 0 \
    --img-size "$IMGSZ" --half --n 200

echo
echo "=== 2. TensorRT batch=1: экспорт FP16 + чистый инференс ==="
export_engine "$MODEL_PT" "$ENGINE" 1
python3 scripts/bench_infer.py --model "$ENGINE" --device 0 \
    --img-size "$IMGSZ" --n 300

echo
echo "=== 3. End-to-end: python -m donsearcher на .engine (прод-путь, cv2-декод) ==="
python3 -m donsearcher --model "$ENGINE" --video "$VIDEO" --device 0 \
    --img-size "$IMGSZ" --frame-step "$FRAME_STEP" --conf 0.5 \
    --skip-vlm --sequential --images-schema 0 --overwrite --run-name "$RUN"
python3 - <<PY
import json
m = json.load(open("vlm_runs/$RUN/run_metadata.json"))
y = m["yolo_elapsed_sec"]
print(f"YOLO-стадия end-to-end: {y:.1f}s на {$DUR:.0f}s видео = {$DUR/y:.1f}x реального времени")
print(f"событий: {m['events']}, детекций: {m['raw_detections']}")
PY
rm -rf "vlm_runs/$RUN"

echo
echo "=== 4. NVDEC: аппаратный декод (если ffmpeg с cuda) ==="
if ffmpeg -hide_banner -hwaccels 2>/dev/null | grep -q cuda; then
    /usr/bin/time -f "hw-декод всего видео: %es" \
        ffmpeg -hwaccel cuda -i "$VIDEO" -f null - -loglevel error
else
    echo "  ffmpeg без cuda hwaccel — NVDEC не замерить (нужна сборка с --enable-cuda)"
    /usr/bin/time -f "sw-декод всего видео (CPU): %es" \
        ffmpeg -i "$VIDEO" -f null - -loglevel error
fi

echo
echo "=== 5. Pipe-конвейер: ffmpeg(NVDEC) -> батчевый TRT (сильнейший одиночный путь) ==="
export_engine "$MODEL_PT" "$ENGINE_B" "$BATCH"
python3 scripts/bench_pipe.py --model "$ENGINE_B" --video "$VIDEO" \
    --img-size "$IMGSZ" --frame-step "$FRAME_STEP" --batch "$BATCH" \
    --json "$OUT/pipe_single.json"

echo
echo "=== Ориентиры (ноутбук, Intel iGPU через cpp-OpenVINO) ==="
echo "  yolo26s(576x1024) cpp-gpu: 10.6s на 148s клипа (~14x реального времени, 84 сэмпл-fps)"
echo "  Интерпретация: п.1/п.2 — потолок карты на batch=1; п.5 — сильнейший"
echo "  одиночный конвейер (сравнивать с ноутом и с уровнем 5 bench-gpu.ipynb)."
echo "  Если п.3 сильно ниже п.5 — прод-путь упёрт в cv2-декод, а не в GPU."
echo "  Пропускную способность карты меряет: $0 --parallel N (сколько видео параллельно)."
} 2>&1 | tee "$OUT/report.txt"

echo
echo "Отчёт: $OUT/report.txt"
