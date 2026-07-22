#!/usr/bin/env bash
# Бенчмарк VLM-стадии (сервер B) на арендованной карте: сколько кропов донатов
# в час пережёвывает GPU при разной конкурентности llama-server (-np N).
#
# Стадия B пайплайна — Qwen3-VL через llama.cpp (OpenAI-совместимый сервер).
# Меритель уже есть: scripts/vlm_stage.py --concurrency N (кропы -> отчёты,
# пишет vlm_elapsed_sec). Здесь мы: (1) собираем llama.cpp с CUDA, (2) поднимаем
# llama-server с -np = макс. конкурентности, (3) гоняем набор кропов при N из
# CONCURRENCY и считаем агрегат кропов/сек и донатов/час на карту.
#
# Зеркалит server_bench_nvidia.sh (стадия A), но меряет VLM, а не детектор.
# Работает и на CPU-сервере: GPU_LAYERS=0 (llama.cpp соберётся всё равно с CUDA,
# но слои не оффлоадятся) — для честного сравнения роли B на CPU против GPU лучше
# собрать build-cpu вручную по llama_cpp_setup.md; этот --setup — CUDA-путь.
#
# Что нужно на сервере: клон репозитория + набор кропов (по умолчанию 178 штук
# test/gt/donations — залить, т.к. test/ в .gitignore). GGUF-модель llama-server
# тянет сам через -hf при первом старте (нужен интернет + сборка с libcurl).
# NVIDIA-драйвер + CUDA toolkit на хосте.
#
# Первый запуск:
#   ./scripts/server_bench_vlm.sh --setup       # apt + сборка llama.cpp(CUDA) + venv
# Свип конкурентности (главный прогон):
#   ./scripts/server_bench_vlm.sh               # дефолтные N: 1 2 4 8
#   ./scripts/server_bench_vlm.sh "4 6 8"       # СВОИ N: первый аргумент = список
#   CONCURRENCY="4 6 8" ./scripts/server_bench_vlm.sh   # то же через окружение
# Разные модели насыщаются при разной конкурентности — подбирай список под модель.
#
# Настройки через окружение (в скобках — дефолты):
#   MODEL_HF (Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M)  CROPS (test/gt/donations)
#   CONCURRENCY ("1 2 4 8" — или 1-й позиц. аргумент)  PORT (8081)
#   IMAGE_MIN_TOKENS (1024)  CTX_PER_SLOT (2048)  KV_TYPE (f16 | q8_0)
#   GPU_LAYERS (999 — все слои на GPU; 0 — целиком CPU)  LLAMA_DIR (~/llama.cpp)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_HF="${MODEL_HF:-Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M}"
CROPS="${CROPS:-test/gt/donations}"
CONCURRENCY="${CONCURRENCY:-1 2 4 8}"
# 1-й позиционный аргумент (если это не --setup) переопределяет список конкурентностей:
#   ./scripts/server_bench_vlm.sh "4 6 8"
[[ -n "${1:-}" && "$1" != "--setup" ]] && CONCURRENCY="$1"
[[ "$CONCURRENCY" =~ ^[0-9]+([[:space:]]+[0-9]+)*$ ]] \
    || { echo "список конкурентностей — числа через пробел, напр. \"4 6 8\" (получено: '$CONCURRENCY')"; exit 1; }
PORT="${PORT:-8081}"
GPU_LAYERS="${GPU_LAYERS:-999}"
# Прод-настройка: Qwen3-VL требует >=1024 image-токенов на кроп (иначе llama
# ругается и просаживает точность grounding). Это же делает prefill тяжелее —
# так и надо мерить, чтобы «донатов/час» совпадали с продом.
IMAGE_MIN_TOKENS="${IMAGE_MIN_TOKENS:-1024}"
# Контекст на слот: сервер делит -c на -np слотов, каждому нужно image(1024) +
# промпт + ответ. 2048/слот — реалистично и позволяет np=8 влезть в 24ГБ на f16.
CTX_PER_SLOT="${CTX_PER_SLOT:-2048}"
# Тип KV-кэша: f16 (полная точность) | q8_0 (вдвое меньше VRAM, near-lossless,
# требует flash-attention). q8_0 позволяет держать больше слотов на 24ГБ-карте.
KV_TYPE="${KV_TYPE:-f16}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
# запрошенный максимум конкурентности; реальный NP скорректируем по VRAM ниже
NP_REQ="${NP:-$(echo "$CONCURRENCY" | tr ' ' '\n' | sort -n | tail -1)}"
LLAMA_BIN="$LLAMA_DIR/build-cuda/bin/llama-server"

if [[ "${1:-}" == "--setup" ]]; then
    command -v nvidia-smi >/dev/null || echo "!! нет nvidia-smi — будет только CPU-сборка смысл (GPU_LAYERS=0)"
    sudo apt-get update -qq
    # libssl-dev ОБЯЗАТЕЛЕН: без него llama.cpp собирается без HTTPS и `-hf`
    # не может скачать GGUF («HTTPS is not supported… rebuild with -DLLAMA_OPENSSL=ON»).
    sudo apt-get install -y -qq git cmake build-essential libcurl4-openssl-dev \
        libssl-dev python3-venv python3-dev
    # llama.cpp с CUDA требует nvcc (CUDA toolkit), а не только драйвер. На «чистом»
    # образе (не-Docker) с одним GPU-драйвером nvcc обычно нет — доставляем пакет
    # Ubuntu. nvcc старее драйвера 580 — это НОРМА (драйвер обратно совместим).
    if command -v nvidia-smi >/dev/null && ! command -v nvcc >/dev/null; then
        echo "--- nvcc не найден: ставлю nvidia-cuda-toolkit (для сборки CUDA) ---"
        sudo apt-get install -y -qq nvidia-cuda-toolkit \
            || { echo "!! nvidia-cuda-toolkit не встал — поставьте CUDA toolkit вручную"; exit 1; }
    fi
    command -v nvcc >/dev/null && echo "nvcc: $(nvcc --version | tail -1)"
    if [[ ! -d "$LLAMA_DIR/.git" ]]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    fi
    echo "--- сборка llama.cpp build-cuda (-DGGML_CUDA=ON; минуты) ---"
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build-cuda" \
        -DGGML_CUDA=ON -DLLAMA_CURL=ON -DLLAMA_OPENSSL=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$LLAMA_DIR/build-cuda" -j"$(nproc)" --target llama-server
    # лёгкое venv для vlm_stage: donsearcher-хелперы (requests/Pillow), без torch
    python3 -m venv vlm_env
    # shellcheck disable=SC1091
    source vlm_env/bin/activate
    pip install -q -U pip
    # cv2 нужен: vlm_stage.py и модули donsearcher (vlm_client/vlm_events) импортят
    # OpenCV на верхнем уровне; headless — без GUI/X11-зависимостей, для сервера.
    pip install -q requests Pillow numpy opencv-python-headless
    echo "=== setup готов; запускайте: $0 ==="
    exit 0
fi

# shellcheck disable=SC1091
[[ -f vlm_env/bin/activate ]] && source vlm_env/bin/activate
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"   # import donsearcher без editable-install
[[ -x "$LLAMA_BIN" ]] || { echo "нет $LLAMA_BIN — сначала $0 --setup"; exit 1; }
[[ -d "$CROPS" ]] || { echo "нет папки кропов: $CROPS (test/ в .gitignore — залить вручную)"; exit 1; }
N_CROPS=$(find "$CROPS" -maxdepth 1 -iname '*.png' | wc -l)
[[ "$N_CROPS" -gt 0 ]] || { echo "в $CROPS нет .png-кропов"; exit 1; }

# --- оценка VRAM: СОВЕТ, не жёсткий кап. Пробуем запрошенный np целиком; если
# реально не влезет — llama-server упадёт с OOM, и мы подскажем ниже (см. health).
# Заранее ничего не режем: сколько влезает — решает сама карта, а не формула.
NP="$NP_REQ"
CTX=$(( CTX_PER_SLOT * NP ))
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
VRAM_MB=${VRAM_MB:-24000}
case "$KV_TYPE" in q8_0|q4_0|q4_1) KV_PT10=6 ;; *) KV_PT10=11 ;; esac   # 0.1 МБ/токен KV
VRAM_NEED_MB=$(( CTX * KV_PT10 / 10 + 7000 ))   # KV + модель/mmproj/буферы ~7ГБ
LLAMA_EXTRA=()
[[ "$KV_TYPE" != "f16" ]] && LLAMA_EXTRA+=(-fa on --cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE")

OUT="benchmarks/server_vlm_$(hostname)_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"
SERVER_URL="http://127.0.0.1:$PORT/v1/chat/completions"

echo "=== Железо ===" | tee "$OUT/report.txt"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$OUT/report.txt"
else
    echo "CPU-режим (нет nvidia-smi)" | tee -a "$OUT/report.txt"
fi
echo "Модель: $MODEL_HF  кропов: $N_CROPS ($CROPS)  ngl $GPU_LAYERS" \
    | tee -a "$OUT/report.txt"
echo "image-min-tokens: $IMAGE_MIN_TOKENS  KV: $KV_TYPE  ctx: $CTX ($CTX_PER_SLOT/слот × np$NP)  конкурентности: [$CONCURRENCY]" \
    | tee -a "$OUT/report.txt"
echo "оценка VRAM: нужно ~${VRAM_NEED_MB}МБ из ${VRAM_MB}МБ$( (( VRAM_NEED_MB > VRAM_MB )) && echo '  ⚠ впритык/OOM — при падении см. подсказку про KV_TYPE=q8_0' )" \
    | tee -a "$OUT/report.txt"

# --- поднять llama-server в фоне, дождаться готовности, гарантированно убить ---
echo "--- старт llama-server (-np $NP; первый запуск тянет GGUF, может быть долго) ---"
"$LLAMA_BIN" -hf "$MODEL_HF" -np "$NP" -ngl "$GPU_LAYERS" -c "$CTX" \
    --image-min-tokens "$IMAGE_MIN_TOKENS" "${LLAMA_EXTRA[@]}" \
    --host 127.0.0.1 --port "$PORT" >"$OUT/llama-server.log" 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT

# HEALTH_TIMEOUT сек ждём /health. Первый запуск включает скачивание GGUF (~5.5 ГБ:
# модель + mmproj), а на медленном датацентр-линке это заметно дольше 10 мин —
# поэтому дефолт 1800с и он настраивается. Совет: прогрей кэш заранее вручную
# (`llama-server -hf ... ` до «listening», Ctrl+C) — тогда бенч стартует за секунды.
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
for _ in $(seq 1 "$HEALTH_TIMEOUT"); do
    kill -0 "$SRV" 2>/dev/null || { echo "!! llama-server упал — смотри $OUT/llama-server.log"; exit 1; }
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 1
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "!! сервер не поднялся за ${HEALTH_TIMEOUT}с"
    if grep -qiE 'out of memory|failed to allocate|cudaMalloc|unable to allocate|OOM' "$OUT/llama-server.log"; then
        echo "   похоже на OOM: np=$NP × ${CTX_PER_SLOT}/слот (ctx $CTX) не влез в ${VRAM_MB}МБ."
        echo "   Решение: KV_TYPE=q8_0 (вдвое меньше KV, near-lossless) ИЛИ меньше CTX_PER_SLOT/np."
    else
        echo "   (медленный download GGUF? прогрей кэш вручную — см. шапку скрипта)"
    fi
    exit 1
fi
echo "сервер готов."

# --- свип конкурентности: тот же набор кропов при каждом N ---
{
echo
printf "%-6s %-10s %-14s %-16s\n" "N" "wall_s" "кропов/сек" "донатов/час"
printf -- "-%.0s" $(seq 1 52); echo
for C in $CONCURRENCY; do
    RUN="_bench_vlm_c${C}_$$"
    python3 scripts/vlm_stage.py --crops "$CROPS" --concurrency "$C" \
        --vlm-server-url "$SERVER_URL" --vlm-model Qwen3-VL \
        --output-dir "$OUT/runs" --run-name "$RUN" --overwrite \
        >"$OUT/vlm_c${C}.log" 2>&1 || { echo "N=$C: упало — $OUT/vlm_c${C}.log"; continue; }
    ELAPSED=$(python3 - "$OUT/runs/$RUN/run_metadata.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["vlm_elapsed_sec"])
PY
)
    CPS=$(python3 -c "print(f'{$N_CROPS/$ELAPSED:.2f}')")
    DPH=$(python3 -c "print(f'{$N_CROPS/$ELAPSED*3600:.0f}')")
    printf "%-6s %-10s %-14s %-16s\n" "$C" "$ELAPSED" "$CPS" "$DPH"
done
echo
echo "Интерпретация: 'донатов/час' на пике конкурентности — потолок роли B этой карты."
echo "Делите ожидаемый поток донатов/час на это число -> сколько таких карт нужно на VLM."
echo "Если рост кропов/сек с ростом N застопорился — карта в компьют-потолке (дальше -np не помогает)."
} 2>&1 | tee -a "$OUT/report.txt"

echo
echo "Отчёт: $OUT/report.txt  (логи сервера/прогонов там же)"
