#!/usr/bin/env bash
# Бенчмарк детектора (стадия A) на серверном CPU (EPYC/Xeon и т.п.), без GPU.
#
# Что нужно на сервере: клон репозитория + models/ (best.pt, best_1024x1024.pt
# и оба *_openvino_model) + тестовое видео. GT/эталоны не нужны (VLM тут не гоняется).
#
# Первый запуск на голом Ubuntu:
#   ./scripts/server_bench_cpu.sh --setup           # apt + venv + pip + сборка cpp
# Полный одиночный бенчмарк (уровни 1-4):
#   ./scripts/server_bench_cpu.sh [видео]           # дефолт test/video/test_fragment.mp4
# Свип параллелизма (агрегатная пропускная способность + атрибуция CPU/RAM):
#   ./scripts/server_bench_cpu.sh --sweep [видео]
#   NLIST="1 2 4 8" ./scripts/server_bench_cpu.sh --sweep [видео]   # свой список N
#   AUTO=1        ./scripts/server_bench_cpu.sh --sweep [видео]     # адаптивный подбор
#
# Настройки через окружение (в скобках — дефолты):
#   MODEL_OV (models/best_1024x1024_openvino_model)  FRAME_STEP (10)  CONF (0.5)
#
# Аналог GPU-набора (server_bench_nvidia.sh уровни 1-5 + server_bench_sweep.sh),
# но для CPU: на CPU нет NVDEC/VRAM, поэтому меряется sw-декод и загрузка ядер/RAM.
# Три уровня одиночного прогона + свип дают тот же охват, что на видеокарте:
#   1) benchmark_app  — потолок инференса OpenVINO (fps, без видео);
#   2) bench_infer.py — потолок через ultralytics (как в py-движке);
#   3) benchmark_detector.py — честный end-to-end (декод+инференс+события);
#   4) sw-декод — изолированная скорость ffmpeg-декода всего видео (аналог NVDEC ур.4);
#   --sweep — N параллельных детекций: агрегат ×rt + CPU%/RAM (аналог GPU-свипа).
set -euo pipefail
cd "$(dirname "$0")/.."
# Локаль-независимость: английские метки free ("Mem:") + точка в дробях (не "15,9"),
# иначе на ru/de-локали парсинг RAM и числовые сравнения ×rt ломаются.
export LC_ALL=C
# shellcheck disable=SC1091
source scripts/bench_hwinfo.sh   # hwinfo: паспорт CPU/RAM в каждый отчёт

MODEL_OV="${MODEL_OV:-models/best_1024x1024_openvino_model}"
FRAME_STEP="${FRAME_STEP:-10}"
CONF="${CONF:-0.5}"

# ---------------------------------------------------------------------------
# --setup: голая Ubuntu -> окружение готово к бенчу
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--setup" ]]; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq ffmpeg cmake g++ python3-venv python3-dev
    python3 -m venv donate_env
    # shellcheck disable=SC1091
    source donate_env/bin/activate
    pip install -q -r requirements.txt
    ./cpp/build.sh
    echo "=== setup готов; запускайте: $0 [видео]  и  $0 --sweep [видео] ==="
    exit 0
fi

# busy/total тики из /proc/stat (агрегат по всем ядрам) — для CPU% вокруг прогона
cpu_busy_total() {
    awk '/^cpu /{b=$2+$3+$4+$7+$8+$9; t=b+$5+$6; print b, t; exit}' /proc/stat
}
# a>b для дробных ×rt (bash умеет только целые)
awk_gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>b)}'; }

# ---------------------------------------------------------------------------
# --sweep N: сколько видео CPU тянет параллельно + во что упор (ядра или RAM).
# На CPU один cpp-процесс уже параллелит инференс по всем ядрам (OpenVINO
# THROUGHPUT, до 8 стримов), поэтому агрегат часто выходит на плато уже при N=1
# (CPU%→100). Свип это ПОКАЗЫВАЕТ и находит фактический knee: агрегат ×rt / (1+запас)
# = сколько стримов реального времени закрывает этот бокс стадией A.
# span меряется по стенке вокруг всех N процессов (включает старт/прогрев OV —
# консервативно); агрегат = N * длительность_видео / span.
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--sweep" ]]; then
    VIDEO="${2:-test/video/test_fragment.mp4}"
    [[ -f "$VIDEO" ]] || { echo "нет видео: $VIDEO"; exit 1; }
    [[ -x cpp/fast_detector ]] || { echo "cpp/fast_detector не собран — ./cpp/build.sh (или --setup)"; exit 1; }
    [[ -d "$MODEL_OV" ]] || { echo "нет OpenVINO-модели: $MODEL_OV (экспортируйте или задайте MODEL_OV=)"; exit 1; }
    # shellcheck disable=SC1091
    [[ -f donate_env/bin/activate ]] && source donate_env/bin/activate

    NCPU=$(nproc)
    RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
    # Список N: AUTO — климб с ранней остановкой; иначе заданный NLIST; иначе —
    # компактный дефолт (на CPU knee обычно низкий, лезть до сотен смысла нет).
    if [[ -n "${AUTO:-}" ]]; then
        NLIST="1 2 4 8 16 32"
    elif [[ -z "${NLIST:-}" ]]; then
        NLIST="1 2 4 8"
    fi

    OUT="benchmarks/sweep_cpu_$(hostname)_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUT"
    : > "$OUT/rows.tsv"
    DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")

    {
    echo "=== CPU-свип параллелизма + атрибуция узкого места (ядра / RAM) ==="
    hwinfo
    echo "Видео: $VIDEO (${DUR}s)   модель: $MODEL_OV   step: $FRAME_STEP   conf: $CONF"
    echo "Режим: ${AUTO:+AUTO-климб }N из: $NLIST"
    echo
    printf "%-4s %-11s %-6s %-9s %-7s %s\n" "N" "агрег_×rt" "CPU%" "RAM_МБ" "done" "вердикт"
    printf -- "-%.0s" $(seq 1 66); echo
    } | tee "$OUT/summary.txt"

    best_rt=0; declines=0; stop_reason=""
    for N in $NLIST; do
        # фоновый сэмплер пиковой RAM (раз в секунду)
        ( while :; do ram_used_mb; sleep 1; done ) > "$OUT/ram_n$N.log" 2>/dev/null &
        RMON=$!
        read -r CB0 CT0 < <(cpu_busy_total)
        S0=$(date +%s.%N)

        pids=()
        for i in $(seq 1 "$N"); do
            mkdir -p "$OUT/tmp_${N}_${i}"
            cpp/fast_detector --video "$VIDEO" --model "$MODEL_OV" --device CPU \
                --frame-step "$FRAME_STEP" --conf "$CONF" --no-save-images --quiet \
                --tmp-dir "$OUT/tmp_${N}_${i}" --out-json "$OUT/run_${N}_${i}.json" \
                > "$OUT/run_${N}_${i}.log" 2>&1 &
            pids+=("$!")
        done
        # ждём ТОЛЬКО детекторы (не бесконечный RAM-сэмплер $RMON)
        for pid in "${pids[@]}"; do wait "$pid" || true; done

        S1=$(date +%s.%N)
        read -r CB1 CT1 < <(cpu_busy_total)
        kill "$RMON" 2>/dev/null || true

        span=$(awk -v a="$S0" -v b="$S1" 'BEGIN{printf "%.3f", b-a}')
        done_n=$( { grep -l '"raw_detections"' "$OUT"/run_${N}_*.json 2>/dev/null || true; } | wc -l )
        RT=$(awk -v n="$done_n" -v dur="$DUR" -v s="$span" \
            'BEGIN{if(s>0 && n>0) printf "%.1f", n*dur/s; else print "?"}')
        CPU=$(awk -v b0="$CB0" -v t0="$CT0" -v b1="$CB1" -v t1="$CT1" \
            'BEGIN{d=t1-t0; if(d>0) printf "%.0f", (b1-b0)/d*100; else print "?"}')
        RAM=$(awk '{if($1+0>m)m=$1+0} END{printf "%d", m}' "$OUT/ram_n$N.log")
        rm -rf "$OUT"/tmp_${N}_*

        verdict="—"
        if [[ "$CPU" =~ ^[0-9]+$ ]]; then
            if   (( done_n < N ));  then verdict="часть процессов упала (см. run_${N}_*.log)"
            elif (( CPU >= 90 ));   then verdict="упор в CPU (ядра насыщены)"
            else verdict="есть запас (не CPU-bound: декод/оркестрация/IO)"; fi
        fi
        printf "%-4s %-11s %-6s %-9s %-7s %s\n" "$N" "${RT:-?}x" "$CPU" "$RAM" "$done_n/$N" "$verdict" \
            | tee -a "$OUT/summary.txt"
        echo -e "$N\t${RT:-0}\t$CPU\t$RAM\t$done_n" >> "$OUT/rows.tsv"

        # адаптивная ранняя остановка (только AUTO): нашли потолок — дальше не лезем
        if [[ -n "${AUTO:-}" && "$RT" =~ ^[0-9.]+$ ]]; then
            (( done_n < N )) && { stop_reason="часть процессов упала при N=$N — предел ресурсов"; break; }
            if awk_gt "$RT" "$best_rt"; then best_rt="$RT"; fi
            if awk_gt "$(awk -v b="$best_rt" 'BEGIN{print b*0.97}')" "$RT"; then
                declines=$((declines+1))
                (( declines >= 2 )) && { stop_reason="агрегат перестал расти (прошли knee) после N=$N"; break; }
            else declines=0; fi
        fi
    done

    {
    echo
    awk -F'\t' '
      {n=$1; rt=$2+0; cpu=$3+0; ram=$4+0;
       if(rt>best){best=rt; bn=n; bcpu=cpu; bram=ram}}
      END{
        if(best<=0){print "🏆 ОПТИМУМ: не удалось замерить (см. run_*.log)"; exit}
        printf "🏆 ОПТИМУМ: N=%s -> %.1fx реального времени (CPU %d%%, RAM %d МБ)\n", bn,best,bcpu,bram
        if(bcpu>=90) print "   предел = CPU (ядра насыщены): помогут только больше/быстрее ядер."
        else         print "   предел = плато (ядра не в 90%): узкое место в декоде/оркестрации/IO, не в инференсе."
        printf "   => стадия A закрывает ~%.0f стримов реального времени (агрегат/1x); с запасом >=2x — ~%.0f.\n", best, best/2
      }' "$OUT/rows.tsv"
    [[ -n "$stop_reason" ]] && echo "   AUTO-стоп: $stop_reason"
    echo
    echo "Как читать:"
    echo "  агрег_×rt — во сколько раз быстрее реального времени бокс жуёт СУММУ N видео."
    echo "  CPU%→100 (все ${NCPU} vCPU) уже при N=1 — один cpp-процесс насыщает ядра"
    echo "     (OpenVINO THROUGHPUT), больше процессов агрегат не поднимут — это норма для CPU."
    echo "  RAM_МБ — пик занятой системной памяти (всего RAM ${RAM_GB} ГБ)."
    } | tee -a "$OUT/summary.txt"

    echo
    echo "Отчёт: $OUT/summary.txt"
    exit 0
fi

# ---------------------------------------------------------------------------
# Полный одиночный бенчмарк (уровни 1-4)
# ---------------------------------------------------------------------------
VIDEO="${1:-test/video/test_fragment.mp4}"
[[ -f "$VIDEO" ]] || { echo "нет видео: $VIDEO"; exit 1; }
[[ -x cpp/fast_detector ]] || { echo "cpp/fast_detector не собран — ./cpp/build.sh (или --setup)"; exit 1; }
# shellcheck disable=SC1091
[[ -f donate_env/bin/activate ]] && source donate_env/bin/activate

OUT="benchmarks/server_cpu_$(hostname)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")

# End-to-end уровня 3 для одной модели; пропуск, если её OpenVINO-каталога нет
# (на сервере может быть экспортирована только часть моделей — не валим прогон).
# py-openvino/cpp-cpu берут OpenVINO-каталог; --model-pt этим движкам не нужен.
run_e2e() {   # $1=openvino-dir  $2=imgsz  $3=подпись  [$4=pt для справки]
    if [[ ! -d "$1" ]]; then
        echo "--- $3: пропуск (нет каталога $1) ---"
        return 0
    fi
    echo "--- $3 (ov=$1, imgsz=$2) ---"
    python3 scripts/benchmark_detector.py --video "$VIDEO" --conf "$CONF" \
        --engines py-openvino,cpp-cpu --no-save-images --repeats 2 \
        --img-size "$2" --model-openvino "$1" --model-pt "${4:-models/best.pt}" \
        || echo "  (ур.3 $3 упал — см. трейс выше, продолжаю)"
}

{
hwinfo
echo "Видео: $VIDEO (${DUR}s)   step: $FRAME_STEP   conf: $CONF"

echo
echo "=== 1. Потолок инференса OpenVINO (benchmark_app, THROUGHPUT, 15s) ==="
for dir in models/best_openvino_model models/best_1024x1024_openvino_model; do
    [[ -d "$dir" ]] || { echo "--- $dir --- (нет каталога, пропуск)"; continue; }
    xml=$( ls "$dir"/*.xml 2>/dev/null | head -1 || true )
    echo "--- $dir ---"
    benchmark_app -m "$xml" -d CPU -hint throughput -t 15 2>/dev/null \
        | grep -E "Throughput|Median|count" || echo "  benchmark_app недоступен (pip install openvino)"
done

echo
echo "=== 2. Потолок через ultralytics (bench_infer.py) ==="
if [[ -d models/best_1024x1024_openvino_model ]]; then
    python3 scripts/bench_infer.py --model models/best_1024x1024_openvino_model \
        --device intel:cpu --img-size 576,1024 --n 50 \
        || echo "  (ур.2 упал — продолжаю)"
else
    echo "  пропуск: нет models/best_1024x1024_openvino_model"
fi

echo
echo "=== 3. End-to-end: детекция видео (py-openvino + cpp-cpu, 2 повтора) ==="
run_e2e models/best_openvino_model            384,640  "best 384x640"      models/best.pt
run_e2e models/best_1024x1024_openvino_model  576,1024 "yolo26s 576x1024"  models/best_1024x1024.pt

echo
echo "=== 4. Изолированный sw-декод всего видео (аналог NVDEC ур.4 на GPU) ==="
D0=$(date +%s.%N)
ffmpeg -hide_banner -loglevel error -i "$VIDEO" -f null -
D1=$(date +%s.%N)
DEC=$(awk -v a="$D0" -v b="$D1" 'BEGIN{printf "%.1f", b-a}')
awk -v d="$DEC" -v dur="$DUR" 'BEGIN{printf "sw-декод всего видео (CPU, все кадры): %ss = %.1fx реального времени\n", d, dur/d}'
echo "  (пайплайн платит примерно столько же: select каждого N-го кадра — уже ПОСЛЕ декода)"

echo
echo "=== Ориентиры (ноутбук Core Ultra 9 285H, 16 ядер, на зарядке) ==="
echo "  cpp-cpu best.pt(384x640):        10.1s на 148s клипа (~15x реального времени)"
echo "  cpp-cpu yolo26s(576x1024):       54.6s на 148s клипа (~2.7x реального времени)"
echo "  Вердикт по аренде: делите ${DUR%.*}s на cpp-cpu-время (ур.3) — множитель реального"
echo "  времени этого CPU. Для потока стримов нужен запас >=2x на стрим (см. --sweep)."
} 2>&1 | tee "$OUT/report.txt"

echo
echo "Отчёт: $OUT/report.txt"
echo "Пропускная способность под нагрузкой: $0 --sweep $VIDEO"
