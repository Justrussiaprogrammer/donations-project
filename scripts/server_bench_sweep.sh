#!/usr/bin/env bash
# Свип параллелизма детектора + атрибуция узкого места (GPU vs CPU vs RAM).
#
# Гоняет server_bench_nvidia.sh --parallel N для каждого N из NLIST и ВОКРУГ
# каждого прогона снимает: среднюю утилизацию GPU (nvidia-smi), среднюю загрузку
# CPU (по /proc/stat, 100% = все vCPU заняты) и прирост занятой RAM. Печатает
# таблицу N | агрегат ×rt | GPU% | CPU% | ΔRAM — сразу видно, что насыщается
# раньше: сама карта (GPU%→100) или её процессор (CPU%→100 при GPU%<100).
#
# Отвечает на: (1) реальный потолок карты (а не гипотеза о насыщении при N=4);
# (2) влияет ли vCPU на пропускную способность GPU и насколько; (3) хватает ли RAM.
#
#   NLIST="1 2 4 8 16 32" ./scripts/server_bench_sweep.sh [видео]
#
# Требует уже поднятого gpu_env (server_bench_nvidia.sh --setup) и собранного
# TRT-engine (создастся при первом --parallel, если ещё нет).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[[ -f gpu_env/bin/activate ]] && source gpu_env/bin/activate

NLIST="${NLIST:-1 2 4 8 16}"
VIDEO="${1:-test/video/test_fragment.mp4}"
[[ -f "$VIDEO" ]] || { echo "нет видео: $VIDEO"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "нет nvidia-smi"; exit 1; }

NCPU=$(nproc)
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
OUT="benchmarks/sweep_$(hostname)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

# busy/total тики из /proc/stat (агрегат по всем ядрам)
cpu_busy_total() {
    awk '/^cpu /{b=$2+$3+$4+$7+$8+$9; t=b+$5+$6; print b, t; exit}' /proc/stat
}

{
echo "=== Свип параллелизма + атрибуция узкого места ==="
echo "Карта: $(nvidia-smi --query-gpu=name --format=csv,noheader) | ${NCPU} vCPU | RAM ${RAM_GB} ГБ"
echo "Видео: $VIDEO   N из: $NLIST"
echo
printf "%-4s %-12s %-7s %-7s %-9s %s\n" "N" "агрег_×rt" "GPU%" "CPU%" "ΔRAM_MB" "вердикт"
printf -- "-%.0s" $(seq 1 60); echo
} | tee "$OUT/summary.txt"

for N in $NLIST; do
    # фоновый сэмплер GPU-утилизации (раз в секунду до конца прогона)
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 \
        > "$OUT/gpu_n$N.log" 2>/dev/null &
    GMON=$!
    read -r CB0 CT0 < <(cpu_busy_total)
    RAM0=$(free -m | awk '/^Mem:/{print $3}')

    ./scripts/server_bench_nvidia.sh --parallel "$N" "$VIDEO" > "$OUT/run_n$N.log" 2>&1 || true

    read -r CB1 CT1 < <(cpu_busy_total)
    RAM1=$(free -m | awk '/^Mem:/{print $3}')
    kill "$GMON" 2>/dev/null || true

    RT=$(grep -h 'АГРЕГАТ' "$OUT/run_n$N.log" | grep -oE '[0-9.]+x реального' | grep -oE '[0-9.]+' | head -1)
    GPU=$(awk '{s+=$1; n++} END{if(n) printf "%.0f", s/n; else print "?"}' "$OUT/gpu_n$N.log")
    CPU=$(awk -v b0="$CB0" -v t0="$CT0" -v b1="$CB1" -v t1="$CT1" \
        'BEGIN{d=t1-t0; if(d>0) printf "%.0f", (b1-b0)/d*100; else print "?"}')
    DRAM=$((RAM1 - RAM0))

    # грубый вердикт по узкому месту
    verdict="—"
    [[ "$GPU" =~ ^[0-9]+$ && "$CPU" =~ ^[0-9]+$ ]] && {
        if   (( CPU >= 90 && GPU < 85 )); then verdict="упор в CPU"
        elif (( GPU >= 90 )); then verdict="упор в GPU (потолок карты)"
        else verdict="есть запас"; fi
    }
    printf "%-4s %-12s %-7s %-7s %-9s %s\n" "$N" "${RT:-?}x" "$GPU" "$CPU" "$DRAM" "$verdict" \
        | tee -a "$OUT/summary.txt"
done

{
echo
echo "Как читать:"
echo "  GPU%→100 при росте N — карта и есть потолок (помощнее карта = больше throughput)."
echo "  CPU%→100 (все ${NCPU} vCPU заняты) при GPU%<100 — упор в ПРОЦЕССОР:"
echo "     карта недогружена, помощнее GPU НЕ поможет без большего числа vCPU."
echo "  агрег_×rt перестал расти с N — фактический потолок этой связки карта+CPU."
echo "  ΔRAM — прирост занятой RAM; сравни с ${RAM_GB} ГБ, чтобы понять, не она ли лимит."
} | tee -a "$OUT/summary.txt"

echo
echo "Отчёт: $OUT/summary.txt"
