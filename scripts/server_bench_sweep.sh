#!/usr/bin/env bash
# Свип параллелизма детектора + атрибуция узкого места (GPU vs CPU vs RAM/VRAM).
#
# Гоняет server_bench_nvidia.sh --parallel N для каждого N и ВОКРУГ каждого прогона
# снимает: среднюю утилизацию GPU и ПИК занятой VRAM (nvidia-smi), среднюю загрузку
# CPU (по /proc/stat, 100% = все vCPU заняты) и ПИК занятой RAM. Печатает таблицу
# N | агрегат ×rt | GPU% | CPU% | VRAM_МБ | RAM_МБ | вердикт — сразу видно, что
# насыщается раньше: карта (GPU%→100 / VRAM→предел) или её процессор (CPU%→100).
#
# Отвечает на: (1) какой N выжимает МАКСИМУМ из карты и где её реальный потолок;
# (2) CPU- или GPU-bound связка и насколько; (3) хватает ли RAM/VRAM.
#
# Режимы:
#   ./scripts/server_bench_sweep.sh [видео]              # авто-NLIST под число vCPU
#   NLIST="1 2 4 8 12 16 24 32" ./scripts/... [видео]    # свой список N
#   AUTO=1 ./scripts/server_bench_sweep.sh [видео]       # адаптивный подбор:
#       лезет вверх (1,2,4,8,16,32,...) и сам останавливается, найдя потолок —
#       когда GPU насыщается, кончается VRAM или агрегат перестаёт расти.
#
# Требует поднятого gpu_env (server_bench_nvidia.sh --setup) и TRT-engine
# (создастся при первом --parallel, если ещё нет).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/bench_hwinfo.sh
# shellcheck disable=SC1091
[[ -f gpu_env/bin/activate ]] && source gpu_env/bin/activate

VIDEO="${1:-test/video/test_fragment.mp4}"
[[ -f "$VIDEO" ]] || { echo "нет видео: $VIDEO"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "нет nvidia-smi"; exit 1; }

NCPU=$(nproc)
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
VRAM_TOTAL=$(gpu_vram_total); VRAM_TOTAL=${VRAM_TOTAL:-0}

# Список N: AUTO — климб с ранней остановкой; иначе — заданный NLIST; иначе —
# авто-скобка вокруг числа vCPU (потолок детектора обычно у knee ≈ vCPU).
if [[ -n "${AUTO:-}" ]]; then
    NLIST="1 2 4 8 16 32 64 128 256"
elif [[ -z "${NLIST:-}" ]]; then
    q=$((NCPU/4)); h=$((NCPU/2))
    NLIST=$(printf "1\n2\n%s\n%s\n%s\n%s\n%s\n" "$q" "$h" "$NCPU" "$((NCPU*3/2))" "$((NCPU*2))" \
            | awk '$1>=1' | sort -n -u | tr '\n' ' ')
fi

OUT="benchmarks/sweep_$(hostname)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
: > "$OUT/rows.tsv"

# busy/total тики из /proc/stat (агрегат по всем ядрам)
cpu_busy_total() {
    awk '/^cpu /{b=$2+$3+$4+$7+$8+$9; t=b+$5+$6; print b, t; exit}' /proc/stat
}
# a>b для дробных ×rt (bash умеет только целые)
awk_gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>b)}'; }

{
echo "=== Свип параллелизма + атрибуция узкого места ==="
hwinfo
echo "Видео: $VIDEO   режим: ${AUTO:+AUTO-климб }N из: $NLIST"
echo
printf "%-4s %-11s %-6s %-6s %-9s %-9s %s\n" "N" "агрег_×rt" "GPU%" "CPU%" "VRAM_МБ" "RAM_МБ" "вердикт"
printf -- "-%.0s" $(seq 1 72); echo
} | tee "$OUT/summary.txt"

best_rt=0; declines=0; stop_reason=""
for N in $NLIST; do
    # фоновые сэмплеры: GPU util+VRAM и системная RAM (раз в секунду)
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 \
        > "$OUT/gpu_n$N.log" 2>/dev/null &
    GMON=$!
    ( while :; do ram_used_mb; sleep 1; done ) > "$OUT/ram_n$N.log" 2>/dev/null &
    RMON=$!
    read -r CB0 CT0 < <(cpu_busy_total)

    ./scripts/server_bench_nvidia.sh --parallel "$N" "$VIDEO" > "$OUT/run_n$N.log" 2>&1 || true

    read -r CB1 CT1 < <(cpu_busy_total)
    kill "$GMON" "$RMON" 2>/dev/null || true

    RT=$(grep -h 'АГРЕГАТ' "$OUT/run_n$N.log" | grep -oE '[0-9.]+x реального' | grep -oE '[0-9.]+' | head -1)
    GPU=$(awk -F', *' '{g+=$1; n++} END{if(n) printf "%.0f", g/n; else print "?"}' "$OUT/gpu_n$N.log")
    VRAM=$(awk -F', *' '{if($2+0>m)m=$2+0} END{printf "%d", m}' "$OUT/gpu_n$N.log")
    RAM=$(awk '{if($1+0>m)m=$1+0} END{printf "%d", m}' "$OUT/ram_n$N.log")
    CPU=$(awk -v b0="$CB0" -v t0="$CT0" -v b1="$CB1" -v t1="$CT1" \
        'BEGIN{d=t1-t0; if(d>0) printf "%.0f", (b1-b0)/d*100; else print "?"}')

    # грубый вердикт по узкому месту
    verdict="—"
    if [[ "$GPU" =~ ^[0-9]+$ && "$CPU" =~ ^[0-9]+$ ]]; then
        if   (( VRAM_TOTAL > 0 && VRAM * 100 / VRAM_TOTAL >= 92 )); then verdict="упор в VRAM"
        elif (( CPU >= 90 && GPU < 85 )); then verdict="упор в CPU"
        elif (( GPU >= 90 )); then verdict="упор в GPU (потолок карты)"
        else verdict="есть запас"; fi
    fi
    printf "%-4s %-11s %-6s %-6s %-9s %-9s %s\n" "$N" "${RT:-?}x" "$GPU" "$CPU" "$VRAM" "$RAM" "$verdict" \
        | tee -a "$OUT/summary.txt"
    echo -e "$N\t${RT:-0}\t$GPU\t$CPU\t$VRAM\t$RAM" >> "$OUT/rows.tsv"

    # адаптивная ранняя остановка (только AUTO): нашли потолок — дальше не лезем
    if [[ -n "${AUTO:-}" && "$RT" =~ ^[0-9.]+$ ]]; then
        if awk_gt "$RT" "$best_rt"; then best_rt="$RT"; fi
        if [[ "$GPU" =~ ^[0-9]+$ ]] && (( GPU >= 95 )); then
            stop_reason="GPU насыщен (${GPU}%) — это потолок карты"; break
        elif (( VRAM_TOTAL > 0 && VRAM * 100 / VRAM_TOTAL >= 92 )); then
            stop_reason="VRAM почти вся (${VRAM}/${VRAM_TOTAL} МБ) — предел памяти карты"; break
        elif awk_gt "$(awk -v b="$best_rt" 'BEGIN{print b*0.97}')" "$RT"; then
            declines=$((declines+1))
            # два падения подряд (не один шумный прогон) = прошли knee
            (( declines >= 2 )) && { stop_reason="агрегат перестал расти (прошли knee) после N=$N"; break; }
        else declines=0; fi
    fi
done

{
echo
# --- автоматический вывод оптимума ---
awk -F'\t' -v vt="$VRAM_TOTAL" '
  {n=$1; rt=$2+0; gpu=$3+0; cpu=$4+0; vram=$5+0;
   if(rt>best){best=rt; bn=n; bgpu=gpu; bcpu=cpu; bvram=vram}}
  END{
    if(best<=0){print "🏆 ОПТИМУМ: не удалось замерить (см. run_n*.log)"; exit}
    printf "🏆 ОПТИМУМ: N=%s -> %.1fx реального времени (GPU %d%%, CPU %d%%, VRAM %d МБ)\n", bn,best,bgpu,bcpu,bvram
    if(vt>0 && bvram*100/vt>=92)      print "   предел = VRAM карты: больше слотов не влезет, нужна карта с бОльшей памятью."
    else if(bgpu>=90)                 print "   предел = САМА КАРТА (GPU насыщен): помощнее GPU поднимет потолок."
    else if(bcpu>=90)                 print "   предел = CPU: карта недогружена, помощнее GPU НЕ поможет — нужен CPU/больше vCPU."
    else                              print "   предел = плато связки (ни GPU, ни CPU не в 90%): узкое место в декоде/оркестрации/PCIe."
  }' "$OUT/rows.tsv"
[[ -n "$stop_reason" ]] && echo "   AUTO-стоп: $stop_reason"
echo
echo "Как читать:"
echo "  GPU%→100 при росте N — карта и есть потолок (помощнее карта = больше throughput)."
echo "  CPU%→100 (все ${NCPU} vCPU заняты) при GPU%<100 — упор в ПРОЦЕССОР:"
echo "     карта недогружена, помощнее GPU НЕ поможет без большего числа vCPU."
echo "  агрег_×rt перестал расти с N — фактический потолок этой связки карта+CPU."
echo "  VRAM_МБ — пик занятой видеопамяти; RAM_МБ — пик системной (всего RAM ${RAM_GB} ГБ, VRAM ${VRAM_TOTAL} МБ)."
} | tee -a "$OUT/summary.txt"

echo
echo "Отчёт: $OUT/summary.txt"
