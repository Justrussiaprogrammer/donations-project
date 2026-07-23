#!/usr/bin/env bash
# Общий сборщик паспорта железа для всех бенчей. Подключается через `source`.
#
#   source "$(dirname "$0")/bench_hwinfo.sh"
#   hwinfo            # печатает блок "=== Железо / контекст ===" на stdout
#
# Зачем: стадия A (детектор) — CPU-bound, поэтому модель процессора и объём RAM
# так же важны для сравнения серверов, как и карта. Раньше GPU-скрипты писали в
# отчёт только name/driver/VRAM карты — модель CPU и память терялись. hwinfo
# фиксирует всё разом, локаль-независимо (поля берутся из /proc, а не из lscpu-
# вывода, который переводится под язык системы).
#
# Функции:
#   hwinfo          — паспорт железа (GPU + CPU + RAM + инстанс).
#   gpu_vram_total  — общий объём VRAM в МБ (или пусто, если нет карты).
#   ram_used_mb     — сейчас занято системной RAM, МБ.

hwinfo() {
    echo "=== Железо / контекст ==="
    echo "host: $(hostname)   дата: $(date -Is)"

    # --- GPU ---
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
            --format=csv,noheader 2>/dev/null \
            | awk -F', *' '{printf "GPU: %s | драйвер %s | VRAM %s (занято %s)\n",$1,$2,$3,$4}'
    else
        echo "GPU: нет (nvidia-smi отсутствует) — CPU-режим"
    fi

    # --- CPU (из /proc/cpuinfo — не зависит от локали lscpu) ---
    local model sockets phys_cores threads mhz flags
    model=$(awk -F: '/model name/{gsub(/^[ \t]+/,"",$2); print $2; exit}' /proc/cpuinfo)
    threads=$(nproc)
    phys_cores=$(awk -F: '/cpu cores/{gsub(/ /,"",$2); print $2; exit}' /proc/cpuinfo)
    sockets=$(awk -F: '/physical id/{print $2}' /proc/cpuinfo | sort -u | wc -l)
    mhz=$(awk -F: '/cpu MHz/{gsub(/ /,"",$2); printf "%.0f",$2; exit}' /proc/cpuinfo)
    # интересующие расширения (влияют на скорость инференса/декода)
    flags=$(grep -om1 'avx512[a-z_]*\|avx2\|amx_tile' /proc/cpuinfo | tr '\n' ' ' 2>/dev/null || true)
    echo "CPU: ${model:-?}"
    echo "     vCPU(threads): ${threads} | физ.ядер/сокет: ${phys_cores:-?} | сокетов: ${sockets:-?} | ~${mhz:-?} MHz | ${flags:-—}"

    # --- RAM ---
    free -m | awk '/^Mem:/{printf "RAM: всего %d МБ | занято сейчас %d МБ | свободно %d МБ\n",$2,$3,$4}'

    # --- намёк на облачный инстанс (best-effort, не критично) ---
    if [[ -r /sys/class/dmi/id/product_name ]]; then
        echo "инстанс: $(cat /sys/class/dmi/id/product_name 2>/dev/null)"
    fi
}

gpu_vram_total() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1
}

ram_used_mb() {
    free -m | awk '/^Mem:/{print $3; exit}'
}
