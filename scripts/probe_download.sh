#!/usr/bin/env bash
# Проба стадии «добыча видео» (acquisition) — НЕ GPU-тест. Проверяет, отдаёт ли
# Twitch/YouTube VOD на IP арендованного сервера и с какой скоростью, чтобы
# оценить риск rate-limit/403/geo с датацентр-IP (в т.ч. Selectel) ДО того, как
# строить прод на скачивании. GPU не нужен — запускать, пока компилится llama.cpp
# (server_bench_vlm.sh --setup): холостое время GPU-часа не тратится впустую.
#
# Первый запуск (ставит yt-dlp + ffmpeg):
#   ./scripts/probe_download.sh --setup
# Проба (качает ТОЛЬКО срез [0, SLICE_SEC], а не весь VOD):
#   ./scripts/probe_download.sh 'https://www.twitch.tv/videos/XXXXXXXXXX'
#   SLICE_SEC=180 QUALITY=720p ./scripts/probe_download.sh <url>
#
# Что меряет и печатает:
#   0) egress-IP сервера (какой именно IP тестируется — репутация IP-специфична);
#   1) метаданные (yt-dlp -J): отдаёт ли IP вообще — быстро ловит 403/geo/удалён;
#   2) скачивание среза: MB/s + http-коды; отдельно подсвечивает 403/429/rate-limit.
# Ненулевой выход => добыча с этого IP проблемна (см. подсвеченные строки лога).
set -uo pipefail
cd "$(dirname "$0")/.."

SLICE_SEC="${SLICE_SEC:-120}"        # сколько секунд VOD скачать для замера
QUALITY="${QUALITY:-best}"           # 'best' | '1080p' | '720p' | '480p' ...
OUTDIR="${OUTDIR:-benchmarks/dl_probe_$(hostname)_$(date +%Y%m%d_%H%M%S)}"

if [[ "${1:-}" == "--setup" ]]; then
    command -v ffmpeg >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg; }
    python3 -m pip install -q -U yt-dlp || pip install -q -U yt-dlp
    echo "=== setup готов; запускайте: $0 <url_vod> ==="
    exit 0
fi

URL="${1:-}"
[[ -n "$URL" ]] || { echo "использование: $0 <url_vod>   (или $0 --setup)"; exit 2; }
command -v yt-dlp >/dev/null || { echo "нет yt-dlp — сначала $0 --setup"; exit 2; }
command -v ffmpeg >/dev/null || { echo "нет ffmpeg (нужен для --download-sections) — $0 --setup"; exit 2; }
mkdir -p "$OUTDIR"
LOG="$OUTDIR/ytdlp.log"

# yt-dlp формат: 'best' как есть; иначе высота (720p -> height<=720)
if [[ "$QUALITY" == "best" ]]; then FMT="best"; else FMT="best[height<=${QUALITY%p}]"; fi

# подсветка тревожных сигналов в логе yt-dlp
flag_trouble() {
    grep -iE 'HTTP Error 403|HTTP Error 429|Too Many Requests|rate.?limit|geo.?block|geo.?restrict|blocked|forbidden' "$1" \
        && return 0 || return 1
}

{
echo "=== Проба добычи VOD ==="
echo "egress-IP сервера: $(curl -s --max-time 10 https://api.ipify.org || echo '(не определить)')"
echo "URL: $URL   срез: ${SLICE_SEC}s   качество: $QUALITY ($FMT)"
echo "yt-dlp: $(yt-dlp --version)   ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)"

echo
echo "--- 1. Метаданные (yt-dlp -J) — отдаёт ли IP ---"
if META=$(yt-dlp -J --no-warnings "$URL" 2>"$LOG.meta"); then
    python3 - "$META" <<'PY'
import json, sys
m = json.loads(sys.argv[1])
dur = m.get("duration")
print(f"  OK: title={m.get('title','?')!r}")
print(f"      duration={dur}s ({dur//3600 if dur else 0}h{(dur%3600)//60 if dur else 0}m)  "
      f"ext={m.get('ext','?')}  formats={len(m.get('formats') or [])}")
PY
    META_OK=1
else
    echo "  !! метаданные не получены — IP не обслужен или VOD недоступен:"
    sed 's/^/     /' "$LOG.meta" | tail -5
    META_OK=0
fi

echo
echo "--- 2. Скачивание среза [0, ${SLICE_SEC}s] — скорость и ошибки ---"
if [[ "$META_OK" == 1 ]]; then
    t0=$(date +%s.%N)
    yt-dlp -f "$FMT" \
        --download-sections "*0-${SLICE_SEC}" --force-keyframes-at-cuts \
        --no-warnings --newline \
        -o "$OUTDIR/slice.%(ext)s" "$URL" 2>&1 | tee "$LOG" | tail -3
    rc=${PIPESTATUS[0]}
    t1=$(date +%s.%N)
    elapsed=$(python3 -c "print(f'{$t1-$t0:.1f}')")
    SLICE_FILE=$(find "$OUTDIR" -maxdepth 1 -name 'slice.*' | head -1)
    if [[ "$rc" == 0 && -n "$SLICE_FILE" ]]; then
        bytes=$(stat -c%s "$SLICE_FILE")
        mb=$(python3 -c "print(f'{$bytes/1048576:.1f}')")
        mbps=$(python3 -c "print(f'{$bytes/1048576/$elapsed:.2f}')" 2>/dev/null || echo "?")
        echo "  OK: скачано ${mb} МБ за ${elapsed}s = ${mbps} MB/s (${SLICE_SEC}s среза)"
        echo "  => полный ${SLICE_SEC}s-эквивалент качается за реальное время в ~$(python3 -c "print(f'{$SLICE_SEC/$elapsed:.1f}')" 2>/dev/null || echo '?')x"
    else
        echo "  !! скачивание не удалось (rc=$rc)"
    fi
else
    echo "  пропущено (метаданные не прошли)"
    rc=1
fi

echo
echo "--- Вердикт ---"
if flag_trouble "$LOG" >/dev/null 2>&1 || flag_trouble "$LOG.meta" >/dev/null 2>&1; then
    echo "  ⚠️  ЕСТЬ тревожные сигналы (403/429/rate-limit/geo) — добыча с этого IP рискованна:"
    { flag_trouble "$LOG.meta"; flag_trouble "$LOG"; } 2>/dev/null | sort -u | sed 's/^/     /'
    VERDICT=1
elif [[ "${rc:-1}" == 0 ]]; then
    echo "  ✅ VOD отдаётся и качается с этого IP без явных ограничений."
    echo "     Замечание: единичная проба ≠ объём — при массовой выкачке IP всё равно"
    echo "     может словить rate-limit. Для прода закладывай ретраи/паузы или прокси."
    VERDICT=0
else
    echo "  ⚠️  скачивание не прошло без явного бан-сигнала — смотри $LOG"
    VERDICT=1
fi
} 2>&1 | tee "$OUTDIR/report.txt"

echo
echo "Отчёт: $OUTDIR/report.txt   (логи yt-dlp: $LOG*)"
exit "${VERDICT:-1}"
