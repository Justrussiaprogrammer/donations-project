# Сбор информации о донатах из стримов

Проект позволяет получать данные о донатах на стримах.

Конвейер: видео -> YOLO-детектор плашки доната -> группировка детекций в события ->
выбор лучшего кропа -> локальный VLM (например, Qwen3-VL) -> JSON/CSV-отчёты.

Для работы нужны две вещи помимо самого проекта:

1. **YOLO-модель** детектора плашки доната (файл кладётся в `models/`).
2. **Локальный VLM-сервер** с OpenAI-совместимым API. Сервер можно поднимать любой;
   проверенный вариант (llama.cpp + Qwen3-VL-8B, в том числе сборка под GPU Intel)
   описан в [llama_cpp_setup.md](llama_cpp_setup.md).

## Какую VLM-модель выбрать

Рекомендую брать Qwen3-VL-8B (Q4_K_M, `--image-min-tokens 1024`).
Эта модель дает достаточное качество распознавания, не требуя для себя слишком много ресурсов.

## Настройка среды

Нужен только Python3 и зависимости из `requirements.txt` - больше для подготовки ничего ставить не требуется.

### Windows 11

Установите Python3, точно работает для Python3.12

Запустите терминал из папки проекта и создайте окружение:

```bash
python -m venv donate_env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\donate_env\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### Ubuntu 24.04

Системные пакеты для окружения:

```bash
sudo apt update
sudo apt install python3 python3-venv libgl1 -y
```

Для cpp-движка `fast_pipeline.py` (он по умолчанию) нужны дополнительные системные
зависимости — компилятор C++, CMake и ffmpeg. Движок кроссплатформенный (Linux /
macOS / Windows), OpenVINO берётся из pip-пакета в `donate_env`:

```bash
# Ubuntu
sudo apt update && sudo apt install -y g++ cmake ffmpeg
# macOS (Homebrew); компилятор — из Xcode Command Line Tools
brew install cmake ffmpeg
# Windows: Visual Studio Build Tools (C++), CMake и ffmpeg (добавить в PATH)
```

Если cpp-движок не нужен, запускайте с `--engine py` — тогда дополнительных
системных зависимостей не требуется.

Запустите терминал из папки проекта и создайте окружение:

```bash
python3 -m venv donate_env
source donate_env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Запуск

Единая точка входа - `scripts/fast_pipeline.py`. Перед запуском:

1. Положите YOLO-модель в `models/` (по умолчанию ищется `models/best.pt`).
2. Поднимите VLM-сервер (см. [llama_cpp_setup.md](llama_cpp_setup.md));
   по умолчанию пайплайн обращается к `http://127.0.0.1:8081/v1/chat/completions`.

Запуск из корня проекта:

```bash
source donate_env/bin/activate

python3 scripts/fast_pipeline.py --video stream.mp4 \
  --frame-step 10 --conf 0.5 \
  --vlm-server-url http://127.0.0.1:8081/v1/chat/completions \
  --vlm-model Qwen3-VL --overwrite
```

В `--video` укажите любой путь к вашему видео — абсолютный или относительный
к корню проекта (`путь/к/видео.mp4` выше — это плейсхолдер, подставьте свой файл).

Результаты прогона появятся в `vlm_runs/<имя_прогона>/`: `events_summary.csv`,
`totals_by_currency.csv`, `donations.jsonl`, `run_metadata.json` и папка `events/`
с лучшими кропами и кадрами.

### Движки детекции

`fast_pipeline.py` поддерживает два движка YOLO-стадии (флаг `--engine`):

- **`cpp`** (по умолчанию) - быстрый нативный детектор (C++/OpenVINO),
  кроссплатформенный (Linux / macOS / Windows). Требует дополнительно:
  - системные компилятор C++, `cmake` и `ffmpeg` (см. раздел установки выше);
  - экспорт модели в OpenVINO в `models/best_openvino_model/` (прямоугольный
    вход 384×640 под кадр 16:9 — на ~30% быстрее квадратного 640×640 при том же
    выходе, т.к. не тратит компьют на чёрные поля леттербокса):
    `python3 -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='openvino', half=True, dynamic=False, imgsz=[384,640])"`;
  - однократную сборку бинарника:
    - Linux/macOS: `./cpp/build.sh`
    - Windows: `cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release` затем
      `cmake --build cpp/build --config Release` (запускать из активированного
      `donate_env`, чтобы OpenVINO-DLL были в PATH);
  - для запуска на GPU добавьте `--cpp-device GPU` (если GPU недоступен —
    автоматический откат на CPU с предупреждением в stderr).
- **`py`** - эталонный Python-движок, без дополнительных системных зависимостей.
  Работает с `models/best.pt` (torch) напрямую: добавьте `--engine py`. Может
  работать и с OpenVINO-экспортом (`--model models/best_openvino_model`) — тогда
  устройство указывается явно: `--device intel:cpu` или `--device intel:gpu`.

По умолчанию VLM работает параллельно с детекцией; флаг `--sequential` запускает
сначала всю детекцию, затем весь VLM (полезно для чистого замера времени стадий).

### Сбор данных для дообучения детектора

Отдельный режим, чтобы улучшать сам YOLO-детектор (только движок `py` — добавьте
`--engine py`). При любой выбранной стратегии VLM-стадия автоматически пропускается
(`--skip-vlm` включается принудительно, об этом печатается сообщение) — сервер VLM
для сбора данных не нужен. В отличие от сохранения «лучших»
кропов (те — для VLM/OCR), здесь пишутся **полные кадры + YOLO-разметка +
аннотированные превью** прямо в каталог прогона (`images/`, `labels/`,
`previews/`, `manifest.csv`), готовые к дообучению в ultralytics. Боксы детектора — это
псевдо-разметка, которую вы потом поправляете вручную (превью — для быстрой проверки).

Стратегии выбора кадров (`--train-select`, через запятую):

- `best` — лучший кадр каждого события (макс. композитный score) — тот же кадр, что ушёл бы в VLM; один на событие.
- `uncertain` — «слабые срабатывания»: детекция с `conf` в `[--train-uncertain-min, --conf)`. Детектор обычно не показывает боксы ниже `--conf`, поэтому при включённой `uncertain` детекция запускается на пониженном пороге `--train-uncertain-min`: боксы `>= --conf` остаются донатами (события не меняются), а боксы из `[min, conf)` идут только в обучающую выборку — как кандидаты в негативы для ручной проверки. В разметку такого кадра попадают только уверенные боксы; слабые видно на превью (с их confidence) — человек решает, поднять их в позитивы или оставить фоном.
- `worst` — самые слабые из принятых детекций (`conf >= --conf`) внутри каждого события (`--train-worst-per-event` штук на событие).
- `negatives` — кадры, где модель вообще не сработала (ни одного бокса) — чистые негативы, снижают ложные срабатывания; иначе просто выбрасываются.
- `random` — **один** случайный кадр на событие доната (не лучший и не худший) — разнообразные позитивы, не привязанные к фазе анимации плашки.

`uncertain` и `negatives` ограничены `--train-budget` кадрами на стратегию
(стратифицированный по времени reservoir-сэмплинг: кадры разнесены по всему
стриму, диск не переполняется). `best` / `worst` / `random` считаются на
событие и бюджетом не ограничены.

```bash
python3 scripts/fast_pipeline.py --engine py --model models/best_openvino_model \
  --video путь/к/видео.mp4 --device intel:cpu --skip-vlm --conf 0.5 \
  --train-select uncertain,negatives,random --train-uncertain-min 0.25 \
  --train-budget 300 --overwrite
```

| Флаг | По умолчанию | Что делает |
| --- | --- | --- |
| `--train-select` | (выкл.) | Список стратегий через запятую: `best,worst,random,uncertain,negatives`. Пусто — сбор выключен. |
| `--train-dir` | каталог прогона | Куда складывать датасет (по умолчанию — прямо в папку прогона, без обёртки). |
| `--train-budget` | `200` | Лимит кадров на стратегию (стратифицированный reservoir) для `uncertain`/`negatives`. На `best`/`worst`/`random` не действует. |
| `--train-worst-per-event` | `1` | Сколько худших кадров сохранять на событие (`worst`). |
| `--train-uncertain-min` | `0.25` | Нижняя граница confidence для `uncertain` (верхняя = `--conf`). При включённой `uncertain` детекция идёт на этом пороге. |

## Флаги

`fast_pipeline.py` принимает **все** флаги py-движка (`python -m donsearcher`) плюс три собственных
(`--engine`, `--cpp-binary`, `--cpp-device`). Поэтому таблицы ниже относятся к обоим; раздел «Только fast_pipeline.py» - к нему одному.

### Пути и прогон

| Флаг | По умолчанию | Что делает |
| --- | --- | --- |
| `--video` | `test/video/test_fragment.mp4` | Входное видео для анализа. |
| `--model` | `models/best.pt` | Путь к YOLO-модели. Для py-движка - `.pt`. Для cpp-движка `fast_pipeline.py` сам подставляет `models/<имя>_openvino_model`. |
| `--project-dir` | `.` | Корень проекта (относительно него считаются пути). |
| `--output-dir` | `vlm_runs` | Папка, куда складываются прогоны. |
| `--run-name` | авто (по имени видео) | Имя папки конкретного прогона. |
| `--overwrite` | выкл. | Перезаписать папку прогона, если она уже существует (иначе запуск падает). |

### Детекция (YOLO-стадия)

| Флаг | По умолчанию | Что делает |
| --- | --- | --- |
| `--device` | `cpu` | Устройство **py-движка**, передаётся в ultralytics как есть. Для `.pt`-модели — `cpu` или CUDA-индекс (`0`, `cuda:0`); для OpenVINO-модели — `intel:cpu` / `intel:gpu` / `intel:npu`. На cpp-движок не влияет (см. `--cpp-device`). |
| `--img-size` | `640` | Размер входного изображения YOLO. |
| `--conf` | `0.5` | Порог уверенности детектора (ниже - отбрасывается). |
| `--frame-step` | `10` | Обрабатывать каждый N-й кадр (10 = каждый десятый). |
| `--padding-x` | `20` | Горизонтальный отступ (px) при вырезании кропа плашки. |
| `--padding-y` | `12` | Вертикальный отступ (px) при вырезании кропа плашки. |
| `--max-processed-frames` | `0` | Лимит числа обрабатываемых кадров (`0` - без лимита, удобно для быстрой проверки). |

### Группировка детекций в события

| Флаг | По умолчанию | Что делает |
| --- | --- | --- |
| `--event-gap-sec` | `3` | Событие закрывается, если новых детекций нет дольше стольких секунд. |
| `--event-iou-thr` | `0.25` | Порог IoU для слияния детекций в одно событие. |
| `--event-center-thr` | `0.05` | Порог расстояния между центрами (доля кадра) для слияния детекций соседних кадров. |
| `--keep-top-candidates` | `1` | Сколько лучших кропов-кандидатов хранить на событие (в VLM уходит один лучший). |

### VLM (распознавание текста плашки)

| Флаг | По умолчанию | Что делает |
| --- | --- | --- |
| `--vlm-server-url` | `http://127.0.0.1:8081/v1/chat/completions` | Адрес OpenAI-совместимого VLM-сервера. |
| `--vlm-model` | `Qwen3-VL` | Имя модели, передаётся в запросе к серверу. |
| `--vlm-prompt` | `v7` | Промпт VLM: имя версии из `prompts/` (`v7`, `v10`, …) или путь к своему `.txt`-файлу. Выбранная версия и текст пишутся в `run_metadata.json`. |
| `--vlm-timeout` | `300` | Таймаут одного запроса к VLM, сек. |
| `--vlm-retries` | `2` | Число повторов при сетевых/серверных ошибках. |
| `--vlm-max-tokens` | `1024` | Лимит токенов в ответе модели. |
| `--vlm-temperature` | `0.0` | Температура генерации (0 - детерминированно). |
| `--skip-vlm` | выкл. | Только YOLO и группировка событий, без вызовов VLM (быстрая проверка детектора). |
| `--sequential` | выкл. | Сначала вся детекция, потом весь VLM (без перекрытия) - чистый тайминг стадий. |
| `--images-schema` | `7` | Битовая маска сохраняемых изображений (сумма степеней двойки): `1`=best_crops, `2`=annotated_frames, `4`=original_frames. `7` - всё, `0` - ничего (только CSV/JSONL), `1` - только кропы (YOLO-стадия для отдельного VLM-сервера), `5` - кропы + оригинальные кадры. |
| `--events-meta` | (выкл.) | Писать `events_meta.jsonl`: `minimal` - id/кроп/время доната + дата/платформа/стример (экономный вариант для передачи кропов на отдельный VLM-сервер); `full` - все детекторные поля событий (полный отчёт для проверки/заказчика). |
| `--streamer` / `--platform` / `--stream-date` | (пусто) | Провенанс стрима для `events_meta.jsonl` - этих данных нет в самом видео. |

### Только fast_pipeline.py

| Флаг | По умолчанию | Что делает |
| --- | --- | --- |
| `--engine` | `cpp` | Движок детекции: `cpp` (быстрый нативный) или `py` (эталонный Python). |
| `--cpp-binary` | `cpp/fast_detector` | Путь к собранному нативному детектору. |
| `--cpp-device` | `CPU` | OpenVINO-устройство для cpp-движка: `CPU` или `GPU` (Intel iGPU). |

## Примеры запуска

Все команды запускаются из корня проекта при активированном окружении
(`source donate_env/bin/activate`) и поднятом VLM-сервере.

Рекомендуемый прогон - быстрый cpp-движок на CPU, VLM параллельно с детекцией:

```bash
python3 scripts/fast_pipeline.py --video путь/к/видео.mp4
```

То же с помощью GPU (детекция на GPU через OpenVINO):

```bash
python3 scripts/fast_pipeline.py --video путь/к/видео.mp4 --cpp-device GPU
```

Последовательный режим (вся детекция, затем весь VLM) - для чистого замера времени
каждой стадии:

```bash
python3 scripts/fast_pipeline.py --video путь/к/видео.mp4 --sequential --overwrite
```

Проверка только детектора без VLM (сервер поднимать не нужно):

```bash
python3 scripts/fast_pipeline.py --video путь/к/видео.mp4 --skip-vlm
```

Раздельный запуск стадий на двух серверах. Сервер A — YOLO-стадия: только
кропы + `events_meta.jsonl`, без лишних annotated/original-кадров:

```bash
python3 scripts/fast_pipeline.py --video путь/к/видео.mp4 \
  --skip-vlm --images-schema 1 --events-meta minimal \
  --streamer Ник --platform twitch --stream-date 2026-07-06 --overwrite
```

Кропы и `events_meta.jsonl` переносятся на сервер B (rsync, сетевой диск),
где `scripts/vlm_stage.py` гонит их через VLM и собирает полные отчёты —
те же `events_summary.csv` / `totals_by_currency.csv` / `donations.jsonl`
с дедупом, что и у обычного пайплайна. `--concurrency N` — параллельные
запросы к VLM (llama-server должен быть поднят с `-np N`):

```bash
python3 scripts/vlm_stage.py --crops incoming/best_crops \
  --meta incoming/events_meta.jsonl --concurrency 4 --overwrite
```

Эталонный Python-движок (без cpp-сборки, работает с `models/best.pt` напрямую):

```bash
python3 scripts/fast_pipeline.py --engine py --video путь/к/видео.mp4 --overwrite
```

Прямой запуск эталонного пайплайна (минуя `fast_pipeline.py`):

```bash
python3 -m donsearcher --model models/best.pt --video путь/к/видео.mp4 \
  --device cpu --frame-step 10 --conf 0.5 --img-size 640 \
  --vlm-server-url http://127.0.0.1:8081/v1/chat/completions \
  --vlm-model Qwen3-VL --overwrite
```
