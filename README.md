# Сбор информации о донатах из стримов

Проект позволяет получать данные о донатах на стримах.

Конвейер: видео -> YOLO-детектор плашки доната -> группировка детекций в события ->
выбор лучшего кропа -> локальный VLM (например, Qwen3-VL) -> JSON/CSV-отчёты.

Для работы нужны две вещи помимо самого проекта:

1. **YOLO-модель** детектора плашки доната (файл кладётся в `models/`).
2. **Локальный VLM-сервер** с OpenAI-совместимым API. Сервер можно поднимать любой;
   проверенный вариант (llama.cpp + Qwen3-VL-8B, в том числе сборка под GPU Intel)
   описан в [llama_cpp_setup.md](llama_cpp_setup.md).

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

Для работы скрипта fast_script.py (по умолчанию) нужно установить дополнительные зависимости:

```bash
sudo apt update
sudo apt install g++ ffmpeg -y
```

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

python3 scripts/fast_pipeline.py --video video_tests/stream.mp4 \
  --frame-step 10 --conf 0.25 \
  --vlm-server-url http://127.0.0.1:8081/v1/chat/completions \
  --vlm-model Qwen3-VL --overwrite
```

Результаты прогона появятся в `vlm_runs/<имя_прогона>/`: `events_summary.csv`,
`totals_by_currency.csv`, `donations.jsonl`, `run_metadata.json` и папка `events/`
с лучшими кропами и кадрами.

### Движки детекции

`fast_pipeline.py` поддерживает два движка YOLO-стадии (флаг `--engine`):

- **`cpp`** (по умолчанию) - быстрый нативный детектор. Требует дополнительно:
  - системные `g++` и `ffmpeg`: `sudo apt install -y g++ ffmpeg`;
  - экспорт модели в OpenVINO в `models/best_openvino_model/`:
    `python3 -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='openvino', half=True, dynamic=False, imgsz=640)"`;
  - однократную сборку бинарника: `./cpp/build.sh`.
  - Для запуска на GPU добавьте к запуску `--cpp-device GPU`.
- **`py`** - эталонный Python-движок, без дополнительных системных зависимостей.
  Работает с `models/best.pt` напрямую: добавьте к запуску `--engine py`.

По умолчанию VLM работает параллельно с детекцией; флаг `--sequential` запускает
сначала всю детекцию, затем весь VLM (полезно для чистого замера времени стадий).

## Флаги

`fast_pipeline.py` принимает **все** флаги `vlm_pipeline.py` плюс три собственных
(`--engine`, `--cpp-binary`, `--cpp-device`). Поэтому таблицы ниже относятся к обоим скриптам; раздел «Только fast_pipeline.py» - к нему одному.

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
| `--device` | `cpu` | Устройство torch для **py-движка**: `cpu` или индекс CUDA (`0`). На cpp-движок не влияет (см. `--cpp-device`). |
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
| `--keep-top-candidates` | `3` | Сколько лучших кропов-кандидатов хранить на событие (из них выбирается финальный). |

### VLM (распознавание текста плашки)

| Флаг | По умолчанию | Что делает |
| --- | --- | --- |
| `--vlm-server-url` | `http://127.0.0.1:8081/v1/chat/completions` | Адрес OpenAI-совместимого VLM-сервера. |
| `--vlm-model` | `Qwen3-VL` | Имя модели, передаётся в запросе к серверу. |
| `--vlm-timeout` | `300` | Таймаут одного запроса к VLM, сек. |
| `--vlm-retries` | `2` | Число повторов при сетевых/серверных ошибках. |
| `--vlm-max-tokens` | `1024` | Лимит токенов в ответе модели. |
| `--vlm-temperature` | `0.0` | Температура генерации (0 - детерминированно). |
| `--skip-vlm` | выкл. | Только YOLO и группировка событий, без вызовов VLM (быстрая проверка детектора). |
| `--sequential` | выкл. | Сначала вся детекция, потом весь VLM (без перекрытия) - чистый тайминг стадий. |
| `--no-save-images` | выкл. | Не сохранять изображения (кропы, кадры) - только CSV/JSONL. |

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
python3 scripts/fast_pipeline.py --video video_tests/stream.mp4
```

То же с помощью GPU (детекция на GPU через OpenVINO):

```bash
python3 scripts/fast_pipeline.py --video video_tests/stream.mp4 --cpp-device GPU
```

Последовательный режим (вся детекция, затем весь VLM) - для чистого замера времени
каждой стадии:

```bash
python3 scripts/fast_pipeline.py --video video_tests/stream.mp4 --sequential --overwrite
```

Проверка только детектора без VLM (сервер поднимать не нужно):

```bash
python3 scripts/fast_pipeline.py --video video_tests/stream.mp4 --skip-vlm
```

Эталонный Python-движок (без cpp-сборки, работает с `models/best.pt` напрямую):

```bash
python3 scripts/fast_pipeline.py --engine py --video video_tests/stream.mp4 --overwrite
```

Прямой запуск эталонного пайплайна (минуя `fast_pipeline.py`):

```bash
python3 scripts/vlm_pipeline.py --model models/best.pt --video video_tests/stream.mp4 \
  --device cpu --frame-step 10 --conf 0.25 --img-size 640 \
  --vlm-server-url http://127.0.0.1:8081/v1/chat/completions \
  --vlm-model Qwen3-VL --overwrite
```
