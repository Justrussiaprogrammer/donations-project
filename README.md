# Сбор информации о донатах из стримов

Проект позволяет получать данные о донатах на стримах

Для работы надо настроить сервер локальной нейросети-OCR и сохранить модель YOLO для этапа обнаружения. Сервер можно поднимать любой, в пример приведена связка модели YOLOn + llama.cpp c Qwen3-VL-8B

## Настройка среды и запуск

В случае если вы не используете openvino для пересборки модели, не используйте модель с openvino в названии

### Windows 11

Для создания среды выполнения в системе должен быть установлен Python3. Запустите терминал из папки проекта. Создайте среду выполнения:

```bash
python -m venv donate_env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\donate_env\Scripts\activate
python -m pip install -U pip
python -m pip install ultralytics requests
```

Запуск поиска из папки проекта:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\donate_env\Scripts\activate
python scripts/vlm_pipeline.py --video stream.mp4
```

### Ubuntu 24.04

Возможно вам нужно подтянуть нужные библиотеки:

```bash
sudo apt update
sudo apt install python3-venv python3-pip ffmpeg libgl1 -y
```

Запустите терминал из папки проекта. Создайте среду выполнения:

```bash
python3 -m venv donate_env
source donate_env/bin/activate
pip install -U pip
pip install ultralytics requests
```

Запуск поиска из папки проекта:

```bash
source donate_env/bin/activate
python3 scripts/vlm_pipeline.py --video stream.mp4
```

## Настройка llama.cpp

В самом простом случае можно установить подходящий вам релиз с гитхаба <https://github.com/ggml-org/llama.cpp/releases>
В случае если вам нужны рекомендации по сборке, в частности на графические процессоры Intel, они приведены ниже

Проверено для Ubuntu 24.04

Скачайте llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### CPU

#### Настройка под CPU

```bash
cmake -B build-cpu -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j$(nproc)
```

#### Запуск на CPU

Лучше поменять флаг количество потоков -t на свой

```bash
 ~/llama.cpp/build-cpu/bin/llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M \
  -np 1 -t 6 --cache-ram 0 \
  --host 127.0.0.1 --port 8081 -c 2048
```

### Vulkan

#### Настройка под Vulkan

```bash
sudo apt update
sudo apt install -y mesa-vulkan-drivers vulkan-tools

cd ~/llama.cpp

cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j$(nproc)
```

#### Запуск на Vulkan

Лучше поменять флаг количество потоков -t на свой

```bash
 ~/llama.cpp/build-vulkan/bin/llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M \
  -np 1 -t 6 --cache-ram 0 \
  --host 127.0.0.1 --port 8081 -c 2048
```

### SYCL

#### Настройка под SYCl

Установка компилятора Intel DPC++/C++:

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
  gpg --dearmor | \
  sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
  sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update

sudo apt install -y \
  intel-oneapi-compiler-dpcpp-cpp \
  intel-oneapi-mkl-devel
```

Установка intel/compute-runtime, oneapi-src/level-zero:

```bash
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-core-2_2.34.4+21428_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-opencl-2_2.34.4+21428_amd64.deb

wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/intel-ocloc_26.18.38308.1-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/intel-opencl-icd_26.18.38308.1-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libigdgmm12_22.10.0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libze-intel-gpu1_26.18.38308.1-0_amd64.deb

# Можно установить и для Windows 11, Ubuntu 22.04
wget https://github.com/oneapi-src/level-zero/releases/download/v1.28.6/libze1_1.28.6+u24.04_amd64.deb
wget https://github.com/oneapi-src/level-zero/releases/download/v1.28.6/libze-dev_1.28.6+u24.04_amd64.deb

sudo dpkg -i *.deb

# может не пригодиться
sudo apt install -y ocl-icd-libopencl1

sudo apt --fix-broken install -y

sudo reboot
```

Сборка SYCL:

```bash
cd ~/llama.cpp

source /opt/intel/oneapi/setvars.sh

cmake -B build-sycl \
  -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-sycl -j$(nproc)
```

#### Запуск на SYCL

Лучше поменять флаг количество потоков -t на свой

```bash
source /opt/intel/oneapi/setvars.sh

 ~/llama.cpp/build-sycl/bin/llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M \
  -np 1 -t 6 --cache-ram 0 --image-min-tokens 1024 \
  --host 127.0.0.1 --port 8081 -c 2048
```

## Команда для сбора данных со стрима

Запускается из корня проекта

```bash
source donate_env/bin/activate

python3 scripts/vlm_pipeline.py --model models/best.pt --video video_tests/test_fragment.mp4 \
  --device cpu --frame-step 10 --conf 0.25 --img-size 640 \
  --vlm-server-url http://127.0.0.1:8081/v1/chat/completions \
  --vlm-model Qwen3-VL --overwrite
```
