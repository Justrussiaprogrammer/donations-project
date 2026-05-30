# Описание проекта по сбору информации о донатах из стримов

## Настройка llama.cpp

## Команды для запуска qwen3-VL-8b

Проверено для Ubuntu 24.04

### Запуск CPU

```bash
 ~/llama.cpp/build/bin/llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M -np 1 -t 6 --cache-ram 0 --host 127.0.0.1   --port 8081   -c 2048
```

### Установка и запуск вулкана

```bash
sudo apt update
sudo apt install -y mesa-vulkan-drivers vulkan-tools

cd ~/llama.cpp

cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j$(nproc)
```

```bash
 ~/llama.cpp/build-vulkan/bin/llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M -np 1 -t 6 --cache-ram 0 --host 127.0.0.1   --port 8081   -c 2048
```

### Установка и запуск sycl

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

Сборка sycl:

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

```bash
source /opt/intel/oneapi/setvars.sh

 ~/llama.cpp/build-sycl/bin/llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M -np 1 -t 6 --cache-ram 0 --image-min-tokens 1024 --host 127.0.0.1   --port 8081   -c 2048
```

Для запуска с моделями с предустановленным размышлением, надо его отключать через добавление флага --reasoning off

## Команда для обработки видео

Запускается из корня проекта

```bash
source donate_env/bin/activate

python3 scripts/vlm_pipeline.py   --project-dir .   --model models/best.pt   --video video_tests/test_fragment.mp4   --device cpu   --frame-step 10   --conf 0.25   --img-size 640   --vlm-server-url http://127.0.0.1:8081/v1/chat/completions   --vlm-model Qwen3-VL   --overwrite
```
