# Cross-platform build for fast_detector (Linux / macOS / Windows).
#
# OpenVINO is taken from the pip wheel inside the project venv (donate_env),
# which ships OpenVINOConfig.cmake + the runtime libs/plugins on every platform
# — no system OpenVINO install required. Point -DOpenVINO_DIR=... at a system
# install if you prefer one.
#
# Usage (any OS, from the cpp/ dir):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release
# The binary lands next to this file as ./fast_detector (fast_detector.exe on
# Windows), which is what scripts/fast_pipeline.py expects by default.

cmake_minimum_required(VERSION 3.16)
project(fast_detector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Emit the binary into the cpp/ source dir for both single- and multi-config
# generators, so the default --cpp-binary path keeps working.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg} "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()

# --- Locate OpenVINO from the pip wheel (unless OpenVINO_DIR is supplied) ---
if(NOT OpenVINO_DIR)
    file(GLOB _ov_cmake_dirs
        "${CMAKE_CURRENT_SOURCE_DIR}/../donate_env/lib/python*/site-packages/openvino/cmake"  # Linux/macOS venv
        "${CMAKE_CURRENT_SOURCE_DIR}/../donate_env/Lib/site-packages/openvino/cmake")          # Windows venv
    if(_ov_cmake_dirs)
        list(GET _ov_cmake_dirs 0 OpenVINO_DIR)
    endif()
endif()
if(OpenVINO_DIR)
    message(STATUS "fast_detector: using OpenVINO from ${OpenVINO_DIR}")
    if(NOT TBB_DIR)
        set(TBB_DIR "${OpenVINO_DIR}")  # TBBConfig.cmake ships next to OpenVINOConfig.cmake
    endif()
endif()

find_package(OpenVINO REQUIRED)
find_package(OpenMP QUIET)

add_executable(fast_detector fast_detector.cpp)
target_link_libraries(fast_detector PRIVATE openvino::runtime)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fast_detector PRIVATE OpenMP::OpenMP_CXX)
endif()
if(NOT WIN32)
    target_link_libraries(fast_detector PRIVATE pthread)
endif()

# manylinux OpenVINO wheels are built with the old libstdc++ string ABI.
if(UNIX AND NOT APPLE)
    target_compile_definitions(fast_detector PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endif()

# -march=native is fast but ties the binary to the build machine's CPU (an
# older CPU would SIGILL). On by default for local build-on-target use; turn
# off (-DNATIVE_ARCH=OFF) when the binary may run on a different machine.
option(NATIVE_ARCH "Optimize for the build machine's CPU (-march=native)" ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(fast_detector PRIVATE -O3 -Wall)
    if(NATIVE_ARCH)
        target_compile_options(fast_detector PRIVATE -march=native)
    endif()
elseif(MSVC)
    target_compile_options(fast_detector PRIVATE /O2)
endif()

# Bake the wheel's lib dir into the rpath so the loader finds libopenvino + the
# device plugins at runtime (Linux/macOS). On Windows the DLLs are resolved via
# PATH — run inside the activated venv or add openvino/libs to PATH.
if(OpenVINO_DIR AND NOT WIN32)
    get_filename_component(_ov_libs "${OpenVINO_DIR}/../libs" ABSOLUTE)
    set_target_properties(fast_detector PROPERTIES
        BUILD_RPATH "${_ov_libs}"
        INSTALL_RPATH "${_ov_libs}")
endif()
