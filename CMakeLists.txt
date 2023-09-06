cmake_minimum_required(VERSION 3.5)
project(avatarXM)

set(onnxruntime_INSTALL_PREFIX /usr/local)
set(onnxruntime_INCLUDE_DIRS
        ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime
)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_PREFIX_PATH)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED PATHS "/home/suimang/ls-dev/env/libtorch")

file(GLOB sources ${CMAKE_SOURCE_DIR}/src/*.cpp)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${onnxruntime_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        /usr/local/include
        include
)

add_executable(avatarXM
        tools/main.cpp
        ${sources}
)

target_link_libraries(avatarXM PRIVATE
        ${OpenCV_LIBS}
        "${onnxruntime_INSTALL_PREFIX}/lib/libonnxruntime.so"
        "${onnxruntime_INSTALL_PREFIX}/lib/libonnxruntime_providers_cuda.so"
        "${onnxruntime_INSTALL_PREFIX}/lib/libonnxruntime_providers_shared.so"
        "${TORCH_LIBRARIES}"
        yaml-cpp
)