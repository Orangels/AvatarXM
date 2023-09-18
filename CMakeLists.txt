cmake_minimum_required(VERSION 3.5)
project(avatarXM)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g2 -Wall")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")

set(onnxruntime_INSTALL_PREFIX /usr/local)
set(onnxruntime_INCLUDE_DIRS
        ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime
)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_PREFIX_PATH)
set(SYS_USR_LIB /usr/lib/x86_64-linux-gnu)


#find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED PATHS "/home/suimang/ls-dev/env/libtorch")

file(GLOB sources ${CMAKE_SOURCE_DIR}/src/*.cpp)

include_directories(
#        ${OpenCV_INCLUDE_DIRS}
        /usr/include/opencv4
        ${onnxruntime_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        /home/suimang/ls-dev/env/libtorch/include/torch/csrc/api/include
        /usr/local/include
        include
        /home/suimang/ls-dev/env/kaldi-master/src
        /home/suimang/ls-dev/env/kaldi-master/tools/openfst/include
)

add_executable(avatarXM
#        UnitTests/RunAvatar.cpp
        include/utils/extraAudioFeature.h
        src/utils/extraAudioFeature.cpp
        #        UnitTests/torch_mat_test.cpp
        #        UnitTests/torch_gaussianBlur_test.cpp
                tools/main.cpp

        ${sources}
        src/task/avatarHandler.cpp
        include/utils/utils.h
        include/structures/structs.h
        include/task/avatarHandler.h
        include/avatar.h
)

file(GLOB KALDI_LIB "/home/suimang/ls-dev/env/kaldi-master/src/lib/*.so")
target_link_options(avatarXM PRIVATE -L/opt/intel/oneapi/mkl/2022.0.2/lib/intel64)
target_link_libraries(avatarXM PRIVATE
#        ${OpenCV_LIBS}
        ${SYS_USR_LIB}/libopencv_core.so
        ${SYS_USR_LIB}/libopencv_imgproc.so
        ${SYS_USR_LIB}/libopencv_highgui.so
        ${SYS_USR_LIB}/libopencv_imgcodecs.so
        ${SYS_USR_LIB}/libopencv_videoio.so
        ${SYS_USR_LIB}/libopencv_dnn.so

        "${onnxruntime_INSTALL_PREFIX}/lib/libonnxruntime.so"
        "${onnxruntime_INSTALL_PREFIX}/lib/libonnxruntime_providers_cuda.so"
        "${onnxruntime_INSTALL_PREFIX}/lib/libonnxruntime_providers_shared.so"
        "${TORCH_LIBRARIES}"
        yaml-cpp

        #kaldi
#        /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_ilp64.so
#        /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so
#        /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.so
        ${KALDI_LIB}

)