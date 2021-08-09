cmake_minimum_required(VERSION 3.19)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(VIT_CUDA_VERSION 10.2) # 11.4 or 10.2 for now
set(CUDA_VERSION ${VIT_CUDA_VERSION})
#set(VIT_CUDA_ARCHITECTURES 75)
set(VIT_CUDA_ARCHITECTURES 62)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-${VIT_CUDA_VERSION}/bin/nvcc")
#set(CMAKE_CXX_COMPILER "/usr/lib/llvm-12/bin/clang++")

project(opencv_and_cuda LANGUAGES CXX CUDA)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-${VIT_CUDA_VERSION}/)
set(CUDA_INCLUDE_DIRS /usr/local/cuda-${VIT_CUDA_VERSION}/include/)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CUDA_FLAGS "")

find_package(CUDA ${VIT_CUDA_VERSION})
find_package(OpenCV REQUIRED)

include_directories(
        "/usr/local/cuda-${VIT_CUDA_VERSION}/include/"
        #        "/usr/local/cuda-${VIT_CUDA_VERSION}/samples/common/inc/"
        "/usr/local/cuda-${VIT_CUDA_VERSION}/targets/x86_64-linux/include/"
)

find_package(OpenCV REQUIRED)
link_libraries(GL GLU glut cufft)

add_executable(opencv_and_cuda main.cu main.h run.cpp)

set_target_properties(
        opencv_and_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${VIT_CUDA_ARCHITECTURES}
)

target_link_libraries(opencv_and_cuda
        ${OpenCV_LIBS}
        )