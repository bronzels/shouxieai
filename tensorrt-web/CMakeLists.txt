cmake_minimum_required(VERSION 3.11)
project(naive-trt-cpp-min)
set(CMAKE_CXX_STANDARD 14)
file(COPY model1.onnx DESTINATION .)
file(COPY model2.onnx DESTINATION .)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message("CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
message("CUDA_LIBRARIES = ${CUDA_LIBRARIES}")

include_directories(/root/trt/include)
link_directories(/root/trt/lib)
link_libraries(nvinfer nvonnxparser ${CUDA_LIBRARIES})

add_executable(example1 example1.cpp)
add_executable(example2 example2.cpp)
add_executable(example2prof example2prof.cpp)

