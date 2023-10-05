cmake_minimum_required(VERSION 3.4...3.18)
project(democuda)
set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(PythonInterp 3.9 REQUIRED)
find_package(PythonLibs 3.9 REQUIRED)
message("PYTHON_INCLUDE_DIR = ${PYTHON_INCLUDE_DIR}")
message("PYTHON_LIBRARY = ${PYTHON_LIBRARY}")

# 指定pybind11路径
set(PYBIND11_DIR pybind11-2.11.1)
add_subdirectory(${PYBIND11_DIR})

#cuda
find_package(CUDA)
message("CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")

# 指定源码
#file(GLOB SOURCES "src/*.{c,cu,cpp}")
file(GLOB SOURCES "src/*.c" "src/*.cu" "src/*.cpp")
message("SOURCES = ${SOURCES}")

# 添加头文件目录
include_directories(${PYTHON_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}  /usr/local/cuda/samples/cuda-samples/Common ${PYBIND11_DIR}/include ${CMAKE_SOURCE_DIR}/src)

link_directories(/usr/local/cuda/lib64 /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64)

cuda_add_library(democuda SHARED
        ${SOURCES})
target_link_libraries(democuda
        ${PYTHON_LIBRARY}
        cudart)
# 设置动态库保存路径
set_target_properties(democuda PROPERTIES PREFIX "")
set_target_properties(democuda PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
