cmake_minimum_required(VERSION 3.4...3.18)
project(demo)
set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif ()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# 指定PYTHON_EXECUTABLE
if (NOT DEFINED PYTHON_EXECUTABLE)
set(PYTHON_EXECUTABLE "/data0/envs/deepspeed/bin/python3.9" CACHE PATH "Path to PYTHON_EXECUTABLE")
endif ()
# 指定PYTHON_INCLUDE_DIR
if (NOT DEFINED PYTHON_INCLUDE_DIR)
set(PYTHON_INCLUDE_DIR "/data0/envs/deepspeed/include/python3.9" CACHE PATH "Path to PYTHON_INCLUDE_DIR")
endif ()
# 指定PYTHON_LIBRARY
if (NOT DEFINED PYTHON_LIBRARY)
set(PYTHON_LIBRARY "/data0/envs/deepspeed/lib/libpython3.9.so" CACHE PATH "Path to PYTHON_LIBRARY")
endif ()

# 指定pybind11路径
add_subdirectory(pybind11-2.11.1)
# 指定源码
set(DEMO_SOURCES ${CMAKE_SOURCE_DIR}/src/demo.cc)
# 生成动态库
pybind11_add_module(demo SHARED ${DEMO_SOURCES})
# 添加头文件目录
target_include_directories(demo PRIVATE ${CMAKE_SOURCE_DIR}/src)

# opencv依赖
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
message(OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS})
target_include_directories(demo PRIVATE ${OpenCV_INCLUDE_DIRS})
message(OpenCV_LIBRARIES: ${OpenCV_LIBRARIES})
target_link_libraries(demo PRIVATE ${OpenCV_LIBRARIES})
else (OpenCV_FOUND)
message(FATAL_ERROR "OpenCV library not found")
endif (OpenCV_FOUND)

# 设置动态库保存路径
set_target_properties(demo PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

target_compile_definitions(demo PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
