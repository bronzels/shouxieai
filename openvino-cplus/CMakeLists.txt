cmake_minimum_required (VERSION 3.10)

if(POLICY CMP0025)
    cmake_policy(SET CMP0025 NEW)
endif()

project(openvino-cplus)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD 20)

string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable:177")
endif()

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64.*|aarch64.*|AARCH64.*|ARM64.*)")
    set(AARCH64 ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
    set(ARM ON)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set (CMAKE_CXX_STANDARD 11)
    set (CMAKE_CXX_EXTENSIONS OFF)
    set (CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

#find_package (gflags PATHS /usr/local/gflags REQUIRED )
find_package (gflags REQUIRED )
if(gflags_FOUND)
    message(STATUS "gflags (${gflags_VERSION}) is found at ${gflags_DIR}, include:${gflags_INCLUDE_DIR}")
endif()
#include_directories (/usr/local/gcc-13.2.0/include/c++/13.2.0 ${gflags_INCLUDE_DIR})
include_directories (${gflags_INCLUDE_DIR})
#link_2directories (/usr/local/gcc-13.2.0/lib64)

add_subdirectory(common)

set(openvinoclient_LIBRARIES
        format_reader
        ie_samples_utils
        )
set(openvinoclient_deps_LIBRARIES
        gflags
        )
include_directories(common/format_reader/include common/utils/include)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
set(ov_link_libraries openvino::runtime)

add_executable(
        samples_hello_classification_wthlabels
        samples_hello_classification_wthlabels.cpp
)
target_link_libraries(
        samples_hello_classification_wthlabels
        PRIVATE
        -Wl,--start-group
        ${openvinoclient_LIBRARIES}
        ${openvinoclient_deps_LIBRARIES}
        -Wl,--end-group
        ${ov_link_libraries}
)

find_package(OpenCV REQUIRED)
message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
add_executable(
        misc_classification_onnx_wthlabels
        misc_classification_onnx_wthlabels.cpp
)
target_include_directories(
        misc_classification_onnx_wthlabels
        PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
        misc_classification_onnx_wthlabels
        PRIVATE
        -Wl,--start-group
        ${openvinoclient_LIBRARIES}
        ${openvinoclient_deps_LIBRARIES}
        -Wl,--end-group
        ${ov_link_libraries}
        ${OpenCV_LIBS}
)
