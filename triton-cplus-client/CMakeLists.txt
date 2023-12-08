cmake_minimum_required (VERSION 3.18)
project(triton-cplus-client)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 14)

find_package(CURL REQUIRED)
message(STATUS "Using curl ${CURL_VERSION}")

find_package(Protobuf REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
include_directories(${Protobuf_INCLUDE_DIRS})

find_package(gRPC REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)

include_directories(/tritonclient/include)
link_directories(/tritonclient/lib)
link_libraries(httpclient_static)
find_package(OpenCV REQUIRED)

add_executable(
  image_client
  image_client.cc
  $<TARGET_OBJECTS:json-utils-library>
)
target_include_directories(
  image_client
  PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
  image_client
  PRIVATE
    grpcclient_static
    httpclient_static
    ${OpenCV_LIBS}
)
install(
  TARGETS image_client
  RUNTIME DESTINATION bin
)

add_executable(simple_http_infer_client simple_http_infer_client.cc)

target_link_libraries(
  simple_http_infer_client
  PRIVATE
    httpclient_static
)
install(
  TARGETS simple_http_infer_client
  RUNTIME DESTINATION bin
)
