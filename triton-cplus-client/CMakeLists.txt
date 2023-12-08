#cmake_minimum_required (VERSION 3.18)
cmake_minimum_required (VERSION 3.16)
project(triton-cplus-client)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 14)

#include_directories(/data0/tritonclient/include library)
#link_directories(/data0/tritonclient/lib)
include_directories(/tritonclient/include library)
link_directories(/tritonclient/lib)


set(tritonclient_LIBRARIES
absl_bad_any_cast_impl
absl_bad_optional_access
absl_bad_variant_access
absl_base
absl_city
absl_civil_time
absl_cord
absl_cord_internal
absl_cordz_functions
absl_cordz_handle
absl_cordz_info
absl_cordz_sample_token
absl_debugging_internal
absl_demangle_internal
absl_examine_stack
absl_exponential_biased
absl_failure_signal_handler
absl_flags
absl_flags_commandlineflag
absl_flags_commandlineflag_internal
absl_flags_config
absl_flags_internal
absl_flags_marshalling
absl_flags_parse
absl_flags_private_handle_accessor
absl_flags_program_name
absl_flags_reflection
absl_flags_usage
absl_flags_usage_internal
absl_graphcycles_internal
absl_hash
absl_hashtablez_sampler
absl_int128
absl_leak_check
absl_log_severity
absl_low_level_hash
absl_malloc_internal
absl_periodic_sampler
absl_random_distributions
absl_random_internal_distribution_test_util
absl_random_internal_platform
absl_random_internal_pool_urbg
absl_random_internal_randen
absl_random_internal_randen_hwaes
absl_random_internal_randen_hwaes_impl
absl_random_internal_randen_slow
absl_random_internal_seed_material
absl_random_seed_gen_exception
absl_random_seed_sequences
absl_raw_hash_set
absl_raw_logging_internal
absl_scoped_set_env
absl_spinlock_wait
absl_stacktrace
absl_status
absl_statusor
absl_str_format_internal
absl_strerror
absl_strings
absl_strings_internal
absl_symbolize
absl_synchronization
absl_throw_delegate
absl_time
absl_time_zone
address_sorting
cares
curl
gmock
gmock_main
gpr
grpc++
grpc++_alts
grpc++_error_details
grpc++_reflection
grpc++_unsecure
grpc
grpc_plugin_support
grpc_unsecure
grpcclient_static
grpcpp_channelz
gtest
gtest_main
httpclient_static
protobuf-lite
protobuf
protoc
re2
tritonasyncworkqueue
tritoncommonerror
tritoncommonlogging
tritoncommonmodelconfig
tritontableprinter
tritonthreadpool
upb
)

find_package(CUDA 11.8)
find_package(OpenCV REQUIRED)

add_executable(
  image_client
  image_client.cc
  library/common.cc
  library/json_utils.cc
)
target_include_directories(
  image_client
  PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
  image_client
  PRIVATE
    -Wl,--start-group
    ${tritonclient_LIBRARIES}
    -Wl,--end-group
    ${OpenCV_LIBS}
    pthread
    rt
    crypto
    ssl
    z
)
install(
  TARGETS image_client
  RUNTIME DESTINATION bin
)

add_executable(
  simple_http_infer_client
  simple_http_infer_client.cc
  library/common.cc
  library/json_utils.cc
)

target_link_libraries(
  simple_http_infer_client
  PRIVATE
    -Wl,--start-group
    ${tritonclient_LIBRARIES}
    -Wl,--end-group
    pthread
    rt
    crypto
    ssl
    z
)
install(
  TARGETS simple_http_infer_client
  RUNTIME DESTINATION bin
)
