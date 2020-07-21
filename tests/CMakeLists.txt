# Set the minimum cmake version to that which supports cuda natively.
# 3.10 required for cuda -std=c++14, however 3.12 fixes some device linker errors
cmake_minimum_required(VERSION VERSION 3.12 FATAL_ERROR)

# Tests require GTest

# Apply the CUDA_ARCH target before gtest is included to resolve cmake 3.18+ warnings?
# Set the location of the ROOT flame gpu project relative to this CMakeList.txt
get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. REALPATH)

# Include common rules.
include(${FLAMEGPU_ROOT}/cmake/common.cmake)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    # Suppress installation target, as this makes a warning
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    mark_as_advanced(FORCE INSTALL_GTEST)
    mark_as_advanced(FORCE BUILD_GMOCK)
    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Name the project and set languages
project(tests CUDA CXX)

# Define output location of binary files
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # If top level project
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_SYSTEM_NAME_LOWER}-x64/${CMAKE_BUILD_TYPE}/)
else()
    # If called via add_subdirectory()
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/${CMAKE_SYSTEM_NAME_LOWER}-x64/${CMAKE_BUILD_TYPE}/)
endif()

# Enable parallel compilation
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
endif()

# Prepare list of source files
SET(TEST_CASE_SRC
    # ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_func_pointer.cu # Does not currently build
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/gpu/test_cuda_agent_model.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/gpu/test_gpu_validation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/gpu/test_cuda_subagent.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/io/test_io.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_environment_description.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_model.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_agent.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_message.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_agent_function.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_layer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_subagent.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_subenvironment.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/pop/test_pop_validation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/pop/test_agent_instance.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/sim/test_host_functions.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_agent_environment.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_agent_function_conditions.cu    
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_agent_random.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_agent_state_transition.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_device_agent_creation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_device_api.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_environment_manager.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_host_api.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_host_agent_sort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_host_agent_creation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_host_environment.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_host_random.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/host_reduction/test_min.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/host_reduction/test_max.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/host_reduction/test_sum.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/host_reduction/test_reduce.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/host_reduction/test_count.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/host_reduction/test_transform_reduce.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/host_reduction/test_histogram_even.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/host_reduction/test_misc.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_subenvironment_manager.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_messaging.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_spatial_2d.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_spatial_3d.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_brute_force.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_array.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_array_2d.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_array_3d.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_bucket.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_append_truncate.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/util/test_compute_capability.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/util/test_nvtx.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/util/test_dependency_versions.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_rtc_device_api.cu
)
SET(DEV_TEST_CASE_SRC
)
SET(OTHER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/host_reductions_common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/host_reductions_common.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/device_initialisation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/device_initialisation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/main.cu
)
SET(ALL_SRC
    ${TEST_CASE_SRC}
    ${OTHER_SRC}
)
SET(DEV_ALL_SRC
    ${DEV_TEST_CASE_SRC}
    ${OTHER_SRC}
)

# Add the executable and set required flags for the target
add_flamegpu_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" FALSE)

# Add the tests directory to the include path,
target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
# Add the targets we depend on (this does link and include)
target_link_libraries("${PROJECT_NAME}" gtest)
# target_link_libraries("${PROJECT_NAME}" gmock) # Currently unused

# Put Within Tests filter
CMAKE_SET_TARGET_FOLDER("${PROJECT_NAME}" "Tests")
CMAKE_SET_TARGET_FOLDER("gtest" "Tests/Dependencies")

# Also set as startup project (if top level project)
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT "${PROJECT_NAME}")

# Set the default (visual studio) debugger configure_file
set_target_properties("${PROJECT_NAME}" PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                                                   VS_DEBUGGER_COMMAND_ARGUMENTS "$<$<CONFIG:Debug>:--gtest_catch_exceptions=0> --gtest_filter=*")

option(TEST_DEV_MODE "Create a mini project for fast building of tests during development" OFF)
if(TEST_DEV_MODE)
    # DEVELOPMENT TESTING THING (Compact repeated version of above)
    project(tests_dev CUDA CXX)
    # Set the location of the ROOT flame gpu project relative to this CMakeList.txt
    get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. REALPATH)
    # Include common rules.
    include(${FLAMEGPU_ROOT}/cmake/common.cmake)
    # Define output location of binary files
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/${CMAKE_SYSTEM_NAME_LOWER}-x64/${CMAKE_BUILD_TYPE}/)
    # Enable parallel compilation
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    endif()
    # Add the executable and set required flags for the target
    add_flamegpu_executable("${PROJECT_NAME}" "${DEV_ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" FALSE)
    # Add the tests directory to the include path,
    target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    # Add the targets we depend on (this does link and include)
    target_link_libraries("${PROJECT_NAME}" gtest)
    # target_link_libraries("${PROJECT_NAME}" gmock) # Currently unused
    CMAKE_SET_TARGET_FOLDER("${PROJECT_NAME}" "Tests")
    
    # Set the default (visual studio) debugger configure_file
    set_target_properties("${PROJECT_NAME}" PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                                                       VS_DEBUGGER_COMMAND_ARGUMENTS "$<$<CONFIG:Debug>:--gtest_catch_exceptions=0> --gtest_filter=*")
endif()
