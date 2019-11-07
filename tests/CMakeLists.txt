# Set the minimum cmake version to that which supports cuda natively.
# 3.10 required for cuda -std=c++14, however 3.12 fixes some device linker errors
cmake_minimum_required(VERSION VERSION 3.12 FATAL_ERROR)

# Tests require boost
find_package(Boost 1.54 REQUIRED)

# Name the project and set languages
project(tests CUDA CXX)

# Project requires boost include
include_directories(${Boost_INCLUDE_DIRS})

# Set the location of the ROOT flame gpu project relative to this CMakeList.txt
get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. REALPATH)

# Include common rules.
include(${FLAMEGPU_ROOT}/cmake/common.cmake)

# Define output location of binary files
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # If top level project
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_SYSTEM_NAME_LOWER}-x64/${CMAKE_BUILD_TYPE}/)
else()
    # If called via add_subdirectory()
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/${CMAKE_SYSTEM_NAME_LOWER}-x64/${CMAKE_BUILD_TYPE}/)
endif()

# Prepare list of source files
SET(TEST_CASE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_func_pointer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/gpu/test_gpu_validation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_message_validation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_model_validation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/pop/test_pop_validation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/sim/test_sim_validation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_agent_random.h
)
SET(OTHER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/device_test_functions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/device_test_functions.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_all.cu
)
SET(ALL_SRC
    ${TEST_CASE_SRC}
    ${OTHER_SRC}
)

# Add the executable and set required flags for the target
add_flamegpu_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" FALSE)

# Add the tests directory to the include path,
target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# Put Within Tests filter
CMAKE_SET_TARGET_FOLDER("${PROJECT_NAME}" "Tests")

# Also set as startup project (if top level project)
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT "${PROJECT_NAME}")
