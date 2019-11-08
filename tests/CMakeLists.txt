# Set the minimum cmake version to that which supports cuda natively.
# 3.10 required for cuda -std=c++14, however 3.12 fixes some device linker errors
cmake_minimum_required(VERSION VERSION 3.12 FATAL_ERROR)

# Tests require GTest
#enable_testing()

# START Download & Build GoogleTest
MACRO (download_gtest)
    configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
        )
    if (result)
        message(WARNING 
                "CMake step for googletest failed: ${result}\n")
    endif()
    execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
    )

if (result)
    message(WARNING 
            "Build step for googletest-build failed: ${result}\n"
            "Attempting to continue\n"
            "You may want to disable BUILD_TESTS")
endif()
ENDMACRO()

IF(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/googletest-src)
    download_gtest()
    # Give option to disable trying to update after first run
    set(UPDATE_GTEST ON CACHE BOOL "Update GTest from Git")
ENDIF()
IF(UPDATE_GTEST)
    download_gtest()
ENDIF()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Suppress installation target, as this makes a warning
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
mark_as_advanced(FORCE INSTALL_GTEST)
mark_as_advanced(FORCE BUILD_GMOCK)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL
                 )
# END Download & Build GoogleTest

# Name the project and set languages
project(tests CUDA CXX)

# Project requires GTest include
include_directories(${GTEST_INCLUDE_DIRS})

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
    # ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_func_pointer.cu # Does not currently build
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/gpu/test_gpu_validation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_message_validation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_model_validation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/pop/test_pop_validation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/sim/test_sim_validation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_agent_random.cu
)
SET(OTHER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/device_test_functions.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/device_test_functions.cu
)
SET(ALL_SRC
    ${TEST_CASE_SRC}
    ${OTHER_SRC}
)

# Add the executable and set required flags for the target
add_flamegpu_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" FALSE)

# Add the tests directory to the include path,
target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries("${PROJECT_NAME}" gtest_main)

# Put Within Tests filter
CMAKE_SET_TARGET_FOLDER("${PROJECT_NAME}" "Tests")
CMAKE_SET_TARGET_FOLDER("gtest" "Tests/Dependencies")
CMAKE_SET_TARGET_FOLDER("gtest_main" "Tests/Dependencies")

# Also set as startup project (if top level project)
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT "${PROJECT_NAME}")
