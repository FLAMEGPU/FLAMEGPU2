# Minimum CMake version 3.18 for CUDA --std=c++17 
cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)

# Only Do anything if BUILD_TESTS or BUILD_TESTS_DEV is set.
if(NOT (BUILD_TESTS OR BUILD_TESTS_DEV))
    message(FATAL_ERROR "${CMAKE_CURRENT_LIST_FILE} requires BUILD_TESTS or BUILD_TESTS_DEV to be ON")
endif()

# Define the source files early, prior to projects.
# Prepare source files for the tests target
SET(TESTS_SRC
    # ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_func_pointer.cu # Does not currently build
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/exception/test_device_exception.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/gpu/test_cuda_simulation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/gpu/test_cuda_simulation_concurrency.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/gpu/test_cuda_ensemble.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/gpu/test_gpu_validation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/gpu/test_cuda_subagent.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/io/test_io.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/io/test_logging.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/io/test_logging_exceptions.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_environment_description.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_model.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_agent.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_message.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_agent_function.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_dependency_graph.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_layer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_subagent.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_subenvironment.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/pop/test_agent_vector.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/pop/test_agent_instance.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/pop/test_device_agent_vector.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/sim/test_host_functions.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/sim/test_RunPlan.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/sim/test_RunPlanVector.cu
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
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/util/test_wddm.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/util/test_nvtx.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/util/test_dependency_versions.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/util/test_multi_thread_device.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/util/test_CUDAEventTimer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/util/test_SteadyClockTimer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/util/test_cxxname.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_rtc_device_api.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/util/test_rtc_multi_thread_device.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/exception/test_rtc_device_exception.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/test_namespaces/test_namespaces.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/test_namespaces/test_rtc_namespaces.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/test_version.cpp
)
# Source files for the tests_dev target 
SET(TESTS_DEV_SRC
)
# Common source files for tests and test_dev
SET(HELPERS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/host_reductions_common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/host_reductions_common.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/device_initialisation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/device_initialisation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/main.cu
)

# Set the location of the ROOT flame gpu project relative to this CMakeList.txt
get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. REALPATH)

# Include googletest as a dependency.
include(${FLAMEGPU_ROOT}/cmake/dependencies/googletest.cmake)

if(BUILD_TESTS)
    # Name the project and set languages
    project(tests CUDA CXX)
    # Include common rules.
    include(${FLAMEGPU_ROOT}/cmake/common.cmake)
    # Set the source for this projcet
    SET(ALL_SRC
        ${TESTS_SRC}
        ${HELPERS_SRC}
    )
    # Define output location of binary files
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        # If top level project
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/)
    else()
        # If called via add_subdirectory()
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/${CMAKE_BUILD_TYPE}/)
    endif()
    # Add the executable and set required flags for the target
    add_flamegpu_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" FALSE)
    # Add the tests directory to the include path,
    target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    # Add the targets we depend on (this does link and include)
    target_link_libraries("${PROJECT_NAME}" GTest::gtest)
    # target_link_libraries("${PROJECT_NAME}" gmock) # Currently unused
    # Put Within Tests filter
    CMAKE_SET_TARGET_FOLDER("${PROJECT_NAME}" "Tests")
    # Also set as startup project (if top level project)
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT "${PROJECT_NAME}")
    # Set the default (visual studio) debugger configure_file
    set_target_properties("${PROJECT_NAME}" PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "$<$<CONFIG:Debug>:--gtest_catch_exceptions=0> --gtest_filter=*")
endif()

# If the tests_dev target is requirest, create it.
if(BUILD_TESTS_DEV)
    # DEVELOPMENT TESTING THING (Compact repeated version of above)
    project(tests_dev CUDA CXX)
    # Include common rules.
    include(${FLAMEGPU_ROOT}/cmake/common.cmake)
    # Set the source for this projcet
    SET(ALL_SRC
        ${TESTS_DEV_SRC}
        ${HELPERS_SRC}
    )
    # Define output location of binary files
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        # If top level project
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/)
    else()
        # If called via add_subdirectory()
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/${CMAKE_BUILD_TYPE}/)
    endif()
    # Add the executable and set required flags for the target
    add_flamegpu_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" FALSE)
    # Add the tests directory to the include path,
    target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    # Add the targets we depend on (this does link and include)
    target_link_libraries("${PROJECT_NAME}" GTest::gtest)
    # target_link_libraries("${PROJECT_NAME}" gmock) # Currently unused
    CMAKE_SET_TARGET_FOLDER("${PROJECT_NAME}" "Tests")
    # Set the default (visual studio) debugger configure_file
    set_target_properties("${PROJECT_NAME}" PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "$<$<CONFIG:Debug>:--gtest_catch_exceptions=0> --gtest_filter=*")
endif()
