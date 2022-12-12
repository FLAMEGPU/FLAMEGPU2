# Minimum CMake version 3.18 for CUDA --std=c++17 
cmake_minimum_required(VERSION 3.18...3.25 FATAL_ERROR)

# Option to enable GTEST_DISCOVER if tests or tests_dev are enabled. Defaults to off due to runtime increase
cmake_dependent_option(FLAMEGPU_ENABLE_GTEST_DISCOVER "Enable GTEST_DISCOVER for more detailed ctest output without -VV. This dramitically increases test suite runtime to CUDA context initialisation." OFF "FLAMEGPU_BUILD_TESTS OR FLAMEGPU_BUILD_TESTS_DEV" OFF)

# Only Do anything if FLAMEGPU_BUILD_TESTS or FLAMEGPU_BUILD_TESTS_DEV is set.
if(NOT (FLAMEGPU_BUILD_TESTS OR FLAMEGPU_BUILD_TESTS_DEV))
    message(FATAL_ERROR "${CMAKE_CURRENT_LIST_FILE} requires FLAMEGPU_BUILD_TESTS or FLAMEGPU_BUILD_TESTS_DEV to be ON")
endif()

# Define the source files early, prior to projects.
# Prepare source files for the tests target
SET(TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/detail/test_compute_capability.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/detail/test_cuda.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/detail/test_wddm.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/detail/test_dependency_versions.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/detail/test_multi_thread_device.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/detail/test_CUDAEventTimer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/detail/test_SteadyClockTimer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/detail/test_cxxname.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/detail/test_rtc_multi_thread_device.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/exception/test_device_exception.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/exception/test_rtc_device_exception.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/io/test_io.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/io/test_logging.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/io/test_logging_exceptions.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/io/test_telemetry.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_environment_description.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_model.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_agent.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_message.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_agent_function.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_dependency_graph.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_layer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_subagent.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/model/test_subenvironment.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/simulation/test_cuda_simulation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/simulation/test_cuda_simulation_concurrency.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/simulation/test_cuda_ensemble.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/simulation/test_gpu_validation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/simulation/detail/test_cuda_subagent.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/simulation/detail/test_cuda_submacroenvironment.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/simulation/test_agent_vector.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/simulation/test_agent_instance.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/simulation/test_host_functions.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/simulation/test_RunPlan.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/simulation/test_RunPlanVector.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_agent_function_conditions.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_device_api.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_host_api.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/test_rtc_device_api.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/environment/test_host_environment.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/environment/test_host_macro_property.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/environment/test_subenvironment_manager.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/environment/test_device_environment.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/environment/test_device_macro_property.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/environment/test_environment_manager.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/random/test_agent_random.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/random/test_host_random.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/test_device_agent_vector.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/test_host_agent_sort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/test_host_agent_creation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/test_device_agent_creation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/detail/test_spatial_agent_sort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/detail/test_agent_state_transition.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/host_reduction/test_min.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/host_reduction/test_max.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/host_reduction/test_sum.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/host_reduction/test_reduce.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/host_reduction/test_count.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/host_reduction/test_transform_reduce.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/host_reduction/test_histogram_even.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/host_reduction/test_mean_standarddeviation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/agent/host_reduction/test_misc.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_messaging.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_spatial_2d.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_spatial_3d.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_brute_force.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_array.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_array_2d.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_array_3d.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_bucket.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/runtime/messaging/test_append_truncate.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/util/test_cleanup.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases/util/test_nvtx.cu
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

# Include CMake for managing CMAKE_CUDA_ARCHITECTURES
include(${FLAMEGPU_ROOT}/cmake/CUDAArchitectures.cmake)

# Include googletest as a dependency.
include(${FLAMEGPU_ROOT}/cmake/dependencies/googletest.cmake)

# If CTest's GoogleTest integreation is required, include GoogleTest.
if((FLAMEGPU_BUILD_TESTS OR FLAMEGPU_BUILD_TESTS_DEV) AND FLAMEGPU_ENABLE_GTEST_DISCOVER)
    include(GoogleTest)
endif()

if(FLAMEGPU_BUILD_TESTS)
    enable_testing()
    # Handle CMAKE_CUDA_ARCHITECTURES and inject code into the tests project() command
    flamegpu_init_cuda_architectures(PROJECT tests)
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
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/)
    # Add the executable and set required flags for the target
    flamegpu_add_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" FALSE)
    # Add the tests directory to the include path,
    target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    # Add the targets we depend on (this does link and include)
    target_link_libraries("${PROJECT_NAME}" PRIVATE GTest::gtest)
    # Pass a tests specific compiler definition for cheking CMAKE_CUDA_ARCHITECTURES behaviour
    flamegpu_get_minimum_cuda_architecture(min_cuda_arch)
    target_compile_definitions(${PROJECT_NAME} PRIVATE FLAMEGPU_TEST_MIN_CUDA_ARCH=${min_cuda_arch})
    # Put Within Tests filter
    flamegpu_set_target_folder("${PROJECT_NAME}" "Tests")
    # Also set as startup project (if top level project)
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT "${PROJECT_NAME}")
    # Set the default (visual studio) debugger configure_file
    set_target_properties("${PROJECT_NAME}" PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "$<$<CONFIG:Debug>:--gtest_catch_exceptions=0> --gtest_filter=*")

    # Add the tests target as to ctest, optionally using the gtest_discover integration.
    if(FLAMEGPU_ENABLE_GTEST_DISCOVER)
        # If GTEST_DISCOVER is enabled, add the unit test executable using it. This results in very long test exeuction times due to CUDA context initialisation per test
        gtest_discover_tests(
            "${PROJECT_NAME}"
            WORKING_DIRECTORY
            ${PROJECT_DIR}
        )
    else()
        # Otherwise add the whole test suite as a single test. Use ctest -VV to view per-test results in this case.
        add_test(
            NAME ${PROJECT_NAME} 
            COMMAND "${PROJECT_NAME}"
            WORKING_DIRECTORY ${PROJECT_DIR}
        )
    endif()
endif()

# If the tests_dev target is requirest, create it.
if(FLAMEGPU_BUILD_TESTS_DEV)
    enable_testing()
    # Handle CMAKE_CUDA_ARCHITECTURES and inject code into the tests project() command
    flamegpu_init_cuda_architectures(PROJECT tests_dev)
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
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/)
    # Add the executable and set required flags for the target
    flamegpu_add_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" FALSE)
    # Add the tests directory to the include path,
    target_include_directories("${PROJECT_NAME}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    # Add the targets we depend on (this does link and include)
    target_link_libraries("${PROJECT_NAME}" PRIVATE GTest::gtest)
    # Pass a tests specific compiler definition for cheking CMAKE_CUDA_ARCHITECTURES behaviour
    flamegpu_get_minimum_cuda_architecture(min_cuda_arch)
    target_compile_definitions(${PROJECT_NAME} PRIVATE FLAMEGPU_TEST_MIN_CUDA_ARCH=${min_cuda_arch})
    # Put Within Tests filter
    flamegpu_set_target_folder("${PROJECT_NAME}" "Tests")
    # Set the default (visual studio) debugger configure_file
    set_target_properties("${PROJECT_NAME}" PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "$<$<CONFIG:Debug>:--gtest_catch_exceptions=0> --gtest_filter=*")
    
    # Add the tests target as to ctest, optionally using the gtest_discover integration.
    if(FLAMEGPU_ENABLE_GTEST_DISCOVER)
        # If GTEST_DISCOVER is enabled, add the unit test executable using it. This results in very long test exeuction times due to CUDA context initialisation per test
        gtest_discover_tests(
            "${PROJECT_NAME}"
            WORKING_DIRECTORY
            ${PROJECT_DIR}
        )
    else()
        # Otherwise add the whole test suite as a single test. Use ctest -VV to view per-test results in this case.
        add_test(
            NAME ${PROJECT_NAME} 
            COMMAND "${PROJECT_NAME}"
            WORKING_DIRECTORY ${PROJECT_DIR}
        )
    endif()
endif()
