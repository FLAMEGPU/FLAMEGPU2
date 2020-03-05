# Set the minimum cmake version to that which supports cuda natively.
# 3.10 required for cuda -std=c++14, however 3.12 fixes some device linker errors
cmake_minimum_required(VERSION VERSION 3.12 FATAL_ERROR)

project(FLAMEGPU2 LANGUAGES NONE)

# Don't create installation scripts (and hide CMAKE_INSTALL_PREFIX from cmake-gui)
set(CMAKE_SKIP_INSTALL_RULES TRUE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE INTERNAL ""  FORCE)

# See what languages are supported
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER STREQUAL NOTFOUND)
    #Not able to build code, so just make docs    
    message("Suitable compilers for building code not found.\n" "Attempting generation of minimal documentation only project.")    
    get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)
    include(./cmake/doxygen.cmake)
    if(${BUILD_API_DOCUMENTATION})
        create_doxygen_target("${FLAMEGPU_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" "")
    endif()
    return()
endif()

# Option to enable/disable building the static library
option(BUILD_FLAMEGPU2 "Enable building FLAMEGPU2 library" ON)
# Option to enable building all examples
option(BUILD_ALL_EXAMPLES "Enable building examples" ON)
# Option to explicitly build the MAS example. Defaults to off.
option(BUILD_EXAMPLE_MAS "Enable building examples/mas" OFF)
# Option to explicitly build the simple_example example. Defaults to off.
option(BUILD_EXAMPLE_SIMPLE_EXAMPLE "Enable building examples/simple_example" OFF)
# Option to explicitly build the rtc_example (Runtime Compilation) example. Defaults to off.
option(BUILD_EXAMPLE_RTC_EXAMPLE "Enable building examples/rtc_example" OFF)
# Option to enable/disable tests.
option(BUILD_TESTS "Enable building tests" OFF)

# Option to enable/disable NVTX markers for improved profiling
option(NVTX "Enable NVTX markers for improved profiling (if aviailable). Implied by profile builds" OFF)

# Define a function to add a lint target.
find_file(CPPLINT NAMES cpplint cpplint.exe)
if(CPPLINT)
  # Add a custom target to lint all child projects. Dependencies are specified in child projects.
  add_custom_target(all_lint)
  # Don't trigger this target on ALL_BUILD or Visual Studio 'Rebuild Solution'
  set_target_properties(all_lint PROPERTIES EXCLUDE_FROM_ALL TRUE)
  # set_target_properties(all_lint PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE)
endif()

# Add the library building subdirectory
if(BUILD_FLAMEGPU2)
    add_subdirectory(src "${PROJECT_BINARY_DIR}/FLAMEGPU2")
    # Set as startup project
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT flamegpu2)
endif()

# Add each example
if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_MAS)
    add_subdirectory(examples/mas)
endif()

if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_SIMPLE_EXAMPLE)
    add_subdirectory(examples/simple_example)
endif()

if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_RTC_EXAMPLE)
    add_subdirectory(examples/rtc_example)
endif()

if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_HOST_FUNCTIONS)
    add_subdirectory(examples/host_functions)
endif()

if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_CIRCLES_BRUTE_FORCE)
    add_subdirectory(examples/circles_bruteforce)
endif()

if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_CIRCLES_SPATIAL_3D)
    add_subdirectory(examples/circles_spatial3D)
endif()

# Add the tests directory (if required)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Put all_lint within Lint filter
CMAKE_SET_TARGET_FOLDER(all_lint "Lint")


# message("Generated with CMAKE_BUILD_TYPE types: ${CMAKE_BUILD_TYPE}")
# message("Generated with config types: ${CMAKE_CONFIGURATION_TYPES}")
