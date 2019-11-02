# Set the minimum cmake version to that which supports cuda natively.
# 3.10 required for cuda -std=c++14, however 3.12 fixes some device linker errors
cmake_minimum_required(VERSION VERSION 3.12 FATAL_ERROR)

project(FLAMEGPU2)

# Don't create installation scripts (and hide CMAKE_INSTALL_PREFIX from cmake-gui)
set(CMAKE_SKIP_INSTALL_RULES TRUE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE INTERNAL ""  FORCE)

# Option to enable/disable building the static library
option(BUILD_FLAMEGPU2 "Enable building FLAMEGPU2 library" ON)
# Option to enable building all examples
option(BUILD_ALL_EXAMPLES "Enable building examples" ON)
# Option to explicitly build the MAS example. Defaults to off.
option(BUILD_EXAMPLE_MAS "Enable building examples/mas" OFF)
# Option to explicitly build the simple_example example. Defaults to off.
option(BUILD_EXAMPLE_SIMPLE_EXAMPLE "Enable building examples/simple_example" OFF)
# Option to enable/disable tests.
option(BUILD_TESTS "Enable building tests" OFF)

# Define a function to add a lint target.
find_file(CPPLINT NAMES cpplint cpplint.exe)
if(CPPLINT)
  # Add a custom target to lint all child projects. Dependencies are specified in child projects.
  add_custom_target(all_lint)
  set_target_properties(all_lint PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

# Add the library building subdirectory
if(BUILD_FLAMEGPU2)
    add_subdirectory(src)
endif()

# Add each example
if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_MAS)
    add_subdirectory(examples/mas)
endif()

if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_SIMPLE_EXAMPLE)
    add_subdirectory(examples/simple_example)
endif()

# Add the tests directory (if required)
if(BUILD_TESTS)
    add_subdirectory(tests)
    # Put within Lint filter
    CMAKE_SET_TARGET_FOLDER(all_lint "Tests")
endif()

# Put all_lint within Lint filter
CMAKE_SET_TARGET_FOLDER(all_lint "Lint")

# message("Generated with CMAKE_BUILD_TYPE types: ${CMAKE_BUILD_TYPE}")
# message("Generated with config types: ${CMAKE_CONFIGURATION_TYPES}")
