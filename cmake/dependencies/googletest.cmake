##############
# GOOGLETEST #
##############

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})
include(FetchContent)
# Temporary CMake >= 3.30 fix https://github.com/FLAMEGPU/FLAMEGPU2/issues/1223
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    # Suppress installation target, as this makes a warning
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    mark_as_advanced(FORCE INSTALL_GTEST)
    set(BUILD_GMOCK OFF CACHE BOOL "Builds the googlemock subproject" FORCE)
    mark_as_advanced(FORCE BUILD_GMOCK)
    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    flamegpu_set_target_folder("gtest" "Tests/Dependencies")
    # Suppress warnigns from this target.
    include(${CMAKE_CURRENT_LIST_DIR}/../warnings.cmake)
    if(TARGET gtest)
        flamegpu_disable_compiler_warnings(TARGET gtest)
    endif()
endif()

# Mark some CACHE vars advanced for a cleaner GUI
mark_as_advanced(FETCHCONTENT_SOURCE_DIR_GOOGLETEST)
mark_as_advanced(FETCHCONTENT_QUIET)
mark_as_advanced(FETCHCONTENT_BASE_DIR)
mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED) 
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_GOOGLETEST) 