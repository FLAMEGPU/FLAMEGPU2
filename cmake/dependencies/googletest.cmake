##############
# GOOGLETEST #
##############

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})
include(FetchContent)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    list(APPEND DEPENDENCY_ARGS EXCLUDE_FROM_ALL)
endif()

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  ${DEPENDENCY_ARGS}
)

FetchContent_GetProperties(googletest)
# Suppress installation target, as this makes a warning
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
mark_as_advanced(FORCE INSTALL_GTEST)
set(BUILD_GMOCK OFF CACHE BOOL "Builds the googlemock subproject" FORCE)
mark_as_advanced(FORCE BUILD_GMOCK)
# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# Download and add_subdirectory
FetchContent_MakeAvailable(googletest)
# Setup the visual studiop project filter
flamegpu_set_target_folder("gtest" "Tests/Dependencies")
# Suppress warnigns from this target.
include(${CMAKE_CURRENT_LIST_DIR}/../warnings.cmake)
if(TARGET gtest)
    flamegpu_disable_compiler_warnings(TARGET gtest)
endif()

# Mark some CACHE vars advanced for a cleaner GUI
mark_as_advanced(FETCHCONTENT_SOURCE_DIR_GOOGLETEST)
mark_as_advanced(FETCHCONTENT_QUIET)
mark_as_advanced(FETCHCONTENT_BASE_DIR)
mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED) 
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_GOOGLETEST) 