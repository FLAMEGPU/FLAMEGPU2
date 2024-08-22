############
# Tinyxml2 #
############

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})
include(FetchContent)

cmake_policy(SET CMP0079 NEW)
# Temporary CMake >= 3.30 fix https://github.com/FLAMEGPU/FLAMEGPU2/issues/1223
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

# Change the source_dir to allow inclusion via tinyxml2/tinyxml2.h rather than tinyxml2.h
FetchContent_Declare(
    tinyxml2
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
    GIT_TAG        9.0.0
    GIT_SHALLOW    1
    SOURCE_DIR     ${FETCHCONTENT_BASE_DIR}/tinyxml2-src/tinyxml2
    GIT_PROGRESS   ON
    # UPDATE_DISCONNECTED   ON
)

# @todo - try finding the pacakge first, assuming it sets system correctly when used.
FetchContent_GetProperties(tinyxml2)
if(NOT tinyxml2_POPULATED)
    FetchContent_Populate(tinyxml2)
    
    # The Tinxyxml2 repository does not include tinyxml2.h in a subdirectory, so include "tinyxml2.h" would be used not "tinyxml2/tinyxml2.h"
    # To avoid this, do not use add_subdirectory, instead create our own cmake target for tinyxml2.
    # @todo - a possible alternative (from tinyxml2 9.0.0?) might be to run the install target of tinyxml2, and then use find package in config mode. Would require investigation. 
    # Disabled:
    # get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_LIST_DIR}/../ REALPATH)
    # add_subdirectory(${tinyxml2_SOURCE_DIR} ${tinyxml2_BINARY_DIR})

    # If the target does not already exist, add it.
    # @todo - make this far more robust.
    if(NOT TARGET tinyxml2)
        # @todo - make a dynamically generated CMakeLists.txt which can be add_subdirectory'd instead, so that the .vcxproj goes in a folder. Just adding a project doesn't work.
        # project(tinyxml2 LANGUAGES CXX)

        # Set location of static library files
        # Define output location of static library
        if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
            # If top level project
            SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE}/)
        else()
            # If called via add_subdirectory()
            SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/${CMAKE_BUILD_TYPE}/)
        endif()

        # Depends on the cpp and header files in case of changes
        add_library(tinyxml2 STATIC ${tinyxml2_SOURCE_DIR}/tinyxml2.cpp ${tinyxml2_SOURCE_DIR}/tinyxml2.h)
        # Pic is sensible for any library
        set_property(TARGET tinyxml2 PROPERTY POSITION_INDEPENDENT_CODE ON)
    
        # Specify the include directory, to be forwared to targets which link against the tinyxml2 target.
        # Mark this as SYSTEM INTERFACE, so that it does not result in compiler warnings being generated for dependent projects.
        # For our use  case, this is up a folder so we can use tinyxml2/tinyxml2.h as the include, by resolving the relative path to get an abs path
        get_filename_component(tinyxml2_inc_dir ${tinyxml2_SOURCE_DIR}/../ REALPATH)
        target_include_directories(tinyxml2 SYSTEM INTERFACE ${tinyxml2_inc_dir})
        
        # Add some compile time definitions
        target_compile_definitions(tinyxml2 INTERFACE $<$<CONFIG:Debug>:TINYXML2_DEBUG>)
        set_target_properties(tinyxml2 PROPERTIES
            COMPILE_DEFINITIONS "TINYXML2_EXPORT"
        )
        if(MSVC)
            target_compile_definitions(tinyxml2 PUBLIC -D_CRT_SECURE_NO_WARNINGS)
        endif(MSVC)

        # Suppress warnigns from this target.
        include(${CMAKE_CURRENT_LIST_DIR}/../warnings.cmake)
        flamegpu_disable_compiler_warnings(TARGET tinyxml2)

        # Create an alias target for tinyxml2 to namespace it / make it more like other modern cmake 
        add_library(Tinyxml2::tinyxml2 ALIAS tinyxml2)

    endif()
endif()

# Mark some CACHE vars advanced for a cleaner GUI
mark_as_advanced(FETCHCONTENT_SOURCE_DIR_TINYXML2)
mark_as_advanced(FETCHCONTENT_QUIET)
mark_as_advanced(FETCHCONTENT_BASE_DIR)
mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED) 
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_TINYXML2) 