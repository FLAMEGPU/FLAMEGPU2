############
# Tinyxml2 #
############

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})
include(FetchContent)

cmake_policy(SET CMP0079 NEW)

# Change the source_dir to allow inclusion via tinyxml2/tinyxml2.h rather than tinyxml2.h
FetchContent_Declare(
    tinyxml2
    GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
    GIT_TAG        8.0.0
    GIT_SHALLOW    1
    SOURCE_DIR     ${FETCHCONTENT_BASE_DIR}/tinyxml2-src/tinyxml2
    GIT_PROGRESS   ON
    # UPDATE_DISCONNECTED   ON
)

FetchContent_GetProperties(tinyxml2)
if(NOT tinyxml2_POPULATED)
    FetchContent_Populate(tinyxml2)
    
    # when adding via subdirectory tinyxml2 target doesn't seem to forward includes
    # Which is super annoying.  
    # get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_LIST_DIR}/../ REALPATH)
    # add_subdirectory(${tinyxml2_SOURCE_DIR} ${tinyxml2_BINARY_DIR})

    # Instead, build our own version. This is quite grim but better than the alternative?
    # @todo - make this far more robust.
    if(NOT TARGET tinyxml2)
        # @todo - make a dynamically generated CMakeLists.txt which can be add_subdirectory'd instead, so that the .vcxproj goes in a folder. Just adding a project doesn't work.
        # project(tinyxml2 LANGUAGES CXX)

        # Set location of static library files
        # Define output location of static library
        STRING(TOLOWER "${CMAKE_SYSTEM_NAME}" CMAKE_SYSTEM_NAME_LOWER)
        if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
            # If top level project
            SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/${CMAKE_SYSTEM_NAME_LOWER}-x64/${CMAKE_BUILD_TYPE}/)
        else()
            # If called via add_subdirectory()
            SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/${CMAKE_SYSTEM_NAME_LOWER}-x64/${CMAKE_BUILD_TYPE}/)
        endif()

        # Depends on the cpp and header files in case of changes
        add_library(tinyxml2 STATIC ${tinyxml2_SOURCE_DIR}/tinyxml2.cpp ${tinyxml2_SOURCE_DIR}/tinyxml2.h)
        # Pic is sensible for any library
        set_property(TARGET tinyxml2 PROPERTY POSITION_INDEPENDENT_CODE ON)
    
        # Specify the interface headers library, to be forwarded.
        # For our use  case, this is up a folder so we can use tinyxml2/tinyxml2.h as the include.
        # Find the resolved abs path to this. 
        get_filename_component(tinyxml2_inc_dir ${tinyxml2_SOURCE_DIR}/../ REALPATH)
        target_include_directories(tinyxml2 INTERFACE ${tinyxml2_inc_dir})
        
        # Add some compile time definitions
        target_compile_definitions(tinyxml2 INTERFACE $<$<CONFIG:Debug>:TINYXML2_DEBUG>)
        set_target_properties(tinyxml2 PROPERTIES
            COMPILE_DEFINITIONS "TINYXML2_EXPORT"
        )
        if(MSVC)
            target_compile_definitions(tinyxml2 PUBLIC -D_CRT_SECURE_NO_WARNINGS)
        endif(MSVC)
    endif()
endif()
