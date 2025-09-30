#######
# glm #
#######
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})
include(FetchContent)

# Head of master at point BugFix for NVRTC support was merged
# Specify an invalid SOURCE_SUBDIR to prevent FetchContent_MakeAvailable from adding the CMakeLists.txt
FetchContent_Declare(
    glm
    URL            "https://github.com/g-truc/glm/archive/66062497b104ca7c297321bd0e970869b1e6ece5.zip"
    SOURCE_SUBDIR "do_not_use_add_subirectory"
)
FetchContent_GetProperties(glm)
FetchContent_MakeAvailable(glm)

if (NOT TARGET glm::glm)
    # glm CMake wants to generate the find file in a system location, so handle it manually
    # Find the path, just incase
    find_path(glm_INCLUDE_DIRS
    NAMES
        glm/glm.hpp
    PATHS
        ${glm_SOURCE_DIR}
    NO_CACHE
    )
    if(glm_INCLUDE_DIRS)
        # Define an imported interface target
        add_library(glm::glm INTERFACE IMPORTED)
        # Specify the location of the headers (but actually the parent dir, so include <glm/glm.hpp> can be used.)
        target_include_directories(glm::glm INTERFACE "${glm_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "Error during creation og `glm::glm` target. Could not find glm/glm.hpp")
    endif()
    unset(glm_INCLUDE_DIRS)
endif()

# Mark some CACHE vars advanced for a cleaner GUI
mark_as_advanced(FETCHCONTENT_QUIET)
mark_as_advanced(FETCHCONTENT_BASE_DIR)
mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_SOURCE_DIR_GLM)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_GLM)