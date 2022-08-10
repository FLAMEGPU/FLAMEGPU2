#######
# glm #
#######

# As the URL method is used for download, set the policy if available
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})
include(FetchContent)

# Head of master at point BugFix for NVRTC support was merged
FetchContent_Declare(
    glm
    URL            "https://github.com/g-truc/glm/archive/66062497b104ca7c297321bd0e970869b1e6ece5.zip"
)
FetchContent_GetProperties(glm)
if(NOT glm_POPULATED)
    FetchContent_Populate(glm)
    # glm CMake wants to generate the find file in a system location, so handle it manually
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${glm_SOURCE_DIR}")
endif()
if (NOT glm_FOUND)
    find_package(glm REQUIRED)
    # Include path is ${glm_INCLUDE_DIRS}
endif()