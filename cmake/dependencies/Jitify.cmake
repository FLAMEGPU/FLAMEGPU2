##########
# Jitify #
##########

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})
include(FetchContent)
cmake_policy(SET CMP0079 NEW)

# Change the source-dir to allow inclusion via jitify/jitify.hpp rather than jitify.hpp
FetchContent_Declare(
    jitify
    GIT_REPOSITORY https://github.com/NVIDIA/jitify.git
    GIT_TAG        cd6b56bf0c63fcce74a59cd021bf63e5c2a32c73
    SOURCE_DIR     ${FETCHCONTENT_BASE_DIR}/jitify-src/jitify
    GIT_PROGRESS   ON
    # UPDATE_DISCONNECTED   ON
)
FetchContent_GetProperties(jitify)
if(NOT jitify_POPULATED)
    FetchContent_Populate(jitify)
    # Jitify is not a cmake project, so cannot use add_subdirectory, use custom find_package.
endif()

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${jitify_SOURCE_DIR}/..")
# Always find the package, even if jitify is already populated.
find_package(Jitify REQUIRED)
