# CMake module to find glm headers/library
# 
# Very basic.
#
# Usage:
#    find_package( glm )
#    if(glm_FOUND)
#        include_directories(${glm_INCLUDE_DIRS})
#    endif()
#
# Variables:
#    glm_FOUND
#    glm_INCLUDE_DIRS
#    glm_VERSION
#
# Manually specify glm paths via -Dglm_ROOT=/path/to/glm

include(FindPackageHandleStandardArgs)

# Find the main Jitify header
find_path(glm_INCLUDE_DIRS
    NAMES
        glm/glm.hpp
)

# if found, get the version number.
if(glm_INCLUDE_DIRS)
    # glm nolonger has official releases, so there isn't a way to detect a version
    set(glm_VERSION "VERSION_UNKNOWN")
endif()
# Apply standard cmake find package rules / variables. I.e. QUIET, glm_FOUND etc.
# Outside the if, so REQUIRED works.
find_package_handle_standard_args(glm
    REQUIRED_VARS glm_INCLUDE_DIRS
    VERSION_VAR glm_VERSION
)
if(NOT TARGET GLM::glm)
    # Create a header only (INTERFACE) target which can be linked against to inherit include directories. Mark this as imported, because there are no build steps requred.
    add_library(GLM::glm INTERFACE IMPORTED)
    target_include_directories(GLM::glm INTERFACE ${glm_INCLUDE_DIRS})
endif()
# Set returned values as advanced?
mark_as_advanced(glm_INCLUDE_DIRS glm_VERSION)
