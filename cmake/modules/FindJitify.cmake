# CMake module to find Jitify headers/library
# 
# Very basic.
#
# Usage:
#    find_package( Jitify )
#    if(Jitify_FOUND)
#        include_directories(${Jitify_INCLUDE_DIRS})
#    endif()
#
# Variables:
#    Jitify_FOUND
#    Jitify_INCLUDE_DIRS
#    Jitify_VERSION
#
# Manually specify Jitify paths via -DJitify_ROOT=/path/to/Jitify

include(FindPackageHandleStandardArgs)

# Find the main Jitify header
find_path(Jitify_INCLUDE_DIRS
    NAMES
        jitify/jitify.hpp
)

# if found, get the version number.
if(Jitify_INCLUDE_DIRS)
    # Get the Jitify version from the headerfile, by reading it and regex matching
    file(READ "${Jitify_INCLUDE_DIRS}/jitify/jitify.hpp" Jitify_header_text)
    string(REGEX MATCH "Jitify ([0-9]+\.[0-9]+(\.[0-9]+)?)" Jitify_VERSION_DEFINE ${Jitify_header_text})
    set(Jitify_VERSION "${CMAKE_MATCH_1}")
endif()
# Apply standard cmake find package rules / variables. I.e. QUIET, Jitify_FOUND etc.
# Outside the if, so REQUIRED works.
find_package_handle_standard_args(Jitify
    REQUIRED_VARS Jitify_INCLUDE_DIRS
    VERSION_VAR Jitify_VERSION
)

# Create a header only (INTERFACE) target which can be linked against to upadte include directories. Mark as IMPORTED as there are no build steps.
add_library(Jitify::jitify INTERFACE IMPORTED)
target_include_directories(Jitify::jitify INTERFACE ${Jitify_INCLUDE_DIRS})

# Set returned values as advanced?
mark_as_advanced(Jitify_INCLUDE_DIRS Jitify_VERSION)
