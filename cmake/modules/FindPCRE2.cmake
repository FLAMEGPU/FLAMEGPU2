# CMake module to find PCRE2 headers & library
#
# Usage:
#    find_package( PCRE2 )
#    if(PCRE2_FOUND)
#        include_directories(${PCRE2_INCLUDE_DIRS})
#        target_link_libraries(target ${PCRE2_LIBRARIES})
#    endif()
#
# Variables:
#    PCRE2_FOUND
#    PCRE2_INCLUDE_DIRS
#    PCRE2_LIBRARIES
#    PCRE2_VERSION
#    PCRE2_CONFIG

# Search for pcre2.h in standard include paths
find_path(PCRE2_INCLUDE_DIRS
    NAMES 
        pcre2.h
)

# Find the various pcre libraries
find_library(PCRE2_LIBRARY_8 NAMES pcre2-8 libpcre2-8)
find_library(PCRE2_LIBRARY_16 NAMES pcre2-16 libpcre2-16 )
find_library(PCRE2_LIBRARY_32 NAMES pcre2-32 libpcre2-32 )
find_library(PCRE2_LIBRARY_POSIX NAMES pcre2-posix libpcre2-posix )

# Combine found libraries into a single list (adjust based on what your project needs)
set(PCRE2_LIBRARIES)
if (PCRE2_LIBRARY_8)
    list(APPEND PCRE2_LIBRARIES ${PCRE2_LIBRARY_8})
endif()
if (PCRE2_LIBRARY_16)
    list(APPEND PCRE2_LIBRARIES ${PCRE2_LIBRARY_16})
endif()
if (PCRE2_LIBRARY_32)
    list(APPEND PCRE2_LIBRARIES ${PCRE2_LIBRARY_32})
endif()
if (PCRE2_LIBRARY_POSIX)
    list(APPEND PCRE2_LIBRARIES ${PCRE2_LIBRARY_POSIX})
endif()

# Find pcre2-config
find_program(PCRE2_CONFIG pcre2-config)

# Extract the version number, which is defiend separately as PCRE2_MAJOR and PCRE2_MINOR (and PCRE2_PRERELEASE, but not checking the prerelease component)
if (PCRE2_INCLUDE_DIRS AND EXISTS "${PCRE2_INCLUDE_DIRS}/pcre2.h")
    file(READ "${PCRE2_INCLUDE_DIRS}/pcre2.h" pcre2_header_text)
    string(REGEX MATCH "define PCRE2_MAJOR +([0-9]+)" PCRE2_MAJOR_DEFINE ${pcre2_header_text})
    set(PCRE2_MAJOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define PCRE2_MINOR +([0-9]+)" PCRE2_MINOR_DEFINE ${pcre2_header_text})
    set(PCRE2_MINOR "${CMAKE_MATCH_1}")
    if (NOT "${PCRE2_MAJOR_DEFINE}" STREQUAL "" AND NOT "${PCRE2_MAJOR_DEFINE}" STREQUAL "")
        set(PCRE2_VERSION "${PCRE2_MAJOR}.${PCRE2_MINOR}")
    else()
        set(PCRE2_VERSION "unknown")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PCRE2
    REQUIRED_VARS
        PCRE2_LIBRARIES PCRE2_INCLUDE_DIRS PCRE2_CONFIG
    VERSION_VAR
        PCRE2_VERSION
)

mark_as_advanced(PCRE2_INCLUDE_DIRS PCRE2_LIBRARY_8 PCRE2_LIBRARY_16 PCRE2_LIBRARY_32 PCRE2_LIBRARY_POSIX PCRE2_LIBRARIES PCRE2_VERSION)

# Unset temporary variables
unset(PCRE2_MAJOR)
unset(PCRE2_MINOR)

# If we wanted to actually depend on pcre2, we could create an imported library target here.
# However there is no need currently as we are just checking for pcre2 presence to emit a warning if building swig from source and swig compilation failed.
