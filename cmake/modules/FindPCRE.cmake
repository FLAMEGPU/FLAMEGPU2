# CMake module to find PCRE headers & library
# 
# Note:
#    This is for the older PCRE interface (i.e. not PCRE2). However this is named libpcre3 on ubuntu/debian...
#
# Usage:
#    find_package( PCRE )
#    if(PCRE_FOUND)
#        include_directories(${PCRE_INCLUDE_DIRS})
#        target_link_libraries(target ${PCRE_LIBRARIES})
#    endif()
#
# Variables:
#    PCRE_FOUND
#    PCRE_INCLUDE_DIRS
#    PCRE_LIBRARIES
#    PCRE_VERSION
#    PCRE_CONFIG

# Search for pcre.h in standard include paths
find_path(PCRE_INCLUDE_DIRS
    NAMES 
        pcre.h
)

# Find the various pcre libraries
find_library(PCRE_LIBRARY_8 NAMES pcre libpcre)
find_library(PCRE_LIBRARY_16 NAMES pcre16 libpcre16 )
find_library(PCRE_LIBRARY_32 NAMES pcre32 libpcre32 )
find_library(PCRE_LIBRARY_POSIX NAMES pcreposix libpcreposix )

# Combine found libraries into a single list (adjust based on what your project needs)
set(PCRE_LIBRARIES)
if (PCRE_LIBRARY_8)
    list(APPEND PCRE_LIBRARIES ${PCRE_LIBRARY_8})
endif()
if (PCRE_LIBRARY_16)
    list(APPEND PCRE_LIBRARIES ${PCRE_LIBRARY_16})
endif()
if (PCRE_LIBRARY_32)
    list(APPEND PCRE_LIBRARIES ${PCRE_LIBRARY_32})
endif()
if (PCRE_LIBRARY_POSIX)
    list(APPEND PCRE_LIBRARIES ${PCRE_LIBRARY_POSIX})
endif()

# Find pcre-config
find_program(PCRE_CONFIG pcre-config)

# Extract the version number, which is defiend separately as PCRE_MAJOR and PCRE_MINOR (and PCRE_PRERELEASE, but not checking the prerelease component)
set(PCRE_VERSION "unknown")
if (PCRE_INCLUDE_DIRS AND EXISTS "${PCRE_INCLUDE_DIRS}/pcre.h")
    file(READ "${PCRE_INCLUDE_DIRS}/pcre.h" pcre_header_text)
    string(REGEX MATCH "define PCRE_MAJOR +([0-9]+)" PCRE_MAJOR_DEFINE ${pcre_header_text})
    set(PCRE_MAJOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define PCRE_MINOR +([0-9]+)" PCRE_MINOR_DEFINE ${pcre_header_text})
    set(PCRE_MINOR "${CMAKE_MATCH_1}")
    if (NOT "${PCRE_MAJOR_DEFINE}" STREQUAL "" AND NOT "${PCRE_MAJOR_DEFINE}" STREQUAL "")
        set(PCRE_VERSION "${PCRE_MAJOR}.${PCRE_MINOR}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PCRE
    REQUIRED_VARS
        PCRE_LIBRARIES PCRE_INCLUDE_DIRS PCRE_CONFIG
    VERSION_VAR
        PCRE_VERSION
)


mark_as_advanced(PCRE_INCLUDE_DIRS PCRE_LIBRARY_8 PCRE_LIBRARY_16 PCRE_LIBRARY_32 PCRE_LIBRARY_POSIX PCRE_LIBRARIES PCRE_VERSION)

# Unset temporary variables
unset(PCRE_MAJOR)
unset(PCRE_MINOR)

# If we wanted to actually depend on pcre, we could create an imported library target here.
# However there is no need currently as we are just checking for pcre presence to emit a warning if building swig from source and swig compilation failed.
