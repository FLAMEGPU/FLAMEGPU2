# Minimum CMake version 3.18 for CUDA --std=c++17 
cmake_minimum_required(VERSION 3.18...3.25 FATAL_ERROR)
include(FetchContent)
# Temporary CMake >= 3.30 fix https://github.com/FLAMEGPU/FLAMEGPU2/issues/1223
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

# Set some policy behaviours for SWIG

# SWIG: use standard target name.
if(POLICY CMP0078)
  cmake_policy(SET CMP0078 NEW)
endif()

# SWIG: use SWIG_MODULE_NAME property.
if(POLICY CMP0086)
  cmake_policy(SET CMP0086 NEW)
endif()

# As the URL method is used for download, set the policy if available
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# Find Python3, optionally an exact version for windows CI, or use in manylinux
if(PYTHON3_EXACT_VERSION)
    set(PYTHON3_EXACT_VERSION_ARG ${PYTHON3_EXACT_VERSION} EXACT)
endif()
# CMake > 3.18 can use Development.Module rather than development
find_package(Python3 ${PYTHON3_EXACT_VERSION_ARG} REQUIRED COMPONENTS Interpreter Development.Module)
unset(PYTHON3_EXACT_VERSION_ARG)
message(STATUS "Python found at " ${Python3_EXECUTABLE})

# Define the minimum and download versions of SWIG which FLAME GPU supports/ requires as cache variables so they may be overridden by users/CI workflows.
# Swig >= 4 is required for C++17
# Swig 4.2.0 is very broken for FLAME GPU 2
# Swig 4.2.1 does not allow != / == comparisons against None
# Swig 4.3.0 is potentially required for Python 3.13 support 
set(FLAMEGPU_SWIG_MINIMUM 4.0.2 CACHE STRING "Minimum version of SWIG to search for via CMake find_package")
mark_as_advanced(FLAMEGPU_SWIG_MINIMUM)
set(FLAMEGPU_SWIG_DOWNLOAD 4.0.2 CACHE STRING "Version of SWIG to download if the minimum required SWIG is not found")
mark_as_advanced(FLAMEGPU_SWIG_DOWNLOAD)
# Use an option to control if minimum or exact swig should be found.
option(FLAMEGPU_SWIG_EXACT "Require the exact version of SWIG specified in FLAMEGPU_SWIG_MINIMUM. This enables downgrading swig." OFF)
mark_as_advanced(FLAMEGPU_SWIG_EXACT)

# If the download version is less than the minimum, set it to the minimum and issue a warning.
if(${FLAMEGPU_SWIG_DOWNLOAD} VERSION_LESS ${FLAMEGPU_SWIG_MINIMUM})
    message(WARNING "FLAMEGPU_SWIG_DOWNLOAD ${FLAMEGPU_SWIG_DOWNLOAD} less than FLAMEGPU_SWIG_MINIMUM ${FLAMEGPU_SWIG_MINIMUM}. Using FLAMEGPU_SWIG_DOWNLOAD=${FLAMEGPU_SWIG_MINIMUM}.")
    # set(FLAMEGPU_SWIG_DOWNLOAD ${FLAMEGPU_SWIG_MINIMUM})
    set(FLAMEGPU_SWIG_DOWNLOAD ${FLAMEGPU_SWIG_MINIMUM} CACHE STRING "Minimum version of SWIG to search for via CMake find_package" FORCE)
endif()

# Look for user-installed SWIG in expectd locations.
# Users can set -DSWIG_EXECUTABLE pointed at their version of SWIG if it cannot be found (@todo - document this)
# If swig is downloaded and installed by cmake, the second invocation will contain cache variables which correctly correspond to cmake.

if(NOT ${FLAMEGPU_SWIG_EXACT})
    find_package(SWIG ${FLAMEGPU_SWIG_MINIMUM})
else()
    find_package(SWIG ${FLAMEGPU_SWIG_MINIMUM} EXACT)
    if(NOT ${FLAMEGPU_SWIG_DOWNLOAD} VERSION_EQUAL ${FLAMEGPU_SWIG_MINIMUM})
        message(WARNING "FLAMEGPU_SWIG_DOWNLOAD ${FLAMEGPU_SWIG_DOWNLOAD} != FLAMEGPU_SWIG_MINIMUM ${FLAMEGPU_SWIG_MINIMUM} with FLAMEGPU_SWIG_EXACT=${FLAMEGPU_SWIG_EXACT}. Using FLAMEGPU_SWIG_DOWNLOAD=${FLAMEGPU_SWIG_MINIMUM}.")
        # set(FLAMEGPU_SWIG_DOWNLOAD ${FLAMEGPU_SWIG_MINIMUM})
        set(FLAMEGPU_SWIG_DOWNLOAD ${FLAMEGPU_SWIG_MINIMUM} CACHE STRING "Minimum version of SWIG to search for via CMake find_package" FORCE)
    endif()
endif()

# If minimum required swig could not be found, download and build it.
if(NOT SWIG_FOUND)
    # SWIG_DIR and SWIG_EXECUTABLE are used by FindSwig. Executable specifies version info etc, swig_dir should contain swig.swg (i.e the value of swig -swiglib so doesn't need specifying if teh swig install is good?)

    # Unset swig related variables from the previous failed attempt at finding swig
    unset(SWIG_FOUND)
    unset(SWIG_VERSION)
    unset(SWIG_DIR)
    unset(SWIG_EXECUTABLE)
    unset(SWIG_FOUND CACHE)
    unset(SWIG_VERSION CACHE)
    unset(SWIG_DIR CACHE)
    unset(SWIG_EXECUTABLE CACHE)


    if(WIN32)
        # Download pre-compiled swig on windows.
        FetchContent_Declare(
            swig
            URL      http://prdownloads.sourceforge.net/swig/swigwin-${FLAMEGPU_SWIG_DOWNLOAD}.zip
            # URL_HASH #@todo - make sure the hash of the download is good?
        )
        FetchContent_GetProperties(swig)
        if(NOT swig_POPULATED)
            message(STATUS "[swig] Downloading swigwin-${FLAMEGPU_SWIG_DOWNLOAD}.zip")
            # Download
            FetchContent_Populate(swig)
            
            # Set variables used by find_swig to find swig.
            # Set locally and in the cache for subsequent invocations of CMake
            set(SWIG_EXECUTABLE "${swig_SOURCE_DIR}/swig.exe")
            set(SWIG_EXECUTABLE "${swig_SOURCE_DIR}/swig.exe" CACHE FILEPATH "Path to SWIG executable")
        endif()
    else()
        # Under linux, download the .tar.gz, extract, build and install.
        # This must be done at configure time, as FindSwig requires the swig executable.
        # FetchContent allows download at configure time, but must use execute_process to run commands at configure time.

        # Download from sourceforge not github, github releases require several extra tools to build (bison, yacc, autoconfigure, automake), and we want to keep dependencies light. 
        FetchContent_Declare(
            swig
            URL      https://downloads.sourceforge.net/project/swig/swig/swig-${FLAMEGPU_SWIG_DOWNLOAD}/swig-${FLAMEGPU_SWIG_DOWNLOAD}.tar.gz
            # URL_HASH #@todo - make sure the hash of the download is good?
        )
        FetchContent_GetProperties(swig)
        if(NOT swig_POPULATED)
            message(STATUS "[swig] Downloading swig-${FLAMEGPU_SWIG_DOWNLOAD}.tar.gz")
            # Download the content
            FetchContent_Populate(swig)

            # Set some paths for error files in case things go wrong.
            set(swig_configure_ERROR_FILE "${swig_BINARY_DIR}/swig-error-configue.log")
            set(swig_make_ERROR_FILE "${swig_BINARY_DIR}/swig-error-make.log")
            set(swig_makeinstall_ERROR_FILE "${swig_BINARY_DIR}/swig-error-make-install.log")

            # run ./configure with an appropraite prefix to install into the _deps/swig-bin directory
            message(STATUS "[swig] ./configure --prefix ${swig_BINARY_DIR}")
            execute_process(
                COMMAND "./configure" "--prefix" "${swig_BINARY_DIR}"
                WORKING_DIRECTORY ${swig_SOURCE_DIR}
                RESULT_VARIABLE swig_configure_RESULT
                OUTPUT_VARIABLE swig_configure_OUTPUT
                ERROR_FILE ${swig_configure_ERROR_FILE}
            )
            if(NOT swig_configure_RESULT EQUAL "0")
                message(FATAL_ERROR 
                " [swig] Error configuring SWIG ${FLAMEGPU_SWIG_DOWNLOAD}.\n"
                " Error log: ${swig_configure_ERROR_FILE}\n"
                " Consider installing SWIG ${FLAMEGPU_SWIG_MINIMUM} yourself and passing -DSWIG_EXECUTABLE=/path/to/swig.")
            endif()

            # run make to compile swig within swig_SOURCE_DIR
            message(STATUS "[swig] make")
            execute_process(
                COMMAND "make"
                WORKING_DIRECTORY ${swig_SOURCE_DIR}
                RESULT_VARIABLE swig_make_RESULT
                OUTPUT_VARIABLE swig_make_OUTPUT
            )
            if(NOT swig_make_RESULT EQUAL "0")
                message(FATAL_ERROR 
                " [swig] Error compiling SWIG ${FLAMEGPU_SWIG_DOWNLOAD}\n"
                " Error log: ${swig_make_ERROR_FILE}"
                " Consider installing SWIG >= ${FLAMEGPU_SWIG_MINIMUM} yourself and passing -DSWIG_EXECUTABLE=/path/to/swig.")
            endif()

            # run make install to copy to the installation location (swig_BINARY_DIR)
            message(STATUS "[swig] make install")
            execute_process(
                COMMAND "make" "install"
                WORKING_DIRECTORY ${swig_SOURCE_DIR}
                RESULT_VARIABLE swig_makeinstall_RESULT
                OUTPUT_VARIABLE swig_makeinstall_OUTPUT
            )
            if(NOT swig_makeinstall_RESULT EQUAL "0")
                message(FATAL_ERROR 
                " [swig] Error installing SWIG ${FLAMEGPU_SWIG_DOWNLOAD}\n"
                " Error log: ${swig_makeinstall_ERROR_FILE}"
                " Consider installing SWIG >= ${FLAMEGPU_SWIG_MINIMUM} yourself and passing -DSWIG_EXECUTABLE=/path/to/swig.")
            endif()

            # Set variables used by find_swig to find swig.
            # Set locally and in the cache for subsequent invocations of CMake
            set(SWIG_EXECUTABLE "${swig_BINARY_DIR}/bin/swig")
            set(SWIG_EXECUTABLE "${swig_BINARY_DIR}/bin/swig" CACHE FILEPATH "Path to SWIG executable")
        endif()
    endif()
    # Attempt to find swig again, but as REQUIRED using the downloaded version
    find_package(SWIG ${FLAMEGPU_SWIG_MINIMUM} REQUIRED)
endif()

# If a bad swig has been found, or downloaded, warn about it. 
if(SWIG_FOUND AND SWIG_VERSION VERSION_EQUAL "4.2.0")
    # https://github.com/FLAMEGPU/FLAMEGPU2/issues/956#issuecomment-2069104478
    # https://github.com/swig/swig/commit/c7ab6a01c6582b92db9328e2f3daa67081f05f6e
    message(WARNING "Swig ${SWIG_VERSION} is likely to error during pyflamegpu compilation, due to a bug with friend constructor/deconstructor declarations. please use an alternate version of SWIG.")
elseif(SWIG_FOUND AND SWIG_VERSION VERSION_EQUAL "4.2.1")
    # https://github.com/swig/swig/issues/2987
    message(WARNING "Swig ${SWIG_VERSION} produces usable pyflamegpu, but swig-wrapped objects cannot be compared to None using == or !=, you must use is / is not. Consider using an alternate version of SWIG.")
endif()

# Mark some CACHE vars advanced for a cleaner GUI
mark_as_advanced(FETCHCONTENT_SOURCE_DIR_SWIG)
mark_as_advanced(FETCHCONTENT_QUIET)
mark_as_advanced(FETCHCONTENT_BASE_DIR)
mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED) 
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_SWIG) 

# Set flags for SWIG.
set(CMAKE_SWIG_FLAGS)

#include(${SWIG_USE_FILE})
include(UseSWIG)

# enable doxygen translation
if (SWIG_VERSION VERSION_GREATER_EQUAL 4)
	list(APPEND CMAKE_SWIG_FLAGS "-doxygen")
endif()

if(Python_VERSION VERSION_GREATER_EQUAL 3)
	list(APPEND CMAKE_SWIG_FLAGS "-py3")
endif()

# If on a system with 64bit longs, need to pass -DSWIGWORDSIZE64 to swig.
if(NOT WIN32)
    list(APPEND CMAKE_SWIG_FLAGS "-DSWIGWORDSIZE64")
endif()

# Add the python swig subdirectory
add_subdirectory(python)
