# Minimum CMake version 3.18 for CUDA --std=c++17 
cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)
include(FetchContent)


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

# Define the minimum version of cmake we support.
# Swig 4.0+ is required for c++14 (and c++11 std::unordered_map). 
set(SWIG_MINIMUM_SUPPORTED_VERSION 4.0.2)
# Define the version of SWIG to be downloaded (to simplify upgrading / output.)
set(SWIG_DOWNLOAD_VERSION 4.0.2)

# Look for user-installed SWIG in expectd locations.
# Users can set -DSWIG_EXECUTABLE pointed at their version of SWIG if it cannot be found (@todo - document this)
# If swig is downloaded and installed by cmake, the second invocation will contain cache variables which correctly correspond to cmake.

find_package(SWIG ${SWIG_MINIMUM_SUPPORTED_VERSION})

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
            URL      http://prdownloads.sourceforge.net/swig/swigwin-${SWIG_DOWNLOAD_VERSION}.zip
            # URL_HASH #@todo - make sure the hash of the download is good?
        )
        FetchContent_GetProperties(swig)
        if(NOT swig_POPULATED)
            message(STATUS "[swig] Downloading swigwin-${SWIG_DOWNLOAD_VERSION}.zip")
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
            URL      https://downloads.sourceforge.net/project/swig/swig/swig-${SWIG_DOWNLOAD_VERSION}/swig-${SWIG_DOWNLOAD_VERSION}.tar.gz
            # URL_HASH #@todo - make sure the hash of the download is good?
        )
        FetchContent_GetProperties(swig)
        if(NOT swig_POPULATED)
            message(STATUS "[swig] Downloading swig-${SWIG_DOWNLOAD_VERSION}.tar.gz")
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
                " [swig] Error configuring SWIG ${SWIG_DOWNLOAD_VERSION}.\n"
                " Error log: ${swig_configure_ERROR_FILE}\n"
                " Consider installing SWIG ${SWIG_MINIMUM_SUPPORTED_VERSION} yourself and passing -DSWIG_EXECUTABLE=/path/to/swig.")
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
                " [swig] Error compiling SWIG ${SWIG_DOWNLOAD_VERSION}\n"
                " Error log: ${swig_make_ERROR_FILE}"
                " Consider installing SWIG >= ${SWIG_MINIMUM_SUPPORTED_VERSION} yourself and passing -DSWIG_EXECUTABLE=/path/to/swig.")
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
                " [swig] Error installing SWIG ${SWIG_DOWNLOAD_VERSION}\n"
                " Error log: ${swig_makeinstall_ERROR_FILE}"
                " Consider installing SWIG >= ${SWIG_MINIMUM_SUPPORTED_VERSION} yourself and passing -DSWIG_EXECUTABLE=/path/to/swig.")
            endif()

            # Set variables used by find_swig to find swig.
            # Set locally and in the cache for subsequent invocations of CMake
            set(SWIG_EXECUTABLE "${swig_BINARY_DIR}/bin/swig")
            set(SWIG_EXECUTABLE "${swig_BINARY_DIR}/bin/swig" CACHE FILEPATH "Path to SWIG executable")
        endif()
    endif()
    # Attempt to find swig again, but as REQUIRED.
    find_package(SWIG ${SWIG_MINIMUM_SUPPORTED_VERSION} REQUIRED)
endif()

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
