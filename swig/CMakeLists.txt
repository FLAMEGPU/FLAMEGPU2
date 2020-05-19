# SWIG: use standard target name.
if(POLICY CMP0078)
  cmake_policy(SET CMP0078 NEW)
endif()

# SWIG: use SWIG_MODULE_NAME property.
if(POLICY CMP0086)
  cmake_policy(SET CMP0086 NEW)
endif()

# Pull in swig (on Windows)
MACRO (download_swig)
    configure_file(../cmake/swig/CMakeLists.txt.in ${CMAKE_CURRENT_BINARY_DIR}/swig-download/CMakeLists.txt)
    # Run CMake generate
    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/swig-download
        )
    if (result)
        message(WARNING
                "CMake step for swig failed: ${result}\n")
    endif ()
    # Run CMake build (this only downloads and extracts)
    execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/swig-download
    )
    if (result)
        message(WARNING
                "Download step for swig failed: ${result}\n"
                "Attempting to continue\n")
    endif ()
    # Set Swig path variable
    message(STATUS ${CMAKE_CURRENT_BINARY_DIR}/swig/swig.exe)
    set(SWIG_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/swig/swig.exe CACHE FILEPATH "Path to SWIG executable")
ENDMACRO()
# If on windows, and SWIG_EXECUTABLE not specified, auto download Swig 4.0.1
if(WIN32 AND NOT SWIG_EXECUTABLE)  
    download_swig()
endif()
# Set up swig
set(CMAKE_SWIG_FLAGS)
find_package(SWIG REQUIRED)

if(NOT SWIG_FOUND)
    message("Swig was not found on your machine. Please install it.")
endif()

#include(${SWIG_USE_FILE})
include(UseSWIG)

# Include python
find_package(PythonLibs REQUIRED)
#include_directories(${PYTHON_INCLUDE_PATH})

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Python found at " ${Python3_EXECUTABLE})

if(Python_VERSION VERSION_GREATER_EQUAL 3)
  list(APPEND CMAKE_SWIG_FLAGS "-py3")
endif()


add_subdirectory(python)
