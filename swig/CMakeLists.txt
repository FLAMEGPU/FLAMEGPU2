# SWIG: use standard target name.
if(POLICY CMP0078)
  cmake_policy(SET CMP0078 NEW)
endif()

# SWIG: use SWIG_MODULE_NAME property.
if(POLICY CMP0086)
  cmake_policy(SET CMP0086 NEW)
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
#find_package(PythonLibs REQUIRED)
#include_directories(${PYTHON_INCLUDE_PATH})

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development)

if(Python_VERSION VERSION_GREATER_EQUAL 3)
  list(APPEND CMAKE_SWIG_FLAGS "-py3")
endif()


add_subdirectory(python)
