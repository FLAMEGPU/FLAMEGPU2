# Include common rules.
# Actually dont as the c Warning levels are too damn high
#include(../../cmake/common.cmake)
STRING(TOLOWER "${CMAKE_SYSTEM_NAME}" CMAKE_SYSTEM_NAME_LOWER)

# Output locations for build
SET(PYTHON_MODULE_NAME flamegpu)
# TODO: Somehow incorporate $<CONFIG> for multi configuration builds of library
SET(PYTHON_LIB_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${CMAKE_SYSTEM_NAME_LOWER}-x64/python)
SET(PYTHON_LIB_TEMP_DIRECTORY ${CMAKE_BINARY_DIR}/swig/python/${CMAKE_BUILD_TYPE})
# set input file name (TODO: consistency of library names e.g. flamegpu nto flamegpu2)
SET(SWIG_INPUT_FILE_NAME flamegpu)


# setup properties of swig input file
set_property(SOURCE ${SWIG_INPUT_FILE_NAME}.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE ${SWIG_INPUT_FILE_NAME}.i PROPERTY SWIG_MODULE_NAME ${PYTHON_MODULE_NAME})

# Add swig module
swig_add_library(${PYTHON_MODULE_NAME}
	TYPE SHARED
	#TYPE STATIC
	LANGUAGE python 
	OUTPUT_DIR ${PYTHON_LIB_TEMP_DIRECTORY}/${PYTHON_MODULE_NAME}
	SOURCES ${SWIG_INPUT_FILE_NAME}.i)

set_source_files_properties(${PYTHON_MODULE_NAME}/${SWIG_INPUT_FILE_NAME}PYTHON_wrap.cxx PROPERTIES LANGUAGE CUDA)

target_include_directories(${PYTHON_MODULE_NAME}
  PRIVATE
  ../../include
  ../../externals
  ${Python_INCLUDE_DIRS}
  )
  
# use the target include directories
set_property(TARGET ${PYTHON_MODULE_NAME} PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)
# needs to have RDC enable for the project
set_property(TARGET ${PYTHON_MODULE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# link with the static flamegpu and cuda libraries
target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE flamegpu2 cuda)

# Variable PYTHON_LIBRARIES can contains keyword `optimized`
# which won't be interpreted inside a generator expression.
# i.e. we can't use: $<$<PLATFORM_ID:Windows>:${PYTHON_LIBRARIES}>
# see: https://cmake.org/cmake/help/git-stage/command/target_link_libraries.html#command:target_link_libraries
if(MSVC)
  target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE ${Python_LIBRARIES} flamegpu2 cuda)
endif()


#######################
## Python Packaging  ##
#######################
# configure the python setup and init files for packaging
# they contain 
configure_file(
	setup.py.in
	${PYTHON_LIB_TEMP_DIRECTORY}/setup.py.in
	@ONLY)
file(GENERATE
	OUTPUT ${PYTHON_LIB_OUTPUT_DIRECTORY}/setup.py
	INPUT ${PYTHON_LIB_TEMP_DIRECTORY}/setup.py.in)
	
configure_file(
	__init__.py.in
	${PYTHON_LIB_TEMP_DIRECTORY}/__init__.py.in
	@ONLY)
file(GENERATE
	OUTPUT ${PYTHON_LIB_OUTPUT_DIRECTORY}/${PYTHON_MODULE_NAME}/__init__.py
	INPUT ${PYTHON_LIB_TEMP_DIRECTORY}/__init__.py.in)

# Find if python module MODULE_NAME is available,
# if not install it to the Python user install directory.
function(search_python_module MODULE_NAME)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import ${MODULE_NAME}; print(${MODULE_NAME}.__version__)"
    RESULT_VARIABLE _RESULT
    OUTPUT_VARIABLE MODULE_VERSION
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  if(${_RESULT} STREQUAL "0")
    message(STATUS "Found python module: ${MODULE_NAME} (found version \"${MODULE_VERSION}\")")
  else()
    message(WARNING "Can't find python module \"${MODULE_NAME}\", user install it using pip...")
    execute_process(
      COMMAND ${Python_EXECUTABLE} -m pip install --upgrade --user ${MODULE_NAME}
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
  endif()
endfunction()

# Look for required python modules
search_python_module(setuptools)
search_python_module(wheel)

add_custom_target(python_package ALL
  #COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/swig/python/$<CONFIG>/setup.py setup.py
  #COMMAND ${CMAKE_COMMAND} -E copy __init__.py.in ${PROJECT_NAME}/__init__.py
  #COMMAND ${CMAKE_COMMAND} -E copy __init__.py.in ${PYTHON_MODULE_NAME}/${PYTHON_MODULE_NAME}/__init__.py
  #COMMAND ${CMAKE_COMMAND} -E remove_directory dist
  # copy generated python file to library output location
  COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_LIB_TEMP_DIRECTORY}/${PYTHON_MODULE_NAME}/${PYTHON_MODULE_NAME}.py ${PYTHON_MODULE_NAME}
  # copy compiled pyd file to library output location
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PYTHON_MODULE_NAME}> ${PYTHON_MODULE_NAME}
  # copy module libary to .libs
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_MODULE_NAME}/.libs
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${SWIG_INPUT_FILE_NAME}> ${PYTHON_MODULE_NAME}/.libs

  COMMAND ${Python_EXECUTABLE} setup.py bdist_wheel
  BYPRODUCTS
	  #${PROJECT_NAME}
	  build
	  dist
	  ${PYTHON_MODULE_NAME}.egg-info
  WORKING_DIRECTORY ${PYTHON_LIB_OUTPUT_DIRECTORY}
)


# Test
if (FALSE)
	if(BUILD_SWIG_PYTHON_TESTS)
	  # Look for python module virtualenv
	  search_python_module(virtualenv)
	  # Testing using a virtual environment
	  set(VENV_EXECUTABLE ${Python_EXECUTABLE} -m virtualenv)
	  set(VENV_DIR ${CMAKE_CURRENT_BINARY_DIR}/venv)
	  if(WIN32)
		set(VENV_Python_EXECUTABLE "${VENV_DIR}\\Scripts\\python.exe")
	  else()
		set(VENV_Python_EXECUTABLE ${VENV_DIR}/bin/python)
	  endif()
	  # make a virtualenv to install our python package in it
	  add_custom_command(TARGET python_package POST_BUILD
		COMMAND ${VENV_EXECUTABLE} -p ${Python_EXECUTABLE} ${VENV_DIR}
		# Must not call it in a folder containing the setup.py otherwise pip call it
		# (i.e. "python setup.py bdist") while we want to consume the wheel package
		COMMAND ${VENV_Python_EXECUTABLE} -m pip install --force-reinstall --find-links=${PROJECT_SOURCE_DIR}/swig/python/dist ${PROJECT_NAME}
		BYPRODUCTS ${VENV_DIR}
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
	  # run the tests within the virtualenv
	  add_test(NAME pytest_venv
		COMMAND ${VENV_Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/swig/python/test.py)
	endif()
endif()

# set project folders (for better organisation in visual studio)
if (CMAKE_USE_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(TARGET "${PYTHON_MODULE_NAME}" PROPERTY FOLDER "FLAMEGPU")
  set_property(TARGET "python_package" PROPERTY FOLDER "Python")
endif ()