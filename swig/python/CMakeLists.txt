set_property(SOURCE flamegpu2.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE flamegpu2.i PROPERTY SWIG_MODULE_NAME pyFlameGPU)

# Add swig module
swig_add_library(pyFlameGPU
	TYPE STATIC
	LANGUAGE python 
	OUTPUT_DIR ${CMAKE_BINARY_DIR}/python/${PROJECT_NAME}
	SOURCES flamegpu2.i)


target_include_directories(pyFlameGPU
  PRIVATE
  ../../include
  ../../externals
  ${Python_INCLUDE_DIRS}
  )
set_property(TARGET pyFlameGPU PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)

target_link_libraries(pyFlameGPU PRIVATE flamegpu2)

# Variable PYTHON_LIBRARIES can contains keyword `optimized`
# which won't be interpreted inside a generator expression.
# i.e. we can't use: $<$<PLATFORM_ID:Windows>:${PYTHON_LIBRARIES}>
# see: https://cmake.org/cmake/help/git-stage/command/target_link_libraries.html#command:target_link_libraries
if(MSVC)
  target_link_libraries(pyFlameGPU PRIVATE ${Python_LIBRARIES})
endif()


# TODO: Python packaging