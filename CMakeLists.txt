### CMakeLists.txt for FLAME GPU

################################
# mkdir build
# cd build
# cmake ..

# cmake --build .
# or
# make BOOST_TEST
# or
# make test
################################

cmake_minimum_required(VERSION VERSION 3.8 FATAL_ERROR)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(FLAMEGPU LANGUAGES CXX CUDA)

include(CTest)

# Find cuda package
find_package(CUDA QUIET REQUIRED)

if (CUDA_FOUND)
   message("CUDA found")
   #add_subdirectory(src.gpu)
else()
   message(STATUS "CUDA not found")
endif()

#################################
set(CUDA_SEPARABLE_COMPILATION ON)

include_directories(
        /usr/local/cuda-9.1/include/
)

link_directories(
        /usr/local/cuda-9.1/lib64/
)

#################################
# if statament here should be treated as comment, may be needed later
if (FALSE)
# Debug build flags
if (dbg EQUAL 1)
    set(NVCCFLAGS "-g -G")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS "-std=c++11  -std=c++14 -lineinfo")

#add_definitions(-rdc=true)

# Pass options to NVCC, Append current NVCC flags by something
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -m64
    $(NVCCFLAGS)
    -gencode arch=compute_50,code=sm_50
    )

set(IDIR 
        *.h  ##??
        gpu/
        pop/
        sim/
        model/
        runtime/
        runtime/cuRVE/
        )

if(WIN32)

endif()
if(UNIX)
    set(INC_DIR
            ${INC_DIR}
            ../../lib/x86_64-linux-gnu/ ## ??
            /usr/include/boost
            /usr/local/cuda/include

            )
endif()

include_directories(${INC_DIR})



# Specify include directories
include_directories(
        $(IDIR)
    )
#include_directories(${GPU_C_FILES} ${GPU_CU_FILES} ${SIM_C_FILES} ${POP_C_FILES} ${MODEL_C_FILES} ${RUNTIME_CU_FILES} ${CURVE_CU_FILES})

# Specify library paths
if(WIN32)
    link_directories(../ext/FLAMEGPU/lib )
endif()
if(UNIX)
    link_directories(/usr/local/lib /usr/lib /usr/lib64)
endif()
endif()

#################################

# The file(GLOB...) allows for wildcard additions (instead of manually add the sources using set command)
FILE(GLOB C_SOURCES "gpu/*.cpp" "sim/*.cpp" "pop/*.cpp" "model/*.cpp")
FILE(GLOB CU_SOURCES "gpu/*.cu" "runtime/*.cu" "runtime/cuRVE/*.cu")


# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(FGPU ${C_SOURCES} ${CU_SOURCES} main.cu)
cuda_add_executable(FGPU_MAS ${C_SOURCES} ${CU_SOURCES} main_MAS.cu )

# Request that FLAMEGPU be built with -std=c++11
# As this is a public compile feature anything that links to the executable will also build with -std=c++11 and 14
target_compile_features(FGPU PUBLIC cxx_std_11 cxx_std_14)
target_compile_features(FGPU_MAS PUBLIC cxx_std_11 cxx_std_14)


# We need to explicitly state that we need all CUDA files in the FLAMEGPU
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( FGPU 
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties( FGPU_MAS
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
                       

# For linking ...
# Specify target & libraries to link it with
#target_link_libraries( FGPU -rdc=true)
    
if(BUILD_TESTING)
  add_executable(BOOST_TEST ${C_SOURCES} ${CU_SOURCES} 
		    tests/test_all.cpp 
		    tests/device_functions.cu)

  set_target_properties(BOOST_TEST PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 # target_link_libraries(BOOST_TEST PRIVATE model sim pop message gpu)

  add_test(NAME test_all COMMAND BOOST_TEST --log_level=message)

  add_test(NAME test_model COMMAND BOOST_TEST --log_level=test_suite --run_test=modelDescTest )
  add_test(NAME test_population COMMAND BOOST_TEST --log_level=test_suite --run_test=PopTest )
  add_test(NAME test_messages COMMAND BOOST_TEST --log_level=test_suite --run_test=MessageTest )
  add_test(NAME test_simulation COMMAND BOOST_TEST --log_level=test_suite --run_test=SimTest )
  add_test(NAME test_gpu COMMAND BOOST_TEST --log_level=test_suite --run_test=GPUTest )

endif()

