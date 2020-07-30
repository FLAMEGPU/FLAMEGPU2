# 3.10 required for cuda -std=c++14, however 3.12 fixes some device linker errors
cmake_minimum_required(VERSION VERSION 3.12 FATAL_ERROR)

# Cmake 3.16 has issues with CUB and Thrust due to the order of the include path it generates.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16 AND CMAKE_VERSION VERSION_LESS 3.17)
    message(WARNING
    " CMake 3.16 does not use the correct CUDA include paths on some platforms.\n"
    " Please consider using a different version of CMake.")
endif()

# Set the location of the ROOT flame gpu project relative to this CMakeList.txt
get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. REALPATH)

# See what languages are supported
# This is primarily here incase someone targets src/CMakeFiles.txt directly
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER STREQUAL NOTFOUND)
    project(flamegpu2 NONE)
    #Not able to build code, so just make docs
    message("Suitable compilers for building code not found.\n" "Attempting generation of minimal documentation only project.")
    include(${FLAMEGPU_ROOT}/cmake/doxygen.cmake)
    if(${BUILD_API_DOCUMENTATION})
        create_doxygen_target("${FLAMEGPU_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" "")
    endif()
    return()
endif()

# Name the project and set languages
project(flamegpu2 CUDA CXX)

# Include common rules.
include(${FLAMEGPU_ROOT}/cmake/common.cmake)
include(${FLAMEGPU_ROOT}/cmake/doxygen.cmake)

if(VISUALISATION)
  include(${FLAMEGPU_ROOT}/cmake/flamegpu2_visualiser.cmake)
endif()

# Define output location of static library
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # If top level project
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/${CMAKE_SYSTEM_NAME_LOWER}-x64/${CMAKE_BUILD_TYPE}/)
    if(${BUILD_API_DOCUMENTATION})
        create_doxygen_target("${FLAMEGPU_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" "")
    endif()
else()
    # If called via add_subdirectory()
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/${CMAKE_SYSTEM_NAME_LOWER}-x64/${CMAKE_BUILD_TYPE}/)
    if(${BUILD_API_DOCUMENTATION})
        create_doxygen_target("${FLAMEGPU_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/.." "")
    endif()
endif()

# Enable parallel compilation
# Only do this for lib, as examples already build target parallel
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
endif()

# Specify source files for the static library
# Can't do this automatically, as CMake wouldn't know when to regen (as CMakeLists.txt would be unchanged)
SET(SRC_INCLUDE
    ${FLAMEGPU_ROOT}/include/flamegpu/io/statereader.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/statewriter.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/jsonReader.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/jsonWriter.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/xmlReader.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/xmlWriter.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/factory.h
    ${FLAMEGPU_ROOT}/include/flamegpu/exception/FGPUException.h
    ${FLAMEGPU_ROOT}/include/flamegpu/exception/FGPUStaticAssert.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/LayerDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/ModelData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentFunctionData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/LayerData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentFunctionDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/EnvironmentDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/ModelDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/Variable.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/AgentInstance.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/AgentPopulation.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/MemoryVector.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/AgentStateMemory.h
    ${FLAMEGPU_ROOT}/include/flamegpu/defines.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAScanCompaction.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAErrorChecking.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAMessageList.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAAgentModel.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAMessage.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAAgent.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAAgentStateList.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAFatAgent.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAFatAgentStateList.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAScatter.h
    ${FLAMEGPU_ROOT}/include/flamegpu/flame_api.h
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/AgentInterface.h
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/Simulation.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/AgentFunction.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/AgentFunction_shim.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/AgentFunctionCondition.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/AgentFunctionCondition_shim.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/flamegpu_api.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/flamegpu_device_api.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/flamegpu_host_api.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/flamegpu_host_api_macros.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/flamegpu_host_agent_api.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/flamegpu_host_new_agent_api.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/cuRVE/curve.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/cuRVE/curve_rtc.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/None.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/BruteForce.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/Spatial2D.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/Spatial3D.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/Array.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/Array2D.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/Array3D.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/AgentRandom.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/DeviceEnvironment.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/EnvironmentManager.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/HostEnvironment.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/HostRandom.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/RandomManager.cuh    
    ${FLAMEGPU_ROOT}/include/flamegpu/util/nvtx.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/SignalHandlers.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/compute_capability.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubModelData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubAgentData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubEnvironmentData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubModelDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubAgentDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubEnvironmentDescription.h
)
SET(SRC_FLAMEGPU2
    ${FLAMEGPU_ROOT}/src/flamegpu/exception/FGPUException.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/LayerDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/ModelData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentFunctionData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/LayerData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/ModelDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/EnvironmentDescription.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentFunctionDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/pop/AgentPopulation.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/pop/AgentInstance.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/pop/AgentStateMemory.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAScanCompaction.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAMessageList.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAAgent.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAAgentStateList.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAFatAgent.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAFatAgentStateList.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAMessage.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAScatter.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/sim/Simulation.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAAgentModel.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/cuRVE/curve.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/cuRVE/curve_rtc.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/flamegpu_host_api.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/flamegpu_host_agent_api.cu 
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/BruteForce.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/Spatial2D.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/Spatial3D.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/Array.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/Array2D.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/Array3D.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/io/jsonReader.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/io/jsonWriter.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/io/xmlReader.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/io/xmlWriter.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/utility/HostEnvironment.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/utility/EnvironmentManager.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/utility/RandomManager.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/util/compute_capability.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubModelData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubAgentData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubEnvironmentData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubModelDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubAgentDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubEnvironmentDescription.cpp
)
if (VISUALISATION)
    SET(SRC_INCLUDE
        ${SRC_INCLUDE}
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/ModelVis.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/AgentVis.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/AgentStateVis.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/StaticModelVis.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/LineVis.h
    )
    SET(SRC_FLAMEGPU2
        ${SRC_FLAMEGPU2}
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/ModelVis.cpp    
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/AgentVis.cpp     
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/AgentStateVis.cpp
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/StaticModelVis.cpp
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/LineVis.cpp
    )
endif()
SET(ALL_SRC
    ${SRC_INCLUDE}
    ${SRC_FLAMEGPU2}
)

# Setup Visual Studio (and eclipse) filters
source_group(TREE ${FLAMEGPU_ROOT}/include/flamegpu PREFIX include FILES ${SRC_INCLUDE})
set(T_SRC "${SRC_FLAMEGPU2}")
list(FILTER T_SRC INCLUDE REGEX ".*\.(h|hpp|cuh)$")
source_group(TREE ${FLAMEGPU_ROOT}/src/flamegpu PREFIX headers FILES ${T_SRC})
set(T_SRC "${SRC_FLAMEGPU2}")
list(FILTER T_SRC EXCLUDE REGEX ".*\.(h|hpp|cuh)$")
source_group(TREE ${FLAMEGPU_ROOT}/src/flamegpu PREFIX src FILES ${T_SRC})

# Define which source files are required for the target executable
add_flamegpu_library("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}")

# Also set as startup project (if top level project)
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT "${PROJECT_NAME}")
