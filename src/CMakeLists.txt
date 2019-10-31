# 3.10 required for cuda -std=c++14, however 3.12 fixes some device linker errors
cmake_minimum_required(VERSION VERSION 3.12 FATAL_ERROR)

# Name the project and set languages
project(flamegpu2 CUDA CXX)

# Set the location of the ROOT flame gpu project relative to this CMakeList.txt
get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. REALPATH)

# Include common rules.
include(${FLAMEGPU_ROOT}/cmake/common.cmake)

# Define output location of static library
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${FLAMEGPU_ROOT}/lib/${CMAKE_SYSTEM_NAME_LOWER}-x64/${CMAKE_BUILD_TYPE}/)

# Specify source files for the static library
SET(SRC_EXTERNALS 
    ${FLAMEGPU_ROOT}/externals/tinyxml2/tinyxml2.cpp
)
SET(SRC_INCLUDE 
    ${FLAMEGPU_ROOT}/include/flamegpu/io/statereader.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/statewriter.h
	${FLAMEGPU_ROOT}/include/flamegpu/io/xmlReader.h
	${FLAMEGPU_ROOT}/include/flamegpu/io/xmlWriter.h
	${FLAMEGPU_ROOT}/include/flamegpu/io/factory.h
    ${FLAMEGPU_ROOT}/include/flamegpu/exception/FGPUException.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/MessageDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentFunctionInput.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentFunctionOutput.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentFunctionDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentStateDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/ModelDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/AgentInstance.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/AgentPopulation.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/MemoryVector.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/AgentStateMemory.h
    ${FLAMEGPU_ROOT}/include/flamegpu/defines.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAErrorChecking.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAMessageList.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAAgentModel.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAMessage.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAAgent.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAAgentStateList.h
    ${FLAMEGPU_ROOT}/include/flamegpu/flame_api.h
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/SimulationLayer.h
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/Simulation.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messagelist.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/agent_function.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/flame_functions_api.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/cuRVE/curve.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/cuRVE/cuRVEInstance.h
)
SET(SRC_FLAMEGPU2
    ${FLAMEGPU_ROOT}/src/flamegpu/model/ModelDescription.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentFunctionInput.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentDescription.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentFunctionDescription.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/model/MessageDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentFunctionOutput.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentStateDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/pop/AgentPopulation.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/pop/AgentInstance.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/pop/AgentStateMemory.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAMessageList.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAAgentStateList.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAAgent.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAMessage.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/sim/SimulationLayer.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/sim/Simulation.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAAgentModel.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/agent_function.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/cuRVE/curve.cu
	${FLAMEGPU_ROOT}/src/flamegpu/io/xmlReader.cu
	${FLAMEGPU_ROOT}/src/flamegpu/io/xmlWriter.cu
)

SET(ALL_SRC 
    ${SRC_EXTERNALS}
    ${SRC_INCLUDE}
    ${SRC_FLAMEGPU2}
)

# Setup Visual Studio (and eclipse) filters
source_group(TREE ${FLAMEGPU_ROOT} FILES ${SRC_EXTERNALS})
source_group(TREE ${FLAMEGPU_ROOT}/include/flamegpu PREFIX include FILES ${SRC_INCLUDE})
source_group(TREE ${FLAMEGPU_ROOT}/src/flamegpu PREFIX src FILES ${SRC_FLAMEGPU2})

# Define which source files are required for the target executable
add_flamegpu_library("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}")
