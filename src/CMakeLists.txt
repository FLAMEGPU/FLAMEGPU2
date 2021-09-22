# Minimum CMake version 3.18 for CUDA --std=c++17 
cmake_minimum_required(VERSION 3.18...3.25 FATAL_ERROR)

# Set the location of the ROOT flame gpu project relative to this CMakeList.txt
get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. REALPATH)

# Include versioning CMake logic, to set the project version 
include(${FLAMEGPU_ROOT}/cmake/version.cmake)

# Include CMake Logic and call a method to record user-provided CMAKE_CUDA_ARCHITECTURES
include(${FLAMEGPU_ROOT}/cmake/CUDAArchitectures.cmake)
flamegpu_init_cuda_architectures()

# Set the project  with no languages iniitailly, in case of doxygen only builds.
project(flamegpu LANGUAGES NONE VERSION ${FLAMEGPU_VERSION})

# See what languages are supported. Must be called within a project.
# This is primarily here incase someone targets src/CMakeFiles.txt directly
include(CheckLanguage)
include(CheckCXXCompilerFlag)
check_language(CXX)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER STREQUAL NOTFOUND)
    #Not able to build code, so just make docs
    message("Suitable compilers for building code not found.\n" "Attempting generation of minimal documentation only project.")
    include(${FLAMEGPU_ROOT}/cmake/dependencies/doxygen.cmake)
    if(${FLAMEGPU_BUILD_API_DOCUMENTATION})
        flamegpu_create_doxygen_target("${FLAMEGPU_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" "")
    endif()
    return()
endif()

# Enable the required languages for compilation
enable_language(C)
enable_language(CXX)
enable_language(CUDA)

# Set CMAKE_CUDA_ARCHITECTURES now CUDA is enabled
flamegpu_set_cuda_architectures()

# Ensure this is not an in-source build
include(${FLAMEGPU_ROOT}/cmake/OutOfSourceOnly.cmake)

# include for dependent modules
include(CMakeDependentOption)

# option to opt-out of sharing usage statistics, reads from the env var FLAMEGPU_SHARE_USAGE_STATISTICS_DEFAULT first which is treated as a cmake bool if it is defined.
set (FLAMEGPU_SHARE_USAGE_STATISTICS_DEFAULT ON)
# If the environment variable is defined and not truth-y, turn it off.
if (DEFINED ENV{FLAMEGPU_SHARE_USAGE_STATISTICS} AND NOT "$ENV{FLAMEGPU_SHARE_USAGE_STATISTICS}")
    set (FLAMEGPU_SHARE_USAGE_STATISTICS_DEFAULT OFF)
endif()
# Option to opt-in to sharing usage statistics
option(FLAMEGPU_SHARE_USAGE_STATISTICS "Support for academic software is dependant on evidence of impact. Enabling this sends anonymous usage statistics so that we can continue to develop it under a free licence." ${FLAMEGPU_SHARE_USAGE_STATISTICS_DEFAULT})
unset(FLAMEGPU_SHARE_USAGE_STATISTICS_DEFAULT)

# dependet option to suppress the telemetry notice if telemetry is off.
set (FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE_DEFAULT OFF)
# If the environment variable is defined and is truth-y, turn it on.
if (DEFINED ENV{FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE} AND "$ENV{FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE}")
    set (FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE_DEFAULT ON)
endif()
cmake_dependent_option(FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE "Suppress notice encouraging telemetry to be enabled, which is emitted once per binary execution if telemetry is disabled." ${FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE_DEFAULT} "FLAMEGPU_SHARE_USAGE_STATISTICS" ${FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE_DEFAULT} )
unset(FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE_DEFAULT)

# advanced developer option to switch telemetery to a test mode, which is reported separately on telemetry hub.
set (FLAMEGPU_TELEMETRY_TEST_MODE_DEFAULT OFF)
# If the environment variable is defined and is truth-y, turn it on.
if (DEFINED ENV{FLAMEGPU_TELEMETRY_TEST_MODE} AND "$ENV{FLAMEGPU_TELEMETRY_TEST_MODE}")
    set (FLAMEGPU_TELEMETRY_TEST_MODE_DEFAULT ON)
endif()
cmake_dependent_option(FLAMEGPU_TELEMETRY_TEST_MODE "Intended for use during development of FLAMEGPU rather than use" ${FLAMEGPU_TELEMETRY_TEST_MODE_DEFAULT} "FLAMEGPU_SHARE_USAGE_STATISTICS" ${FLAMEGPU_TELEMETRY_TEST_MODE_DEFAULT} )
mark_as_advanced(FLAMEGPU_TELEMETRY_TEST_MODE)
unset(FLAMEGPU_TELEMETRY_TEST_MODE_DEFAULT)

# Option to enable/disable runtime checks which may impact performance
# This will primarily prevent device code from reporting errors
option(FLAMEGPU_SEATBELTS "Enable runtime checks which harm performance for release/profile builds.\nThis should only be disabled after a model is known to be correct." ON)

# Option to enable/disable building the flamegpu visualiser into the flamegpu
option(FLAMEGPU_VISUALISATION "Enable FLAMEGPU visualisation support" OFF)

# Option to enable/disable the default status of JitifyCache
option(FLAMEGPU_RTC_DISK_CACHE "Enable caching of RTC kernels to disk by default (this can still be overridden programatically)." ON)

# Option to enable/disable logging of dynamic RTC files to disk
option(FLAMEGPU_RTC_EXPORT_SOURCES "Export RTC source files to disk at runtime" OFF)

# Option to enable / disable linting of the flamegpu target by default, so repos which consume this repository via add_subdir do not create the lint_flamegpu target or include it in all_lint by default.
# Defaults to OFF unless this, or FLAMEGPU_ROOT is the top level project.
# cmake_dependent option wouldn't cooperate.
set(FLAMEGPU_ENABLE_LINT_FLAMEGPU_DEFAULT OFF)
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}" OR "${CMAKE_SOURCE_DIR}" STREQUAL "${FLAMEGPU_ROOT}")
    set(FLAMEGPU_ENABLE_LINT_FLAMEGPU_DEFAULT ON)
endif()
option(FLAMEGPU_ENABLE_LINT_FLAMEGPU "Enable creation of the lint_flamegpu target" ${FLAMEGPU_ENABLE_LINT_FLAMEGPU_DEFAULT})
unset(FLAMEGPU_ENABLE_LINT_FLAMEGPU_DEFAULT)

# Option to enable/disable MPI support
option(FLAMEGPU_ENABLE_MPI "Enable MPI support for distributed ensembles" OFF)

# Option to enable GLM support and put GLM on the include path
option(FLAMEGPU_ENABLE_GLM "Experimental: Make GLM available to flamegpu2 projects on the include path" OFF)
mark_as_advanced(FLAMEGPU_ENABLE_GLM)

# Option to enable the "advanced API" - access to high performance internals that may break at any time 
option(FLAMEGPU_ENABLE_ADVANCED_API "Enable advanced API functionality (C++ only), providing access to internal sim components for high-performance extensions." OFF)
mark_as_advanced(FLAMEGPU_ENABLE_ADVANCED_API)

# Include common rules.
include(${FLAMEGPU_ROOT}/cmake/common.cmake)
# Include the doxygen cmake logic.
include(${FLAMEGPU_ROOT}/cmake/dependencies/doxygen.cmake)

# Include the visualiser cmake if requested
if(FLAMEGPU_VISUALISATION)
  include(${FLAMEGPU_ROOT}/cmake/dependencies/flamegpu2-visualiser.cmake)
endif()

# Output the full FLAMEGPU version
if(FLAMEGPU_VERSION_FULL)
    message(STATUS "${PROJECT_NAME} version ${FLAMEGPU_VERSION_FULL}")
endif()

# Configure the version header file. Ideally this would configure a .cpp file in the build directory, but that doesn't allow for rtc header include checking sub major/minor/patch.
set(DYNAMIC_VERSION_SRC_SRC "${FLAMEGPU_ROOT}/cmake/version.cpp.in")
set(DYNAMIC_VERSION_SRC_DEST "${CMAKE_CURRENT_BINARY_DIR}/src/flamegpu/version.cpp")
configure_file(${DYNAMIC_VERSION_SRC_SRC} ${DYNAMIC_VERSION_SRC_DEST} @ONLY)

# Define output location of static library
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # If top level project
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE}/)
    if(${FLAMEGPU_BUILD_API_DOCUMENTATION})
        flamegpu_create_doxygen_target("${FLAMEGPU_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" "")
    endif()
else()
    # If called via add_subdirectory()
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/${CMAKE_BUILD_TYPE}/)
    if(${FLAMEGPU_BUILD_API_DOCUMENTATION})
        flamegpu_create_doxygen_target("${FLAMEGPU_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/.." "")
    endif()
endif()

# Specify source files for the static library
# Can't do this automatically, as CMake wouldn't know when to regen (as CMakeLists.txt would be unchanged)
# version.h is automatically generated so does not exist prior to configure time
SET(SRC_INCLUDE
    ${FLAMEGPU_ROOT}/include/flamegpu/version.h
    ${FLAMEGPU_ROOT}/include/flamegpu/defines.h
    ${FLAMEGPU_ROOT}/include/flamegpu/flamegpu.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/StateReader.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/StateWriter.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/JSONStateReader.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/JSONStateWriter.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/XMLStateReader.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/XMLStateWriter.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/StateReaderFactory.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/StateWriterFactory.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/Logger.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/LoggerFactory.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/XMLLogger.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/JSONLogger.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/JSONGraphReader.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/JSONGraphWriter.h
    ${FLAMEGPU_ROOT}/include/flamegpu/io/Telemetry.h
    ${FLAMEGPU_ROOT}/include/flamegpu/exception/FLAMEGPUException.h
    ${FLAMEGPU_ROOT}/include/flamegpu/exception/FLAMEGPUDeviceException.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/exception/FLAMEGPUDeviceException_device.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/model/ModelData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/ModelDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentFunctionData.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentFunctionDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/DependencyNode.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/DependencyGraph.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/LayerData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/LayerDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/EnvironmentData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/EnvironmentDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubModelData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubModelDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubAgentData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubAgentDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubEnvironmentData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubEnvironmentDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/HostFunctionDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/Variable.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/EnvironmentDirectedGraphData.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/model/EnvironmentDirectedGraphDescription.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/AbstractSimRunner.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/MemoryVector.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/GenericMemoryVector.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/CUDAScanCompaction.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/CUDAErrorChecking.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/CubTemporaryMemory.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/CUDAMessageList.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/CUDAMessage.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/CUDAAgent.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/CUDAAgentStateList.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/CUDAFatAgent.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/CUDAFatAgentStateList.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/CUDAScatter.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/CUDAMacroEnvironment.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/MPISimRunner.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/MPIEnsemble.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/SimRunner.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/SimLogger.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/AgentInterface.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/EnvironmentManager.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/RandomManager.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/CUDAEnvironmentDirectedGraphBuffers.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/DeviceStrings.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/EnvironmentManager.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/detail/RandomManager.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/AgentVector.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/AgentVector_Agent.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/CUDASimulation.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/CUDAEnsemble.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/AgentLoggingConfig.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/AgentLoggingConfig_SumReturn.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/AgentLoggingConfig_Reductions.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/LoggingConfig.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/LogFrame.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/RunPlan.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/RunPlanVector.h
    ${FLAMEGPU_ROOT}/include/flamegpu/simulation/Simulation.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/AgentFunction.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/AgentFunction_shim.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/AgentFunctionCondition.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/AgentFunctionCondition_shim.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/HostFunctionCallback.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/DeviceAPI.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/HostAPI.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/HostAPI_macros.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/detail/SharedBlock.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/detail/curve/Curve.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/detail/curve/DeviceCurve.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/detail/curve/HostCurve.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/detail/curve/curve_rtc.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/agent/HostAgentAPI.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/agent/HostNewAgentAPI.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/agent/AgentInstance.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/agent/DeviceAgentVector.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/agent/DeviceAgentVector_impl.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging_device.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageSpecialisationHandler.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageNone.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageNone/MessageNoneHost.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageNone/MessageNoneDevice.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageBruteForce.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceDevice.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceHost.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageSpatial2D.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageSpatial2D/MessageSpatial2DHost.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageSpatial2D/MessageSpatial2DDevice.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageSpatial3D.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageSpatial3D/MessageSpatial3DHost.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageSpatial3D/MessageSpatial3DDevice.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageArray.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageArray/MessageArrayHost.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageArray/MessageArrayDevice.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageArray2D.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageArray2D/MessageArray2DHost.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageArray2D/MessageArray2DDevice.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageArray3D.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageArray3D/MessageArray3DHost.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageArray3D/MessageArray3DDevice.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageBucket.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageBucket/MessageBucketHost.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageBucket/MessageBucketDevice.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/messaging/MessageSortingType.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/random/AgentRandom.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/random/HostRandom.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/environment/DeviceEnvironment.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/environment/DeviceMacroProperty.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/environment/HostEnvironment.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/environment/HostMacroProperty.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/environment/DeviceEnvironmentDirectedGraph.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/environment/HostEnvironmentDirectedGraph.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/util/cleanup.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/dstring.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/nvtx.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/StringPair.h
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/Any.h
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/type_decode.h
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/compute_capability.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/curand.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/wddm.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/CUDAEventTimer.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/cuda.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/cxxname.hpp
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/SignalHandlers.h
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/StaticAssert.h
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/SteadyClockTimer.h
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/Timer.h
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/TestSuiteTelemetry.h
    ${FLAMEGPU_ROOT}/include/flamegpu/detail/JitifyCache.h
)
SET(SRC_FLAMEGPU
    ${FLAMEGPU_ROOT}/src/flamegpu/exception/FLAMEGPUException.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/exception/FLAMEGPUDeviceException.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/model/ModelDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/ModelData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentFunctionData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentFunctionDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/LayerData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/LayerDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/EnvironmentData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/EnvironmentDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/DependencyNode.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/DependencyGraph.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubModelData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubModelDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubAgentData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubAgentDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubEnvironmentData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubEnvironmentDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/HostFunctionDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/EnvironmentDirectedGraphData.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/model/EnvironmentDirectedGraphDescription.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/AbstractSimRunner.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/CubTemporaryMemory.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/CUDAScanCompaction.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/CUDAMessageList.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/CUDAAgent.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/CUDAAgentStateList.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/CUDAFatAgent.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/CUDAFatAgentStateList.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/CUDAMessage.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/CUDAScatter.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/CUDAMacroEnvironment.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/DeviceStrings.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/CUDAEnvironmentDirectedGraphBuffers.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/MPISimRunner.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/MPIEnsemble.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/SimRunner.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/SimLogger.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/EnvironmentManager.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/detail/RandomManager.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/AgentVector.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/AgentVector_Agent.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/AgentInstance.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/CUDASimulation.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/CUDAEnsemble.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/AgentLoggingConfig.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/LoggingConfig.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/LogFrame.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/RunPlan.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/RunPlanVector.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/simulation/Simulation.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/detail/curve/Curve.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/detail/curve/curve_rtc.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/detail/curve/HostCurve.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/agent/DeviceAgentVector_impl.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/agent/HostAgentAPI.cu 
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/HostAPI.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageBruteForce.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageSpatial2D.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageSpatial3D.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageArray.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageArray2D.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageArray3D.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageBucket.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/environment/HostEnvironment.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/environment/HostEnvironmentDirectedGraph.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/random/HostRandom.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/io/JSONStateReader.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/io/JSONStateWriter.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/io/StateReader.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/io/XMLStateReader.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/io/XMLStateWriter.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/io/XMLLogger.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/io/JSONLogger.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/io/JSONGraphReader.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/io/JSONGraphWriter.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/io/Telemetry.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/util/cleanup.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/detail/compute_capability.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/detail/wddm.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/detail/JitifyCache.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/detail/TestSuiteTelemetry.cpp
)
SET(SRC_DYNAMIC
    ${DYNAMIC_VERSION_SRC_DEST}
)
# List include/visualiser files so they can be linted outside of visualiser builds
SET(SRC_INCLUDE_VISUALISER
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/visualiser_api.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/ModelVis.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/AgentVis.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/AgentStateVis.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/StaticModelVis.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/LineVis.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/PanelVis.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/Color.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/ColorFunction.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/DiscreteColor.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/StaticColor.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/HSVInterpolation.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/ViridisInterpolation.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/Palette.h
    ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/AutoPalette.h
)
# List flamegpu/visualiser files so they can be linted outside of visualiser builds
set(SRC_FLAMEGPU_VISUALISER
    ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/ModelVis.cpp    
    ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/AgentVis.cpp     
    ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/AgentStateVis.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/StaticModelVis.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/LineVis.cpp
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/PanelVis.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/color/DiscreteColor.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/color/StaticColor.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/color/HSVInterpolation.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/color/ViridisInterpolation.cpp
)
# If visualisation is enabled, the the visualiser inc/src files must be in the appropriate lists
if (FLAMEGPU_VISUALISATION)
    SET(SRC_INCLUDE
        ${SRC_INCLUDE}
        ${SRC_INCLUDE_VISUALISER}
    )
    SET(SRC_FLAMEGPU
        ${SRC_FLAMEGPU}
        ${SRC_FLAMEGPU_VISUALISER}
    )
endif()
# Build the list of all source files.
SET(ALL_SRC
    ${SRC_INCLUDE}
    ${SRC_FLAMEGPU}
    ${SRC_DYNAMIC}
)
SET(FLAMEGPU_INCLUDE ${SRC_INCLUDE} CACHE INTERNAL "Include files required by FLAMEGPU RTC")

# Setup Visual Studio (and eclipse) filters, using the TREE variant separating header and source files.
# Potentially the build directory is not a child of the repository,
# in which case dynamic files (version.cpp) cannot use the tree component, so we must account for that still.

# Get a regex escaped represenatation of the FLAMEGPU_ROOT dir, for paths containg + etc.
escape_regex("${FLAMEGPU_ROOT}" FLAMEGPU_ROOT_ESCAPE)

# Convert all paths to abs paths, to remove any ../ components
set(ABS_ALL_SRC "")
foreach(FILEPATH IN LISTS ALL_SRC)
    get_filename_component(ABS_FILEPATH ${FILEPATH} REALPATH)
    list(APPEND ABS_ALL_SRC ${ABS_FILEPATH})
    unset(ABS_FILEPATH)
endforeach()

# Filter files which cannot be used with sourge_group(TREE) into separate lists.
set(SRC_GROUP_TREE_COMPATIBLE "${ABS_ALL_SRC}")
set(SRC_GROUP_MANUAL "${ABS_ALL_SRC}")
list(FILTER SRC_GROUP_TREE_COMPATIBLE INCLUDE REGEX "^${FLAMEGPU_ROOT_ESCAPE}/")
list(FILTER SRC_GROUP_MANUAL EXCLUDE REGEX "^${FLAMEGPU_ROOT_ESCAPE}/")
unset(ABS_ALL_SRC)

# Filter out header and source files separately for those which can use TREE
set(SRC_GROUP_TREE_COMPATIBLE_HEADERS "${SRC_GROUP_TREE_COMPATIBLE}")
list(FILTER SRC_GROUP_TREE_COMPATIBLE_HEADERS INCLUDE REGEX ".*\.(h|hpp|cuh)$")
set(SRC_GROUP_TREE_COMPATIBLE_SOURCES "${SRC_GROUP_TREE_COMPATIBLE}")
list(FILTER SRC_GROUP_TREE_COMPATIBLE_SOURCES EXCLUDE REGEX ".*\.(h|hpp|cuh)$")
# Apply source group filters with TREE, using CMake's default "Header Files" and "Source Files" for consistency
source_group(TREE ${FLAMEGPU_ROOT} PREFIX "Header Files" FILES ${SRC_GROUP_TREE_COMPATIBLE_HEADERS})
source_group(TREE ${FLAMEGPU_ROOT} PREFIX "Source Files" FILES ${SRC_GROUP_TREE_COMPATIBLE_SOURCES})
# Clean up variables
unset(SRC_GROUP_TREE_COMPATIBLE_HEADERS)
unset(SRC_GROUP_TREE_COMPATIBLE_SOURCES)
unset(SRC_GROUP_TREE_COMPATIBLE)

# Filter out header and source files which CANNOT use TREE (dynamic files if build is not a child of the repo, i.e. model template)
set(SRC_GROUP_MANUAL_HEADERS "${SRC_GROUP_MANUAL}")
list(FILTER SRC_GROUP_MANUAL_HEADERS INCLUDE REGEX ".*\.(h|hpp|cuh)$")
set(SRC_GROUP_MANUAL_SOURCES "${SRC_GROUP_MANUAL}")
list(FILTER SRC_GROUP_MANUAL_SOURCES EXCLUDE REGEX ".*\.(h|hpp|cuh)$")
# Apply source group filters WITHOUT TREE, using CMake's default "Header Files" and "Source Files" for consistency
# These will just be placed in the root of the folder, as we cannot have a ../ type setup in sources, so no point bothering with directories
source_group("Header Files" FILES ${SRC_GROUP_MANUAL_HEADERS})
source_group("Source Files" FILES ${SRC_GROUP_MANUAL_SOURCES})
# Clean up variables
unset(SRC_GROUP_MANUAL_HEADERS)
unset(SRC_GROUP_MANUAL_SOURCES)
unset(SRC_GROUP_MANUAL)

# Create the library target and set various properties

# Define which source files are required for the target executable
add_library(${PROJECT_NAME} STATIC ${ALL_SRC})

# Require C++17 as a public target property for C++ and CUDA, with no extensions, and the standard is required
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_compile_features(${PROJECT_NAME} PUBLIC cuda_std_17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_EXTENSIONS OFF)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD_REQUIRED ON)

# Add the library headers as public so they are forwarded on.
target_include_directories(${PROJECT_NAME} PUBLIC "${FLAMEGPU_ROOT}/include")
# Add any private headers.
target_include_directories(${PROJECT_NAME} PRIVATE "${FLAMEGPU_ROOT}/src")

# Set target level warnings.
flamegpu_enable_compiler_warnings(TARGET "${PROJECT_NAME}")
# Apply common compiler settings
flamegpu_common_compiler_settings(TARGET "${PROJECT_NAME}")

# Ensure that DEBUG macros and -G are set for debug NVCC builds, as public properties so they are inherrited by consumers. Used in include files.
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>")
target_compile_definitions(${PROJECT_NAME} PUBLIC $<$<CONFIG:Debug>:DEBUG>)
target_compile_definitions(${PROJECT_NAME} PUBLIC $<$<CONFIG:Debug>:_DEBUG>)

# Prevent windows.h from defining max and min, this can be private as windows.h is only included in src files (directly, or via nvrtc/jitify)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
endif()

# If FLAMEGPU_SEATBELTS is enabled, set it as a public compile defition (i.e it's in the headers)
if (FLAMEGPU_SEATBELTS)
    # If on, all build configs have  seatbelts
    target_compile_definitions(${PROJECT_NAME} PUBLIC FLAMEGPU_SEATBELTS=1)
else()
    # If off, debug builds have seatbelts, non debug builds do not.
    target_compile_definitions(${PROJECT_NAME} PUBLIC $<IF:$<CONFIG:Debug>,FLAMEGPU_SEATBELTS=1,FLAMEGPU_SEATBELTS=0>)
endif()

# Detect and forward the minimum CMAKE_CUDA_ARCHITECTURES value to the compiler, if possible. This is only used in a library source file so can be private.
flamegpu_get_minimum_cuda_architecture(min_cuda_arch)
target_compile_definitions(${PROJECT_NAME} PRIVATE FLAMEGPU_MIN_CUDA_ARCH=${min_cuda_arch})

# Select the curand engine to use for FLAMEGPU, used in include, so must match the library build (PUBLIC)
string(TOUPPER FLAMEGPU_CURAND_ENGINE FLAMEGPU_CURAND_ENGINE_UPPER)
if(${FLAMEGPU_CURAND_ENGINE_UPPER} STREQUAL "MRG")
    target_compile_definitions(${PROJECT_NAME} PUBLIC FLAMEGPU_CURAND_MRG32k3a)
elseif(${FLAMEGPU_CURAND_ENGINE_UPPER} STREQUAL "PHILOX")
    target_compile_definitions(${PROJECT_NAME} PUBLIC FLAMEGPU_CURAND_Philox4_32_10)
elseif(${FLAMEGPU_CURAND_ENGINE_UPPER} STREQUAL "XORWOW")
    target_compile_definitions(${PROJECT_NAME} PUBLIC FLAMEGPU_CURAND_XORWOW)
elseif(DEFINED FLAMEGPU_CURAND_ENGINE)
    message(FATAL_ERROR "${FLAMEGPU_CURAND_ENGINE} is not a suitable value of FLAMEGPU_CURAND_ENGINE\nOptions: \"MRG\", \"PHILOX\", \"XORWOW\"")
endif()
unset(FLAMEGPU_CURAND_ENGINE_UPPER)

# Telemetry (perform this check here are not in common as it only effects the library build)
if (FLAMEGPU_SHARE_USAGE_STATISTICS)
    # If on, then set pre-processor
    target_compile_definitions(${PROJECT_NAME} PRIVATE FLAMEGPU_SHARE_USAGE_STATISTICS=1)
else()
    message(STATUS "FLAMEGPU_SHARE_USAGE_STATISTICS is not selected. \n\n"
    "   Support for academic software is dependant on evidence of impact. \n"
    "   Enabling this option sends minimal anonymous usage statistics so \n"
    "   that we can continue to develop the software under a FOSS licence.\n"
    "   See our documentation for more information of what we collect and \n"
    "   why: https://docs.flamegpu.com/guide/telemetry")
endif()

# If telemetry notice suppression is enabled, forward it to the compiler. 
if (FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE)
    # If on, then set pre-processor
    target_compile_definitions(${PROJECT_NAME} PRIVATE FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE=1)
endif()

# If telemetry test mode is enabled, forward it to the compiler. 
if (FLAMEGPU_TELEMETRY_TEST_MODE)
    # If on, then set pre-processor
    target_compile_definitions(${PROJECT_NAME} PRIVATE FLAMEGPU_TELEMETRY_TEST_MODE=1)
endif()

# enable "fpic" for linux to allow shared libraries to be build from the static library (required for swig)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Activate visualisation if requested
if (FLAMEGPU_VISUALISATION)
    # @todo - these could/should be private, but must be PUBLIC for swig to build. Alternatively it could be PRIVATE but also set for swig.
    target_link_libraries(${PROJECT_NAME} PUBLIC flamegpu_visualiser)
    flamegpu_set_target_folder(flamegpu_visualiser "FLAMEGPU")
    target_compile_definitions(${PROJECT_NAME} PUBLIC FLAMEGPU_VISUALISATION) 
endif()

# Activate MPI support (distributed ensembles)
if (FLAMEGPU_ENABLE_MPI)
    find_package(MPI REQUIRED)
    if (MPI_CXX_FOUND)
        # If CMake is older than 3.20.1, FindMPI may incorrectly wrap link options when CUDA is enabled.
        # It is not guaranteed to be a problem, so we should warn rather than error.
        # https://gitlab.kitware.com/cmake/cmake/-/issues/21887
        if(CMAKE_VERSION VERSION_LESS 3.20.1)
            message(WARNING "CMake < 3.20.1 may result in link errors with FLAMEGPU_ENABLE_MPI=ON for some MPI installations. Consider using CMake >= 3.20.1.")
        endif()
        # If the MPI installation brings in -flto (i.e. Ubuntu 22.04 libmpich-dev), warn about it and suggest a reconfiguration.
        if(MPI_CXX_COMPILE_OPTIONS MATCHES ".*\-flto.*")
            message(WARNING
                " MPI_CXX_COMPILE_OPTIONS contains '-flto' which is likely to result in linker errors. \n"
                " Consider an alternate MPI implementation which does not embed -flto,\n"
                " Or reconfiguring CMake with -DMPI_CXX_COMPILE_OPTIONS=\"\" if linker error occur.")
        endif()
        # This sets up include directories and lib
        target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX)
        target_compile_definitions(${PROJECT_NAME} PUBLIC FLAMEGPU_ENABLE_MPI)
    else()
        message(FATAL_ERROR "MPI C++ Support was not found!")
    endif()
endif()

# Make GLM accessible via include. PUBLIC so this is usable by downstream projects
# @todo - make the vis cmake/glm create a target to use.
if (FLAMEGPU_ENABLE_GLM)
    if(TARGET glm::glm)
        # Get the interface include directories property from the glm target to forward onto nvrtc, which is a bit grim (i.e. not portable/installable)
        target_link_libraries(${PROJECT_NAME} PUBLIC glm::glm)
        target_compile_definitions(${PROJECT_NAME} PUBLIC FLAMEGPU_USE_GLM)
        get_target_property(glm_inc_path glm::glm INTERFACE_INCLUDE_DIRECTORIES)
        target_compile_definitions(${PROJECT_NAME} PUBLIC FLAMEGPU_GLM_PATH="${glm_inc_path}")
        unset(glm_inc_path)
    else()
        message(WARNING "FLAMEGPU_ENABLE_GLM enabled, but glm was not found")
    endif()
endif() 

# Activate advanced API macro
if (FLAMEGPU_ENABLE_ADVANCED_API)
    target_compile_definitions(${PROJECT_NAME} PUBLIC FLAMEGPU_ADVANCED_API)
endif()  


if (NOT FLAMEGPU_RTC_DISK_CACHE)
    # Only used in source files, so can be private
    target_compile_definitions(${PROJECT_NAME} PRIVATE FLAMEGPU_DISABLE_RTC_DISK_CACHE)
endif()    
if (FLAMEGPU_RTC_EXPORT_SOURCES)
    # Only used in source files, so can be private
    target_compile_definitions(${PROJECT_NAME} PRIVATE FLAMEGPU_OUTPUT_RTC_DYNAMIC_FILES)
endif ()

# Enable RDC
set_property(TARGET ${PROJECT_NAME}  PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link against dependency targets / directories.

# Cub and thrust targets are not imported targets, so they do not use -isystem, so warnings must be suppressed as pragmas as requied. This is due to nvcc magic preventing isystem from being reliable with them. 
target_link_libraries(${PROJECT_NAME} PUBLIC CUB::CUB)
target_link_libraries(${PROJECT_NAME} PUBLIC Thrust::Thrust)

# tinyxml2 static library
target_link_libraries(${PROJECT_NAME} PRIVATE Tinyxml2::tinyxml2)

# If rapidjson was found, add it to the include dirs.
target_link_libraries(${PROJECT_NAME} PRIVATE RapidJSON::rapidjson)

target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::nvrtc)

target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cuda_driver)

if(FLAMEGPU_ENABLE_NVTX AND TARGET NVTX::nvtx)
    target_link_libraries(${PROJECT_NAME} PUBLIC NVTX::nvtx)
    # Get the version to set a definition value 
    # Interface targets only support the version property from 3.19
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
        get_target_property(nvtxversion NVTX::nvtx VERSION)
    else()
        # fallback to the old cmake var.
        set(nvtxversion ${NVTX_VERSION})
    endif()
    target_compile_definitions(${PROJECT_NAME} PUBLIC "FLAMEGPU_USE_NVTX=${nvtxversion}")
    unset(nvtxversion)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC Jitify::jitify)
# jitify is included in public headers, so this definiition must be public too.
target_compile_definitions(${PROJECT_NAME} PUBLIC "JITIFY_PRINT_LOG")

# If gcc < 9, needs to link against stdc++fs. If using NVHPC,this is required if the underlying toolchain is also < 9. @todo - need to detect gcc version when nvhpc is the host compiler somehow...
if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0) OR OR (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC"))
    target_link_libraries(${PROJECT_NAME} PUBLIC stdc++fs)
endif()

# Link against dl if available. This used to be implied from somewhere...
if(CMAKE_DL_LIBS)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_DL_LIBS})
endif()

# Ensure we link against pthread / windows equilvalent. This was previously implied
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# Flag the new linter target and the files to be linted, if enabled. Ensure vis sources are linted even if vis is not enabled (mostly for CI)
if(FLAMEGPU_ENABLE_LINT_FLAMEGPU)
    if (FLAMEGPU_VISUALISATION)
        # If vis is enabled, just use ALL_SRC
        flamegpu_new_linter_target(${PROJECT_NAME} "${ALL_SRC}")
    else()
        # If vis is not enabled, pass in the vis source files too.
        flamegpu_new_linter_target(${PROJECT_NAME} "${ALL_SRC};${SRC_INCLUDE_VISUALISER};${SRC_FLAMEGPU_VISUALISER}")
    endif()
endif()
# Put within FLAMEGPU filter
flamegpu_set_target_folder(${PROJECT_NAME} "FLAMEGPU")
# Put the tinyxml2 in the folder
flamegpu_set_target_folder("tinyxml2" "FLAMEGPU/Dependencies")

# Emit some warnings that should only be issued once and are related to this file (but not this target)
# @todo - move these author warnigns elsewhere, so they will be emitted if not building FLAMEGPU iteself? Common but with some form of guard?
if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS_EQUAL "10.2")
    message(AUTHOR_WARNING "MSVC and NVCC <= 10.2 may encounter compiler errors due to an NVCC bug exposed by Thrust. Cosider using a newer CUDA toolkit.")
endif()
if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS_EQUAL "11.0")
    # https://github.com/FLAMEGPU/FLAMEGPU2/issues/483
    message(AUTHOR_WARNING "MSVC and NVCC <= 11.0 may encounter errors at link time with incremental rebuilds. Cosider using a newer CUDA toolkit.")
endif()

# Also set as startup project (if top level project)
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT "${PROJECT_NAME}")
