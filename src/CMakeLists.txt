# Minimum CMake version 3.18 for CUDA --std=c++17 
cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)

# Set the location of the ROOT flame gpu project relative to this CMakeList.txt
get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. REALPATH)

# Include versioning CMake logic, to set the project version 
include(${FLAMEGPU_ROOT}/cmake/version.cmake)

# Set the project  with no languages iniitailly, in case of doxygen only builds.
project(flamegpu LANGUAGES NONE VERSION ${FLAMEGPU_VERSION})

# See what languages are supported. Must be called within a project.
# This is primarily here incase someone targets src/CMakeFiles.txt directly
include(CheckLanguage)
check_language(CXX)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER STREQUAL NOTFOUND)
    #Not able to build code, so just make docs
    message("Suitable compilers for building code not found.\n" "Attempting generation of minimal documentation only project.")
    include(${FLAMEGPU_ROOT}/cmake/dependencies/doxygen.cmake)
    if(${BUILD_API_DOCUMENTATION})
        create_doxygen_target("${FLAMEGPU_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" "")
    endif()
    return()
endif()

# Enable the required languages for compilation
enable_language(C)
enable_language(CXX)
enable_language(CUDA)

# Ensure this is not an in-source build
include(${FLAMEGPU_ROOT}/cmake/OutOfSourceOnly.cmake)

# Option to enable/disable runtime checks which may impact performance
# This will primarily prevent device code from reporting errors
option(SEATBELTS "Enable runtime checks which harm performance for release/profile builds.\nThis should only be disabled after a model is known to be correct." ON)

# Option to enable/disable the default status of JitifyCache
option(RTC_DISK_CACHE "Enable caching of RTC kernels to disk by default (this can still be overridden programatically)." ON)

# Option to make put glm on the include path
option(USE_GLM "Experimental: Make GLM available to flamegpu2 projects on the include path" OFF)
mark_as_advanced(USE_GLM)

# Include common rules.
include(${FLAMEGPU_ROOT}/cmake/common.cmake)
# Include the doxygen cmake logic.
include(${FLAMEGPU_ROOT}/cmake/dependencies/doxygen.cmake)

# Include the visualiser cmake if requested
if(VISUALISATION)
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
    if(${BUILD_API_DOCUMENTATION})
        create_doxygen_target("${FLAMEGPU_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" "")
    endif()
else()
    # If called via add_subdirectory()
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/${CMAKE_BUILD_TYPE}/)
    if(${BUILD_API_DOCUMENTATION})
        create_doxygen_target("${FLAMEGPU_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/.." "")
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
    ${FLAMEGPU_ROOT}/include/flamegpu/exception/FLAMEGPUException.h
    ${FLAMEGPU_ROOT}/include/flamegpu/exception/FLAMEGPUDeviceException.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/exception/FLAMEGPUDeviceException_device.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/model/LayerDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/ModelData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentFunctionData.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/model/DependencyNode.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/DependencyGraph.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/LayerData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentFunctionDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/HostFunctionDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/EnvironmentDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/AgentDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/ModelDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/Variable.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/detail/MemoryVector.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/detail/GenericMemoryVector.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/AgentVector.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/AgentVector_Agent.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/AgentInstance.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/DeviceAgentVector.h
    ${FLAMEGPU_ROOT}/include/flamegpu/pop/DeviceAgentVector_impl.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAScanCompaction.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/detail/CUDAErrorChecking.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAMessageList.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDASimulation.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAEnsemble.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAMessage.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAAgent.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAAgentStateList.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAFatAgent.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAFatAgentStateList.h
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAScatter.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/gpu/CUDAMacroEnvironment.h
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/AgentInterface.h
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/AgentLoggingConfig.h
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/AgentLoggingConfig_Reductions.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/LoggingConfig.h
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/LogFrame.h
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/RunPlan.h
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/RunPlanVector.h
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/SimRunner.h
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/SimLogger.h
    ${FLAMEGPU_ROOT}/include/flamegpu/sim/Simulation.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/AgentFunction.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/AgentFunction_shim.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/AgentFunctionCondition.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/AgentFunctionCondition_shim.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/HostFunctionCallback.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/DeviceAPI.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/HostAPI.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/HostAPI_macros.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/HostAgentAPI.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/HostNewAgentAPI.h
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/detail/curve/curve.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/detail/curve/curve_rtc.cuh
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
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/AgentRandom.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/DeviceEnvironment.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/DeviceMacroProperty.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/EnvironmentManager.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/HostEnvironment.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/HostMacroProperty.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/HostRandom.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/runtime/utility/RandomManager.cuh    
    ${FLAMEGPU_ROOT}/include/flamegpu/util/Any.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/nvtx.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/StringPair.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/StringUint32Pair.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/detail/compute_capability.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/util/detail/wddm.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/util/detail/CUDAEventTimer.cuh
    ${FLAMEGPU_ROOT}/include/flamegpu/util/detail/cxxname.hpp
    ${FLAMEGPU_ROOT}/include/flamegpu/util/detail/filesystem.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/detail/SignalHandlers.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/detail/StaticAssert.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/detail/SteadyClockTimer.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/detail/Timer.h
    ${FLAMEGPU_ROOT}/include/flamegpu/util/detail/JitifyCache.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubModelData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubAgentData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubEnvironmentData.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubModelDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubAgentDescription.h
    ${FLAMEGPU_ROOT}/include/flamegpu/model/SubEnvironmentDescription.h
)
SET(SRC_FLAMEGPU
    ${FLAMEGPU_ROOT}/src/flamegpu/exception/FLAMEGPUException.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/exception/FLAMEGPUDeviceException.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/model/LayerDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/ModelData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentFunctionData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/LayerData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/ModelDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/EnvironmentDescription.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/AgentFunctionDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/HostFunctionDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/DependencyNode.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/DependencyGraph.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/pop/AgentVector.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/pop/AgentVector_Agent.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/pop/AgentInstance.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/pop/DeviceAgentVector_impl.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAScanCompaction.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAMessageList.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAAgent.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAAgentStateList.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAFatAgent.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAFatAgentStateList.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAMessage.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAScatter.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDASimulation.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAEnsemble.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/gpu/CUDAMacroEnvironment.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/sim/AgentLoggingConfig.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/sim/LoggingConfig.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/sim/LogFrame.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/sim/RunPlan.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/sim/RunPlanVector.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/sim/SimRunner.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/sim/SimLogger.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/sim/Simulation.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/detail/curve/curve.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/detail/curve/curve_rtc.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/HostAPI.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/HostAgentAPI.cu 
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageBruteForce.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageSpatial2D.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageSpatial3D.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageArray.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageArray2D.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageArray3D.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/messaging/MessageBucket.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/io/JSONStateReader.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/io/JSONStateWriter.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/io/XMLStateReader.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/io/XMLStateWriter.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/io/XMLLogger.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/io/JSONLogger.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/utility/HostEnvironment.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/utility/EnvironmentManager.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/utility/RandomManager.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/runtime/utility/HostRandom.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/util/detail/compute_capability.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/util/detail/wddm.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/util/detail/JitifyCache.cu
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubModelData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubAgentData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubEnvironmentData.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubModelDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubAgentDescription.cpp
    ${FLAMEGPU_ROOT}/src/flamegpu/model/SubEnvironmentDescription.cpp
)
SET(SRC_DYNAMIC
    ${DYNAMIC_VERSION_SRC_DEST}
)
if (VISUALISATION)
    SET(SRC_INCLUDE
        ${SRC_INCLUDE}
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/visualiser_api.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/ModelVis.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/AgentVis.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/AgentStateVis.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/StaticModelVis.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/LineVis.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/Color.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/ColorFunction.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/DiscreteColor.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/StaticColor.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/HSVInterpolation.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/ViridisInterpolation.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/Palette.h
        ${FLAMEGPU_ROOT}/include/flamegpu/visualiser/color/AutoPalette.h
    )
    SET(SRC_FLAMEGPU
        ${SRC_FLAMEGPU}
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/ModelVis.cpp    
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/AgentVis.cpp     
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/AgentStateVis.cpp
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/StaticModelVis.cpp
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/LineVis.cpp
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/color/DiscreteColor.cpp
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/color/StaticColor.cpp
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/color/HSVInterpolation.cpp
        ${FLAMEGPU_ROOT}/src/flamegpu/visualiser/color/ViridisInterpolation.cpp
    )
endif()
SET(ALL_SRC
    ${SRC_INCLUDE}
    ${SRC_FLAMEGPU}
    ${SRC_DYNAMIC}
)
SET(FLAMEGPU_INCLUDE ${SRC_INCLUDE} CACHE INTERNAL "Include files required by FLAMEGPU RTC")

# Setup Visual Studio (and eclipse) filters
source_group(TREE ${FLAMEGPU_ROOT}/include/flamegpu PREFIX include FILES ${SRC_INCLUDE})
list(FILTER T_SRC INCLUDE REGEX ".*\.(h|hpp|cuh)$")
set(T_SRC_FLAMEGPU "${SRC_FLAMEGPU}")
list(FILTER T_SRC_FLAMEGPU INCLUDE REGEX ".*\.(h|hpp|cuh)$")
source_group(TREE ${FLAMEGPU_ROOT}/src/flamegpu PREFIX headers FILES ${T_SRC_FLAMEGPU})
set(T_SRC_FLAMEGPU "${SRC_FLAMEGPU}")
list(FILTER T_SRC_FLAMEGPU EXCLUDE REGEX ".*\.(h|hpp|cuh)$")
source_group(TREE ${FLAMEGPU_ROOT}/src/flamegpu PREFIX src FILES ${T_SRC_FLAMEGPU})
set(T_SRC_EXTERNAL "${SRC_EXTERNAL}")
list(FILTER T_SRC_EXTERNAL INCLUDE REGEX ".*\.(h|hpp|cuh)$")
source_group(TREE ${FLAMEGPU_ROOT}/include PREFIX external FILES ${T_SRC_EXTERNAL})
set(T_SRC_EXTERNAL "${SRC_EXTERNAL}")
list(FILTER T_SRC_EXTERNAL EXCLUDE REGEX ".*\.(h|hpp|cuh)$")
source_group(TREE ${FLAMEGPU_ROOT}/include PREFIX external FILES ${T_SRC_EXTERNAL})


# Create the library target and set various properties

# @todo - correctly set PUBLIC/PRIVATE/INTERFACE for the flamegpu library target

# Define which source files are required for the target executable
add_library(${PROJECT_NAME} STATIC ${ALL_SRC})

# Add the library headers as public so they are forwarded on.
target_include_directories(${PROJECT_NAME} PUBLIC "${FLAMEGPU_ROOT}/include")
# Add any private headers.
target_include_directories(${PROJECT_NAME} PRIVATE "${FLAMEGPU_ROOT}/src")

# Set target level warnings.
EnableFLAMEGPUCompilerWarnings(TARGET "${PROJECT_NAME}")
# Apply common compiler settings
CommonCompilerSettings(TARGET "${PROJECT_NAME}")
# Set the cuda gencodes, potentially using the user-provided CUDA_ARCH
SetCUDAGencodes(TARGET "${PROJECT_NAME}")

# enable "fpic" for linux to allow shared libraries to be build from the static library (required for swig)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Activate visualisation if requested
if (VISUALISATION)
    # @todo - these could/should be private, but must be PUBLIC for swig to build. Alternateively it could be PRIVATE but also set for swig.
    target_link_libraries(${PROJECT_NAME} PUBLIC flamegpu_visualiser)
    CMAKE_SET_TARGET_FOLDER(flamegpu_visualiser "FLAMEGPU")
    target_compile_definitions(${PROJECT_NAME} PUBLIC VISUALISATION)

    # Make the visualisers GLM accessible via include
    # @todo - make the vis cmake/glm create a target to use.
    if (USE_GLM)
        if(glm_FOUND)
            target_include_directories(${PROJECT_NAME} PUBLIC "${glm_INCLUDE_DIRS}")
            target_compile_definitions(${PROJECT_NAME} PUBLIC GLM_PATH="${glm_INCLUDE_DIRS}")
        else()
            message(WARNING "USE_GLM enabled, but glm_FOUND is False.")
        endif()
        target_compile_definitions(${PROJECT_NAME} PRIVATE USE_GLM)
    endif()   
endif()

if (NOT RTC_DISK_CACHE)
    # Only used in source files, so can be private
    target_compile_definitions(${PROJECT_NAME} PRIVATE DISABLE_RTC_DISK_CACHE)
endif()    
if (EXPORT_RTC_SOURCES)
    # Only used in source files, so can be private
    target_compile_definitions(${PROJECT_NAME} PRIVATE OUTPUT_RTC_DYNAMIC_FILES)
endif ()

# Enable RDC
set_property(TARGET ${PROJECT_NAME}  PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link against dependency targets / directories.

# SYSTEM includes prevent warnings from 3rd party includes where possible. This is implied by target_link_libraries if it is an imported target.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # CUB (and thrust) cause many compiler warnings at high levels, including Wreorder. 
    # CUB:CUB does not use -isystem to prevent the automatic -I<cuda_path>/include  from being more important, and the CUDA disributed CUB being used. 
    # Instead, if possible we pass the include directory directly rather than using the imported target.
    # And also pass {CUDAToolkit_INCLUDE_DIRS}/../include" as isystem so the include order is correct for isystem to work (a workaround for a workaround). The `../` is required to prevent cmake from removing the duplicate path.

    # Include CUB via isystem if possible (via _CUB_INCLUDE_DIR which may be subject to change), otherwise use it via target_link_libraries.
    # If we used the system CUB, via find_package (CUDA 11.3+ package the config) then this convoluted approach could be avoided. 
    if(DEFINED _CUB_INCLUDE_DIR)
        target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "${_CUB_INCLUDE_DIR}")
    else()
        target_link_libraries(${PROJECT_NAME} PUBLIC CUB::CUB)
    endif()
    target_include_directories(${PROJECT_NAME}  SYSTEM PUBLIC "${CUDAToolkit_INCLUDE_DIRS}/../include")  
else()
    # MSVC just includes cub via the CUB::CUB target as no isystem to worry about.
    target_link_libraries(${PROJECT_NAME} PUBLIC CUB::CUB)
    # Same for Thrust.
    # Visual studio 2015 needs to suppress deprecation messages from CUB/Thrust.
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.10)
        target_compile_definitions(${PROJECT_NAME} PUBLIC "CUB_IGNORE_DEPRECATED_CPP_DIALECT")
        target_compile_definitions(${PROJECT_NAME} PUBLIC "THRUST_IGNORE_DEPRECATED_CPP_DIALECT")
    endif()
endif()

# Thrust uses isystem if available
target_link_libraries(${PROJECT_NAME} PUBLIC Thrust::Thrust)

# tinyxml2 static library
target_link_libraries(${PROJECT_NAME} PRIVATE Tinyxml2::tinyxml2)

# If rapidjson was found, add it to the include dirs.
target_link_libraries(${PROJECT_NAME} PRIVATE RapidJSON::rapidjson)

target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::nvrtc)

target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cuda_driver)

if(USE_NVTX AND TARGET NVTX::nvtx)
    target_link_libraries(${PROJECT_NAME} PUBLIC NVTX::nvtx)
    # Get the version to set a definition value 
    get_target_property(nvtxversion NVTX::nvtx VERSION)
    target_compile_definitions(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:C,CXX,CUDA>:USE_NVTX=${nvtxversion}>")
    unset(nvtxversion)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC Jitify::jitify)
target_compile_definitions(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:C,CXX,CUDA>:JITIFY_PRINT_LOG>")

# If gcc, need to add linker flag for std::experimental::filesystem pre c++17
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(${PROJECT_NAME} PUBLIC stdc++fs)
endif()

# Link against dl if available. This used to be implied from somewhere...
if(CMAKE_DL_LIBS)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_DL_LIBS})
endif()

# Make GLM accessible via include. PUBLIC so this is usable by downstream projects
if (USE_GLM)
    if(TARGET GLM::glm)
        target_link_libraries(${PROJECT_NAME} PUBLIC GLM::glm)
        target_compile_definitions(${PROJECT_NAME} PUBLIC USE_GLM)
    else()
        message(WARNING "USE_GLM enabled, but glm was not found")
    endif()
endif()

# Ensure we link against pthread / windows equilvalent. This was previously implied
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# Flag the new linter target and the files to be linted.
new_linter_target(${PROJECT_NAME} "${ALL_SRC}")

# Put within FLAMEGPU filter
CMAKE_SET_TARGET_FOLDER(${PROJECT_NAME} "FLAMEGPU")
# Put the tinyxml2 in the folder
CMAKE_SET_TARGET_FOLDER("tinyxml2" "FLAMEGPU/Dependencies")

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
