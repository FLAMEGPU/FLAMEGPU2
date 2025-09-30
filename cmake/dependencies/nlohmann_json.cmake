#################
# nlohmann/json #
#################

include(FetchContent)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    list(APPEND DEPENDENCY_ARGS EXCLUDE_FROM_ALL)
endif()

FetchContent_Declare(
    nlohmann_json
    URL            "https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz"
    ${DEPENDENCY_ARGS}
)
FetchContent_GetProperties(nlohmann_json)
# Download and add_subdirectory
FetchContent_MakeAvailable(nlohmann_json)
# If we are using CUDA 12.0 or 12.1 on linux or windows, add an interface definition of JSON_HAS_RANGES to 0 to the nlohmann target to workaround an nvcc bug. See https://github.com/nlohmann/json/issues/3907.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.2 AND TARGET nlohmann_json)
    target_compile_definitions(nlohmann_json INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:JSON_HAS_RANGES=0>)
endif()

# Mark some CACHE vars advanced for a cleaner GUI
mark_as_advanced(FETCHCONTENT_QUIET)
mark_as_advanced(FETCHCONTENT_BASE_DIR)
mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_NLOHMANN_JSON)
mark_as_advanced(JSON_BuildTests)
mark_as_advanced(JSON_CI)
mark_as_advanced(JSON_Diagnostics)
mark_as_advanced(JSON_DisableEnumSerialization)
mark_as_advanced(JSON_GlobalUDLs)
mark_as_advanced(JSON_ImplicitConversions)
mark_as_advanced(JSON_Install)
mark_as_advanced(JSON_LegacyDiscardedValueComparison)
mark_as_advanced(JSON_MultipleHeaders)
mark_as_advanced(JSON_SystemInclude)
