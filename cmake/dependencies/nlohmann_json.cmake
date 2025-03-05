#################
# nlohmann/json #
#################

include(FetchContent)
cmake_policy(SET CMP0079 NEW)
# Temporary CMake >= 3.30 fix https://github.com/FLAMEGPU/FLAMEGPU2/issues/1223
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()
FetchContent_Declare(
    nlohmann_json
    URL            "https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz"
)
FetchContent_GetProperties(nlohmann_json)
if(NOT nlohmann_json_POPULATED)
    FetchContent_Populate(nlohmann_json)
    # Add the project as a subdirectory
    add_subdirectory(${nlohmann_json_SOURCE_DIR} ${nlohmann_json_BINARY_DIR} EXCLUDE_FROM_ALL)
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
