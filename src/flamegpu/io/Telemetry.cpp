#include "flamegpu/io/Telemetry.h"

#include <cstdlib>
#include <cerrno>
#include <algorithm>
#include <memory>
#include <array>
#include <iostream>
#include <cstdio>
#include <cctype>


#include "flamegpu/version.h"
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/util/Environment.h"



namespace flamegpu {
namespace io {
namespace Telemetry {

const char TELEMETRY_ENDPOINT[] = "https://nom.telemetrydeck.com/v1/";

namespace {

const char TELEMETRY_APP_ID[] = "94AC5E3F-F674-4E29-BF87-DAF4BA7F8F79";

}  // Anonymous namespace

std::string generateTelemetryData(std::string event_name, std::map<std::string, std::string> payload_items) {
    const std::string var_testmode = "$TEST_MODE";
    const std::string var_appID = "$APP_ID";
    const std::string var_buildHash = "$BUILD_HASH";
    const std::string var_eventName = "$EVENT_TYPE";
    const std::string var_payload = "$PAYLOAD";

    // check ENV for test variable FLAMEGPU_TEST_ENVIRONMENT
    std::string testmode;
    if (flamegpu::util::isTestEnvironment())
        testmode = "true";
    else
        testmode = "false";
    std::string appID = TELEMETRY_APP_ID;
    std::string buildHash = flamegpu::BUILD_HASH;

    // Differentiate pyflamegpu in the payload via the SWIG compiler macro, which we only define when building for pyflamegpu.
    // A user could potenitally static link against a build using that macro, but that's not a use-case we are currently concerned with.
    #ifdef SWIG
        std::string py_version = "pyflamegpu" + std::string(flamegpu::VERSION_STRING);
        payload_items["appVersion"] = py_version;  // e.g. 'pyflamegpu2.0.0-alpha.3' (graphed in Telemetry deck)
    #else  // SWIG
        payload_items["appVersion"] = flamegpu::VERSION_STRING;  // e.g. '2.0.0-alpha.3' (graphed in Telemetry deck)
    #endif  // SWIG

    // other version strings
    payload_items["appVersionFull"] = flamegpu::VERSION_FULL;
    payload_items["majorSystemVersion"] = std::to_string(flamegpu::VERSION_MAJOR);        // e.g. '2' (graphed in Telemetry deck)
    std::string major_minor_patch = std::to_string(flamegpu::VERSION_MAJOR) + "." + std::to_string(flamegpu::VERSION_MINOR) + "." + std::to_string(flamegpu::VERSION_PATCH);
    payload_items["majorMinorSystemVersion"] = major_minor_patch;                         // e.g. '2.0.0' (graphed in Telemetry deck)
    payload_items["appVersionPatch"] = std::to_string(flamegpu::VERSION_PATCH);
    payload_items["appVersionPreRelease"] = flamegpu::VERSION_PRERELEASE;
    payload_items["buildNumber"] = flamegpu::VERSION_BUILDMETADATA;  // e.g. '0553592f' (graphed in Telemetry deck)

    // OS
#ifdef _WIN32
    payload_items["operatingSystem"] = "Windows";
#elif __linux__
    payload_items["operatingSystem"] = "Linux";
#elif __unix__
    payload_items["operatingSystem"] = "Unix";
#else
    payload_items["operatingSystem"] = "Other";
#endif
    // VIS
#ifdef VISUALISATION
    payload_items["Visualisation"] = "true";
#else
    payload_items["Visualisation"] = "false";
#endif

    // create the payload
    std::string payload = "";
    bool first = true;
    // iterate payload to generate the payload array
    for (const auto& [key, value] : payload_items) {
        std::string item_str;
        if (!first)
            item_str = ",\"" + key + ":" + value + "\"";
        else
            item_str = "\"" + key + ":" + value + "\"";
        payload.append(item_str);
        first = false;
    }

    // create the telemetry json package
    std::string telemetry_data = R"json(
    [{
        "isTestMode": "$TEST_MODE",
        "appID": "$APP_ID",
        "clientUser": "$BUILD_HASH",
        "sessionID": "",
        "type" : "$EVENT_TYPE",
        "payload" : [$PAYLOAD]
    }])json";
    // update the placeholders
    telemetry_data.replace(telemetry_data.find(var_testmode), var_testmode.length(), testmode);
    telemetry_data.replace(telemetry_data.find(var_appID), var_appID.length(), appID);
    telemetry_data.replace(telemetry_data.find(var_buildHash), var_buildHash.length(), buildHash);
    telemetry_data.replace(telemetry_data.find(var_eventName), var_eventName.length(), event_name);
    telemetry_data.replace(telemetry_data.find(var_payload), var_payload.length(), payload);
    // murder the formatting into a single compact line
    telemetry_data.erase(std::remove(telemetry_data.begin(), telemetry_data.end(), '\n'), telemetry_data.end());  // Remove newlines and replace with space
    telemetry_data.erase(std::remove(telemetry_data.begin(), telemetry_data.end(), '\t'), telemetry_data.end());  // Remove tabs and replace with space
    telemetry_data.erase(std::remove_if(telemetry_data.begin(), telemetry_data.end(), [](char c) {
            return std::isspace(static_cast<unsigned char>(c));
        }), telemetry_data.end());  // Remove spaces
    size_t pos = 0;
    while ((pos = telemetry_data.find("\"", pos)) != std::string::npos) {   // Use escape characters
        telemetry_data.replace(pos, 1, "\\\"");
        pos += 2;
    }

    return telemetry_data;
}

bool sendTelemetryData(std::string telemetry_data) {
    // Maximum duration curl to attempt to connect to the endpoint
    const float CURL_CONNECT_TIMEOUT = 2.0;
    // Maximum total duration for the curl call, including connection and payload
    const float CURL_MAX_TIME = 4.0;
    // Silent curl command (-s) and redirect response output to null
    std::string null;
#if _WIN32
    null = "nul";
#else
    null = "/dev/null";
#endif
    std::string curl_command = "curl -s -o " + null + " --connect-timeout " + std::to_string(CURL_CONNECT_TIMEOUT) + " --max-time " + std::to_string(CURL_MAX_TIME) +  " -X POST \"" + std::string(TELEMETRY_ENDPOINT) + "\" -H \"Content-Type: application/json; charset=utf-8\" --data-raw \"" + telemetry_data + "\"";

    // capture the return value
    if (std::system(curl_command.c_str()) != EXIT_SUCCESS)
        return false;

    return true;
}

void hintTelemetryUsage() {
    float random_normalised = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    // Dont print telemetry hints in test environment
    if ((random_normalised < PROBABILITY_TELEMETERY_HINT) && !(flamegpu::util::isTestEnvironment()) && !(flamegpu::util::hasEnvironmentVariable("FLAMEGPU_SILENCE_TELEMETRY_NOTICE"))) {
        fprintf(stdout, "NOTICE: The FLAME GPU software is reliant on evidence to support is continued development. Please "
                        "consider enabling SHARE_USAGE_STATISTICS in your CMake options or calling 'shareUsageStatistics()' on "
                        "your Simulation or Ensemble object. This message is randomly (p=%f) displayed but can be silenced "
                        "using '--quiet' or by setting 'SILENCE_TELEMETRY_NOTICE' in your environment.\n", PROBABILITY_TELEMETERY_HINT);
    }
}

bool silenceTelemetryNotice() {
    return flamegpu::util::setEnvironmentVariable("FLAMEGPU_SILENCE_TELEMETRY_NOTICE", "True");
}

bool globalTelemetryEnabled() {
    bool cmake_shared_usage_stats = false;
#ifdef FLAMEGPU_SHARE_USAGE_STATISTICS
    cmake_shared_usage_stats = true;
#endif
    std::string env_val_str = flamegpu::util::getEnvironmentVariable("FLAMEGPU_SHARE_USAGE_STATISTICS");
    // Return false if explicitly disabled
    if ((env_val_str == "0") || (env_val_str == "False") || (env_val_str == "FALSE") || (env_val_str == "Off") || (env_val_str == "OFF"))
        return false;
    // True if either are set
    return (cmake_shared_usage_stats || flamegpu::util::hasEnvironmentVariable("FLAMEGPU_SHARE_USAGE_STATISTICS"));
}

}  // namespace Telemetry
}  // namespace io
}  // namespace flamegpu
