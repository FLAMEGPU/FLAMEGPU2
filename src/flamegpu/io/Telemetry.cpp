#include "flamegpu/io/Telemetry.h"

#include <cstdlib>
#include <cerrno>
#include <algorithm>
#include <memory>
#include <array>
#include <iostream>
#include <cstdio>
#include <cctype>
#include <sstream>

#include "flamegpu/version.h"

namespace flamegpu {
namespace io {

namespace {
    // the FLAMEGPU2 telemetry app id.
    const char TELEMETRY_APP_ID[] = "94AC5E3F-F674-4E29-BF87-DAF4BA7F8F79";

    // Flag tracking if telemetry is enabled, initialised subject to preproc and env vars in initialiseFromEnvironmentIfNeeded
    static bool enabled = false;
    // Flag indicating if users have suppressed telemetry warnings
    static bool suppressed = false;
    // Flag indicating if FLAMEGPU telemetry test mode is enabled.
    static bool testMode = false;
    // Flag indicating if these anon namespace statics have been enabled or not
    static bool initialised = false;
    // Flag indicating if the user has been notified / encouraged to enable usage statistics or not
    static bool haveNotified = false;

    /**
     * Convert a string to a boolean using CMake's truthy/falsey values
     * @return if the value is truthy or falsey in CMake's opinion.
     */
    bool cmakeStrToBool(const char * input) {
        // Assume truthy
        bool rtn = true;
        if (input != NULL) {
            std::string s = std::string(input);
            // Trim leading whitespace
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !std::isspace(ch); }));
            // Trim trailing whitespace
            s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base(), s.end());

            // Transform the input to lower case
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<unsigned char>(std::tolower(c)); });
            // If it's a falsey option, set it to falesy.
            if (s == "0" || s == "false" || s == "off") {
                rtn = false;
            }
        }
        return rtn;
    }

    /*
     * Initialise namespace scoped static variables based from environment variables, if not done already.
     This is done in a method to avoid potential UB with initialisation order of static members and the system environment, which maybe undefined based on SO posts / is not just a direct mapping.
    */
    void initialiseFromEnvironmentIfNeeded() {
        if (!initialised) {
            // Enabled by default
            enabled = true;
            // If the sharing environment var is set, parse it following cmake boolean logic
            char * env_FLAMEGPU_SHARE_USAGE_STATISTICS = std::getenv("FLAMEGPU_SHARE_USAGE_STATISTICS");
            if (env_FLAMEGPU_SHARE_USAGE_STATISTICS != NULL) {
                enabled = cmakeStrToBool(env_FLAMEGPU_SHARE_USAGE_STATISTICS);
            } else {
                // if the environment variable is not specified, use the value from the preprocessor
                #ifdef FLAMEGPU_SHARE_USAGE_STATISTICS
                    enabled = true;
                #else
                    enabled = false;
                #endif
            }
            // Parse env and cmake variables to find the default value for suppression.
            suppressed = false;
            char * env_FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE = std::getenv("FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE");
            if (env_FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE != NULL) {
                suppressed = cmakeStrToBool(env_FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE);
            } else {
                // if the environment variable is not specified, use the value from the preprocessor
                #ifdef FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE
                    suppressed = true;
                #else
                    suppressed = false;
                #endif
            }
            // Parse env and cmake variables to find the default value for test dev mode. .
            testMode = false;
            char * env_FLAMEGPU_TELEMETRY_TEST_MODE = std::getenv("FLAMEGPU_TELEMETRY_TEST_MODE");
            if (env_FLAMEGPU_TELEMETRY_TEST_MODE != NULL) {
                testMode = cmakeStrToBool(env_FLAMEGPU_TELEMETRY_TEST_MODE);
            } else {
                // if the environment variable is not specified, use the value from the preprocessor
                #ifdef FLAMEGPU_TELEMETRY_TEST_MODE
                    testMode = true;
                #else
                    testMode = false;
                #endif
            }
            // Mark this as initialised.
            initialised = true;
        }
    }

}  // anonymous namespace

void Telemetry::enable() {
    // Initialise from the env var is needed. A ctor might be nicer.
    initialiseFromEnvironmentIfNeeded();
    // set to enabled.
    enabled = true;
}

void Telemetry::disable() {
    // initialise from the env var if needed
    initialiseFromEnvironmentIfNeeded();
    // set to disabled.
    enabled = false;
}

bool Telemetry::isEnabled() {
    // initialise from the env var if needed
    initialiseFromEnvironmentIfNeeded();
    // return if it is enabled or not.
    return enabled;
}


void Telemetry::suppressNotice() {
    // initialise from the env var if needed
    initialiseFromEnvironmentIfNeeded();
    // set the suppressed flag in the anon namespace
    suppressed = true;
}

bool Telemetry::isTestMode() {
    // initialise from the env var if needed
    initialiseFromEnvironmentIfNeeded();
    // return if it is enabled or not.
    return testMode;
}


std::string Telemetry::generateData(std::string event_name, std::map<std::string, std::string> payload_items, bool isSWIG) {
    // Initialise from the env var is needed. A ctor might be nicer.
    initialiseFromEnvironmentIfNeeded();

    const std::string var_testmode = "$TEST_MODE";
    const std::string var_appID = "$APP_ID";
    const std::string var_telemetryRandomID = "$TELEMETRY_RANDOM_ID";
    const std::string var_eventName = "$EVENT_TYPE";
    const std::string var_payload = "$PAYLOAD";

    // check ENV for test variable FLAMEGPU_TEST_ENVIRONMENT
    std::string testmode = isTestMode() ? "true" : "false";
    std::string appID = TELEMETRY_APP_ID;
    std::string telemetryRandomID = flamegpu::TELEMETRY_RANDOM_ID;

    // Differentiate pyflamegpu in the payload via the SWIG compiler macro, which we only define when building for pyflamegpu.
    // A user could potentially static link against a build using that macro, but that's not a use-case we are currently concerned with.
    if (isSWIG) {
        std::string py_version = "pyflamegpu" + std::string(flamegpu::VERSION_STRING);
        payload_items["appVersion"] = py_version;  // e.g. 'pyflamegpu2.0.0-alpha.3' (graphed in Telemetry deck)
    } else {
        payload_items["appVersion"] = flamegpu::VERSION_STRING;  // e.g. '2.0.0-alpha.3' (graphed in Telemetry deck)
    }

    // other version strings
    payload_items["appVersionFull"] = flamegpu::VERSION_FULL;
    payload_items["majorSystemVersion"] = std::to_string(flamegpu::VERSION_MAJOR);  // e.g. '2' (graphed in Telemetry deck)
    std::string major_minor_patch = std::to_string(flamegpu::VERSION_MAJOR) + "." + std::to_string(flamegpu::VERSION_MINOR) + "." + std::to_string(flamegpu::VERSION_PATCH);
    payload_items["majorMinorSystemVersion"] = major_minor_patch;  // e.g. '2.0.0' (graphed in Telemetry deck)
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
    // visualiastion status
#ifdef FLAMEGPU_VISUALISATION
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
        "clientUser": "$TELEMETRY_RANDOM_ID",
        "sessionID": "",
        "type" : "$EVENT_TYPE",
        "payload" : [$PAYLOAD]
    }])json";
    // update the placeholders
    telemetry_data.replace(telemetry_data.find(var_testmode), var_testmode.length(), testmode);
    telemetry_data.replace(telemetry_data.find(var_appID), var_appID.length(), appID);
    telemetry_data.replace(telemetry_data.find(var_telemetryRandomID), var_telemetryRandomID.length(), telemetryRandomID);
    telemetry_data.replace(telemetry_data.find(var_eventName), var_eventName.length(), event_name);
    telemetry_data.replace(telemetry_data.find(var_payload), var_payload.length(), payload);
    // Remove newlines and replace with space
    telemetry_data.erase(std::remove(telemetry_data.begin(), telemetry_data.end(), '\n'), telemetry_data.end());
    // Remove tabs and replace with space
    telemetry_data.erase(std::remove(telemetry_data.begin(), telemetry_data.end(), '\t'), telemetry_data.end());
     // Remove spaces
    telemetry_data.erase(std::remove_if(telemetry_data.begin(), telemetry_data.end(), [](char c) {
            return std::isspace(static_cast<unsigned char>(c));
        }), telemetry_data.end());
    // Use escape characters
    size_t pos = 0;
    while ((pos = telemetry_data.find("\"", pos)) != std::string::npos) {
        telemetry_data.replace(pos, 1, "\\\"");
        pos += 2;
    }
    return telemetry_data;
}

bool Telemetry::sendData(std::string telemetry_data) {
    // Initialise from the env var is needed. A ctor might be nicer.
    initialiseFromEnvironmentIfNeeded();

    // Maximum duration curl to attempt to connect to the endpoint
    const float CURL_CONNECT_TIMEOUT = 0.5;
    // Maximum total duration for the curl call, including connection and payload
    const float CURL_MAX_TIME = 1.0;
    // Silent curl command (-s) and redirect response output to null
    std::string null;
#if _WIN32
    null = "nul";
#else
    null = "/dev/null";
#endif
    std::stringstream curl_command;
    curl_command << "curl";
    curl_command << " -s";
    curl_command << " -o " << null;
    curl_command << " --connect-timeout " << std::to_string(CURL_CONNECT_TIMEOUT);
    curl_command << " --max-time " << std::to_string(CURL_MAX_TIME);
    curl_command << " -X POST \"" << std::string(TELEMETRY_ENDPOINT) << "\"";
    curl_command << " -H \"Content-Type: application/json; charset=utf-8\"";
    curl_command << " --data-raw \"" << telemetry_data + "\"";
    curl_command << " > " << null << " 2>&1";
    // capture the return value
    if (std::system(curl_command.str().c_str()) != EXIT_SUCCESS) {
        return false;
    }

    return true;
}

void Telemetry::encourageUsage() {
    // Only print the usage encouragement notice if it has not already been encouraged, telemetry is not enabled and telemetry has not been suppressed.
    if (!haveNotified && !suppressed && !isEnabled()) {
        fprintf(stdout,
            "NOTICE: The FLAME GPU software is reliant on evidence to support is continued development. Please "
            "consider enabling FLAMEGPU_SHARE_USAGE_STATISTICS during CMake configuration, "
            "setting FLAMEGPU_SHARE_USAGE_STATISTICS to true as an environment variable, "
            "or setting the Simulation/Ensemble config telemetry property to true.\n"
            "This message can be silenced by suppressing all output (--quiet), "
            "calling flamegpu::io::Telemetry::suppressNotice, or defining a system environment variable FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE\n");
        // Set the flag that this has already been emitted once during execution of the current binary file, so it doesn't happen again.
        haveNotified = true;
    }
}

}  // namespace io
}  // namespace flamegpu
