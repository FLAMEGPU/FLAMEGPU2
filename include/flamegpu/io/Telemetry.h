#ifndef INCLUDE_FLAMEGPU_IO_TELEMETRY_H_
#define INCLUDE_FLAMEGPU_IO_TELEMETRY_H_

#include <string>
#include <map>

namespace flamegpu {

// forward declare friendship classes
class CUDASimulation;
class CUDAEnsemble;

namespace io {

/**
 * Class for interacting with the telemetry, using static methods.
 * This is a class rather than a namespace to prevent users from directly calling some methods which must be accessible from CUDASimulation and CUDAEnsemble.
*/
class Telemetry {
    // Mark friend classes to allow access to methods not intended for users to call directly.
    friend class flamegpu::CUDASimulation;
    friend class flamegpu::CUDAEnsemble;
 public:
/**
 * Opt-in to sending anonymous usage telemetry information, if currently disabled.
 * This controls the default value used for CUDASimulation and CUDAEnsemble configuration objects, which can then independently be opted out.
 */
static void enable();

/**
 * Opt-out of sending anonymous usage telemetry information, if currently enabled.
 * This controls the default value used for CUDASimulation and CUDAEnsemble configuration objects, which can then independently be opted out.
 */
static void disable();

/**
 * Get the current enabled/disabled status of telemetry.
 * If the system environment variable FLAMEGPU_SHARE_USAGE_STATISTICS is defined, and is false-y (0, false, False, FALSE, off, Off, OFF) it will be disabled by default.
 * Otherwise, the CMake FLAMEGPU_SHARE_USAGE_STATISTICS option will be used, which defaults to On/True.
 * Otherwise, if the define was not specified at build time, it will default to enabled.
 * @return if telemetry is currently enabled or disabled.
 */
static bool isEnabled();

/**
 * If telemetry is not enabled, a notice will be emitted to encourage users to enable this as a way to support FLAMEGPU development, once per application run. This method can be called to disable that message from being printed.
 * I.e. this is used within the test suite(s).
 */
static void suppressNotice();

 protected:
/*
 * The remote endpoint which telemetry is pushed to.
 */
constexpr static char TELEMETRY_ENDPOINT[] = "https://nom.telemetrydeck.com/v1/";

/**
 * Generates the telemetry data packet as a string.
 * Function is used by sendTelemetryData but is useful for returning the actual json for transparency.
 * See documentation for data which is sent and why
 * @param event_name the name of the event to record. This will either be "simulation-run, ensemble-run"
 * @param payload_items a map of key value items to embed in the payload of the telemetry packet
 * @param isSWIG True if a swig build of flamegpu
 * @return The json string that should be sent via sendTelemetryData
 */
static std::string generateData(std::string event_name, std::map<std::string, std::string> payload_items, bool isSWIG);

/**
 * Sends telemetry data in the form as the provided json to the TelemetryDeck web service.
 * @param telemetry_data json data to send
 * @return false if failed for any reason (including inability to reach host)
 */
static bool sendData(std::string telemetry_data);

/**
 * Prints a notice that telemetry is helpful to the development of the software.
 * Notice will not be printed if telemetry is disabled, and the notice has not been suppressed.
 */
static void encourageUsage();

/**
 * Get the current test mode status of telemetry, which will be false unless the system environment variable FLAMEGPU_TELEMETRY_TEST_MODE is defined and not a false-y value, or if the CMake option FLAMEGPU_TELEMETRY_TEST_MODE was set to ON.
 * Otherwise, if the define was not specified at build time, it will default to enabled.
 * @return if telemetry is currently in test mode or not.
 */
static bool isTestMode();

/**
 * Gets a cross platform location for storing user configuration data. Required to store a per user Id. On windows this uses the AppData folder (CSIDL_APPDATA) and in Linux this uses XDG_CONFIG_HOME.
 * @return Root directory for storing configuration files
 */
static std::string getConfigDirectory();

/**
 * Generates a randomised 36 character alphanumeric string for use as a User Id. 
 * @return A 36 character randomised alphanumeric string
 */
static std::string generateRandomId();

/**
 * Obtains a unique user Id. If a configuration file (i.e. ${XDG_CONFIG_HOME}/flamegpu/telemetry_user.cfg on linux) exists this will be loaded from disk otherwise it will be generated and stored in the configuration location. If the configuration location is not writeable a new user Id will be generated each time. The user Id will be further obfuscated by Telemetry Deck which will salt and hash the Id.
 * @return A 36 character randomised alphanumeric string representing a unique user Id
 */
static std::string getUserId();
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_TELEMETRY_H_
