#ifndef INCLUDE_FLAMEGPU_IO_TELEMETRY_H_
#define INCLUDE_FLAMEGPU_IO_TELEMETRY_H_

#include <string>
#include <map>

namespace flamegpu {

namespace io {

namespace Telemetry {

/**
 * The telemetry endpoint in which data is sent. This is via the TelemetryDeck web service.
 */
extern const char TELEMETRY_ENDPOINT[];

/**
 * The probability that a user will be shown a notice regardiung enabling usage statictics and supporting the software
 */
static constexpr float PROBABILITY_TELEMETERY_HINT = 0.1f;

/**
 * Generates the telemetry data packet as a string.
 * Function is used by sendTelemetryData but is useful for returning the actual json for transparency.
 * See docuemtation for data which is sent and why
 * @param event_name the name of the event to record. This will either be "simulation-run, ensemble-run, googletest-run, pythontest-run"
 * @param payload_items a map of key value items to embed in the payload of the telemetry packet
 * @return The json string that should be sent via sendTelemetryData
 */
std::string generateTelemetryData(std::string event_name, std::map<std::string, std::string> payload_items);

/**
 * Sends telemetry data in the form as the provided json to the TelemetryDeck web service. 
 * @param json json data to send
 * @return false if failed for any reason (including inability to reach host)
 */
bool sendTelemetryData(std::string json);

/**
 * Prints a notice that telemetry is helpful to the development of the software. The notice is displayed with probability=PROBABILITY_TELEMETERY_HINT.
 * Notice will not be printed if;
 * - flamegpu::util::isTestEnvironment() is true (i.e. If FLAMEGPU_TEST_ENVIRONMENT exists in environment) or,
 * - SILENCE_TELEMETRY_NOTICE is defined in the environment
 */
void hintTelemetryUsage();

/**
 * Silences any notices about enabling telemetry by setting SILENCE_TELEMETRY_NOTICE environment variable. Used in google test to prevent poluting expected outputs.
 * @return True if successfull
 */
bool silenceTelemetryNotice();

/**
 * Returns true if CMake FLAMEGPU_SHARE_USAGE_STATISTICS is true or if FLAMEGPU_SHARE_USAGE_STATISTICS Environment variable is set (to anything other than 0, Off or False values)
 * @return True global telemetry enabled
 */
bool globalTelemetryEnabled();

}  // namespace Telemetry
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_TELEMETRY_H_
