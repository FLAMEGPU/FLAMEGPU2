#include <cstdio>

#include "flamegpu/detail/TestSuiteTelemetry.h"

namespace flamegpu {
namespace detail {

bool TestSuiteTelemetry::sendResults(std::string reportName, std::string outcome, unsigned int total, unsigned int selected, unsigned int skipped, unsigned int passed, unsigned int failed, bool verbose, bool isSWIG) {
    // Construct the payload
    std::map<std::string, std::string> telemetry_payload;
    telemetry_payload["TestOutcome"] = outcome;
    telemetry_payload["TestsTotal"] = std::to_string(total);
    telemetry_payload["TestsSelected"] = std::to_string(selected);
    telemetry_payload["TestsSkipped"] = std::to_string(skipped);
    telemetry_payload["TestsPassed"] = std::to_string(passed);
    telemetry_payload["TestsFailed"] = std::to_string(failed);
    // generate telemetry data
    std::string telemetry_data = flamegpu::io::Telemetry::generateData(reportName, telemetry_payload, isSWIG);
    // send telemetry
    bool telemetrySuccess = flamegpu::io::Telemetry::sendData(telemetry_data);
    // print telemetry payload to the user if requested
    if (verbose) {
        if (telemetrySuccess) {
            fprintf(stdout, "Telemetry packet sent to '%s' json was: %s\n", flamegpu::io::Telemetry::TELEMETRY_ENDPOINT, telemetry_data.c_str());
            fflush(stdout);
        } else {
            fprintf(stderr, "Warning: Usage statistics for Test suite failed to send.\n");
        }
    }
    return true;
}

}  // namespace detail
}  // namespace flamegpu
