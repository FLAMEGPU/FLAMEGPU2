#ifndef INCLUDE_FLAMEGPU_DETAIL_TESTSUITETELEMETRY_H_
#define INCLUDE_FLAMEGPU_DETAIL_TESTSUITETELEMETRY_H_

#include <cstdio>
#include <string>

#include "flamegpu/io/Telemetry.h"

namespace flamegpu {
namespace detail {

/**
 * detail / internal class for test suite result telemetry/reporting, which extends flamegpu::io::Telemetry to gain access to the private member methods it requires.
 */
class TestSuiteTelemetry : private flamegpu::io::Telemetry {
 public:
/**
 * Send the results of a test suite run to TelemtryHub, callable from the google test or pytest test suites.
 * @param reportName String identifying which test suite this was.
 * @param outcome String indicating if the test suite passed or not ("Passed/Failed")
 * @param total The total number of tests discovered
 * @param selected The number of tests selected to run (i.e. not filtered out)
 * @param skipped The number of skipped or disabled tests
 * @param passed The number of successful tests
 * @param verbose flag indicating if the telemetry packet should be printed to stdout or not.
 * @return success of the telemetry push request
*/
static bool sendResults(std::string reportName, std::string outcome, unsigned int total, unsigned int selected, unsigned int skipped, unsigned int passed, unsigned int failed, bool verbose);
};

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_TESTSUITETELEMETRY_H_
