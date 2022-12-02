#ifndef INCLUDE_FLAMEGPU_UTIL_ENVIRONMENT_H_
#define INCLUDE_FLAMEGPU_UTIL_ENVIRONMENT_H_

/**
 * Provides utility methods for interactiong with environment variables
 */

#include <string>

namespace flamegpu {
namespace util {

/**
 * Utility function to get an environment variable
 * @param variable_name the Environment variable name
 * @return Return a string of the enviroinment variable value or an empty string if not found
 */
std::string getEnvironmentVariable(std::string variable_name);

/**
 * Utility function to test if an environment variable exists (and that is has a non zero, non False value)
 * @param variable_name the Environment variable name
 * @return Returns true if environment varibale exists (and is value is not "0", "False", "FALSE", "Off" or "OFF")
 */
bool hasEnvironmentVariable(std::string variable_name);

/**
 * Utility function to set an environment variable
 * @param variable_name the Environment variable name
 * @param variable_value the Environment variable value
 * @return Return true if set false if the operation failed for any reason
 */
bool setEnvironmentVariable(std::string variable_name, std::string variable_value);

/**
 * Determines if the current envrinment has FLAMEGPU_TEST_ENVIRONMENT defined 
 * Test enviroinment is for use by developers. Telemetry is still sent but is flagged as test data.
 * @return True if FLAMEGPU_TEST_ENVIRONMENT is defined
 */
bool isTestEnvironment();

/**
 * Sets the FLAMEGPU_TEST_ENVIRONMENT environment varibale to True 
 * Test enviroinment is for use by developers. Telemetry is still sent but is flagged as test data.
 * @return True if successfull
 */
bool setTestEnvironment();

}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_ENVIRONMENT_H_
