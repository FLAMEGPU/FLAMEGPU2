#include "flamegpu/util/Environment.h"

#include <stdlib.h>

// Implementation of setenv for windows (https://stackoverflow.com/questions/17258029/c-setenv-undefined-identifier-in-visual-studio)
#if _WIN32
int setenv(const char* name, const char* value, int overwrite) {
    int errcode = 0;
    if (!overwrite) {
        size_t envsize = 0;
        errcode = getenv_s(&envsize, NULL, 0, name);
        if (errcode || envsize) return errcode;
    }
    return _putenv_s(name, value);
}
#endif

namespace flamegpu {
namespace util {

std::string getEnvironmentVariable(std::string variable_name) {
    char* env_val;
    env_val = std::getenv(variable_name.c_str());
    if (env_val == NULL)
        return std::string("");
    else
        return std::string(env_val);
}


bool hasEnvironmentVariable(std::string variable_name) {
    char* env_val;
    env_val = std::getenv(variable_name.c_str());
    if (env_val == NULL)
        return false;
    // return true if value is not "0"
    std::string env_val_str = env_val;
    return ((env_val_str != "0") && (env_val_str != "False") && (env_val_str != "FALSE") && (env_val_str != "Off") && (env_val_str != "OFF"));
}

bool setEnvironmentVariable(std::string variable_name, std::string variable_value) {
    if (setenv(variable_name.c_str(), variable_value.c_str(), true) != 0)
        return false;
    return true;
}

bool isTestEnvironment() {
    return hasEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT");
}

bool setTestEnvironment() {
    return setEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT", "True");
}

}  // namespace util
}  // namespace flamegpu
