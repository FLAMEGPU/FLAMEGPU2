#include "flamegpu/exception/FLAMEGPUException.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <limits>

namespace flamegpu {
namespace exception {

const char * FLAMEGPUException::file = nullptr;
unsigned int FLAMEGPUException::line = std::numeric_limits<unsigned int>::max();

FLAMEGPUException::FLAMEGPUException()
    : err_message("") {
    if (file) {
        std::stringstream ss;
        ss << file << "(" << line << "): ";
        err_message.append(ss.str());
        file = nullptr;
    }
}

const char * FLAMEGPUException::what() const noexcept {
    return err_message.c_str();
}

void FLAMEGPUException::setLocation(const char *_file, const unsigned int _line) {
    file = _file;
    line = _line;
}


std::string FLAMEGPUException::parseArgs(const char * format, va_list argp) {
    if (!argp)
        return format;
    std::string rtn = format;
    // Create a copy of the va_list, as vsnprintf can invalidate elements of argp and find the required buffer length
    va_list argpCopy;
    va_copy(argpCopy, argp);
    const int buffLen = vsnprintf(nullptr, 0, format, argpCopy) + 1;
    va_end(argpCopy);
    char *buffer = reinterpret_cast<char *>(malloc(buffLen * sizeof(char)));
    // Populate the buffer with the original va_list
    int ct = vsnprintf(buffer, buffLen, format, argp);
    if (ct >= 0) {
        // Success!
        buffer[buffLen - 1] = '\0';
        rtn = std::string(buffer);
    }
    free(buffer);
    return rtn;
}

}  // namespace exception
}  // namespace flamegpu
