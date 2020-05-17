#include "flamegpu/exception/FGPUException.h"

#include <cstdio>
#include <cstring>
#include <sstream>
#include <limits>

const char *FGPUException::file = nullptr;
unsigned int FGPUException::line = std::numeric_limits<unsigned int>::max();

FGPUException::FGPUException()
    : err_message("") {
    if (file) {
        std::stringstream ss;
        ss << file << "(" << line << "): ";
        err_message.append(ss.str());
        file = nullptr;
    }
}

const char *FGPUException::what() const noexcept {
    return err_message.c_str();
}

void FGPUException::setLocation(const char *_file, const unsigned int &_line) {
    file = _file;
    line = _line;
}


std::string FGPUException::parseArgs(const char * format, va_list argp) {
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
