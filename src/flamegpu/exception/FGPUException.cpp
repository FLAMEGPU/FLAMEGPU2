#include "flamegpu/exception/FGPUException.h"

#include <cstdio>
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
    const int buffLen = vsnprintf(nullptr, 0, format, argp) + 1;
    char *buffer = reinterpret_cast<char *>(malloc(buffLen * sizeof(char)));
    int ct = vsnprintf(buffer, buffLen, format, argp);
    va_end(argp);
    if (ct >= 0) {
        // Success!
        buffer[buffLen - 1] = '\0';
        rtn = std::string(buffer);
    }
    free(buffer);
    return rtn;
}
