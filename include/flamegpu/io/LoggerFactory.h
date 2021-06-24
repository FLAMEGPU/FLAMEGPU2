#ifndef INCLUDE_FLAMEGPU_IO_LOGGERFACTORY_H_
#define INCLUDE_FLAMEGPU_IO_LOGGERFACTORY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <algorithm>

#include "flamegpu/io/Logger.h"
#include "flamegpu/io/JSONLogger.h"
#include "flamegpu/io/XMLLogger.h"
#include "flamegpu/util/filesystem.h"

namespace flamegpu {
namespace io {
/**
 * Factory for creating instances of flamegpu::Logger
 */
class LoggerFactory {
 public:
    /**
     * @param output_path File for the log to be output to, this will be used to determine the logger type
     * @param prettyPrint If false, the output data will be in a compact/minified format which may not be very readable
     * @param truncateFile If true and output file already exists, it will be truncated
     */
    static std::unique_ptr<Logger> createLogger(const std::string &output_path, bool prettyPrint, bool truncateFile = true) {
        const std::string extension = util::filesystem::getFileExt(output_path);

        if (extension == "xml") {
            return std::make_unique<XMLLogger>(output_path, prettyPrint, truncateFile);
        } else if (extension == "json") {
            return std::make_unique<JSONLogger>(output_path, prettyPrint, truncateFile);
        }
        THROW UnsupportedFileType("File '%s' is not a type which can be written "
            "by StateWriterFactory::createLogger().",
            output_path.c_str());
    }
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_LOGGERFACTORY_H_
