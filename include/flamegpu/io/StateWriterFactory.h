#ifndef INCLUDE_FLAMEGPU_IO_STATEWRITERFACTORY_H_
#define INCLUDE_FLAMEGPU_IO_STATEWRITERFACTORY_H_

#include <string>
#include <algorithm>
#include <filesystem>

#include "flamegpu/io/StateWriter.h"
#include "flamegpu/io/XMLStateWriter.h"
#include "flamegpu/io/JSONStateWriter.h"

namespace flamegpu {
namespace io {

/**
 * Factory for creating instances of StateWriter
 */
class StateWriterFactory {
 public:
    /**
     * Returns a writer capable of writing model state to 'output_file'
     * Environment properties from the Simulation instance pointed to by 'sim_instance_id' will be used
     * Agent data will be read from 'model_state'
     * @param output_file Filename of the input file (This will be used to determine which reader to return)
     * @throws exception::UnsupportedFileType If the file extension does not match an appropriate reader
     */
    static StateWriter* createWriter(
        const std::string& output_file) {
        const std::string extension = std::filesystem::path(output_file).extension().string();

        if (extension == ".xml") {
            return new XMLStateWriter();
        } else if (extension == ".json") {
            return new JSONStateWriter();
        } else if (extension.empty()) {
            THROW exception::InvalidFilePath("Filepath '%s' contains unsuitable characters or lacks a file extension, "
                "in StateWriterFactory::createLogger().", output_file.c_str());
        }
        THROW exception::UnsupportedFileType("File '%s' is not a type which can be written "
            "by StateWriterFactory::createWriter().",
            output_file.c_str());
    }
    /**
     * Return a clean file extension from the provided string
     * If the file extension is not supported empty string is returned instead
     */
    static std::string detectSupportedFileExt(const std::string &user_file_ext) {
        std::string rtn = user_file_ext;
        // Move entire string to lower case
        std::transform(rtn.begin(), rtn.end(), rtn.begin(), [](unsigned char c) { return std::use_facet< std::ctype<char>>(std::locale()).tolower(c); });
        // Strip first character if it is '.'
        if (rtn[0] == '.')
          rtn = rtn.substr(1);
        // Compare against supported formats
        if (rtn == "xml" ||
            rtn == "json") {
            return rtn;
        }
        return "";
    }
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_STATEWRITERFACTORY_H_
