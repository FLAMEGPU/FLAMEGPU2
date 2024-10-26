#ifndef INCLUDE_FLAMEGPU_IO_STATEREADERFACTORY_H_
#define INCLUDE_FLAMEGPU_IO_STATEREADERFACTORY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <filesystem>
#include <vector>

#include "flamegpu/io/StateReader.h"
#include "flamegpu/io/XMLStateReader.h"
#include "flamegpu/io/JSONStateReader.h"
#include "flamegpu/util/StringPair.h"

namespace flamegpu {
class AgentVector;

namespace io {

/**
 * Factory for creating instances of StateReader
 */
class StateReaderFactory {
 public:
    /**
     * Returns a reader capable of reading 'input'
     * @param input Filename of the input file (This will be used to determine which reader to return)
     * @throws exception::UnsupportedFileType If the file extension does not match an appropriate reader
     */
    static StateReader* createReader(
        const std::string& input) {
        const std::string extension = std::filesystem::path(input).extension().string();

        if (extension == ".xml") {
            return new XMLStateReader();
        } else if (extension == ".json") {
            return new JSONStateReader();
        } else if (extension.empty()) {
                THROW exception::InvalidFilePath("Filepath '%s' contains unsuitable characters or lacks a file extension, "
                    "in StateReaderFactory::createLogger().", input.c_str());
        }
        THROW exception::UnsupportedFileType("File '%s' is not a type which can be read "
            "by StateReaderFactory::createReader().",
            input.c_str());
    }
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_STATEREADERFACTORY_H_
