#ifndef INCLUDE_FLAMEGPU_IO_JSONSTATEREADER_H_
#define INCLUDE_FLAMEGPU_IO_JSONSTATEREADER_H_

#include <memory>
#include <string>

#include "flamegpu/io/StateReader.h"

namespace flamegpu {
namespace io {

/**
 * JSON format StateReader
 */
class JSONStateReader : public StateReader {
 public:
    /**
    * Loads the specified XML file to an internal data-structure 
    * @param input_file Path to file to be read
    * @param model Model description to ensure file loaded is suitable
    * @param verbosity Verbosity level to use during load
    */
    void parse(const std::string &input_file, const std::shared_ptr<const ModelData> &model, Verbosity verbosity) override;
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_JSONSTATEREADER_H_
