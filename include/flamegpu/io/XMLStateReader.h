#ifndef INCLUDE_FLAMEGPU_IO_XMLSTATEREADER_H_
#define INCLUDE_FLAMEGPU_IO_XMLSTATEREADER_H_

#include <memory>
#include <string>

#include "flamegpu/io/StateReader.h"

namespace flamegpu {
namespace io {
/**
 * XML format StateReader
 */
class XMLStateReader : public StateReader {
 public:
    /**
     * Loads the specified XML file to an internal data-structure 
     * @param input_file Path to file to be read
     * @param model Model description to ensure file loaded is suitable
     * @param verbosity Verbosity level to use during load
     */
    void parse(const std::string &input_file, const std::shared_ptr<const ModelData> &model, Verbosity verbosity) override;

 private:
    /**
     * Flamegpu1 xml input files are allowed to omit state
     * This function extracts the initial state for the named agent from model_state;
     */
    static std::string getInitialState(const std::shared_ptr<const ModelData> &model, const std::string& agent_name);
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_XMLSTATEREADER_H_
