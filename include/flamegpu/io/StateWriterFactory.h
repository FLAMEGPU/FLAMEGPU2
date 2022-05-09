#ifndef INCLUDE_FLAMEGPU_IO_STATEWRITERFACTORY_H_
#define INCLUDE_FLAMEGPU_IO_STATEWRITERFACTORY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <algorithm>

#include "flamegpu/io/StateWriter.h"
#include "flamegpu/io/XMLStateWriter.h"
#include "flamegpu/io/JSONStateWriter.h"
#include "flamegpu/io/JSONLogger.h"
#include "flamegpu/io/XMLLogger.h"
#include "flamegpu/util/StringPair.h"

namespace flamegpu {

class AgentVector;

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
     * @param model_name Name from the model description hierarchy of the model to be exported
     * @param sim_instance_id Instance is from the Simulation instance to export the environment properties from
     * @param model_state Map of AgentVector to read the agent data from per agent, key should be agent name
     * @param iterations The value from the step counter at the time of export.
     * @param output_file Filename of the input file (This will be used to determine which reader to return)
     * @param sim_instance Instance of the Simulation object (This is used for setting/getting config)
     * @throws exception::UnsupportedFileType If the file extension does not match an appropriate reader
     */
    static StateWriter* createWriter(
        const std::string& model_name,
        const unsigned int& sim_instance_id,
        const util::StringPairUnorderedMap<std::shared_ptr<AgentVector>>& model_state,
        const unsigned int& iterations,
        const std::string& output_file,
        const Simulation* sim_instance) {
        const std::string extension = std::filesystem::path(output_file).extension().string();

        if (extension == ".xml") {
            return new XMLStateWriter(model_name, sim_instance_id, model_state, iterations, output_file, sim_instance);
        } else if (extension == ".json") {
            return new JSONStateWriter(model_name, sim_instance_id, model_state, iterations, output_file, sim_instance);
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
