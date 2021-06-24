#ifndef INCLUDE_FLAMEGPU_IO_STATEREADERFACTORY_H_
#define INCLUDE_FLAMEGPU_IO_STATEREADERFACTORY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <algorithm>

#include "flamegpu/io/StateReader.h"
#include "flamegpu/io/XMLStateReader.h"
#include "flamegpu/io/JSONStateReader.h"
#include "flamegpu/util/StringPair.h"
#include "flamegpu/util/StringUint32Pair.h"
#include "flamegpu/util/filesystem.h"

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
     * Environment properties will be read into the Simulation instance pointed to by 'sim_instance_id'
     * Agent data will be read into 'model_state'
     * @param model_name Name from the model description hierarchy of the model to be loaded
     * @param env_desc Environment description for validating property data on load
     * @param env_init Dictionary of loaded values map:<{name, index}, value>
     * @param model_state Map of AgentVector to load the agent data into per agent, key should be agent name
     * @param input Filename of the input file (This will be used to determine which reader to return)
     * @param sim_instance Instance of the Simulation object (This is used for setting/getting config)
     * @throws UnsupportedFileType If the file extension does not match an appropriate reader
     */
    static StateReader* createReader(
        const std::string& model_name,
        const std::unordered_map<std::string, EnvironmentDescription::PropData>& env_desc,
        util::StringUint32PairUnorderedMap<util::Any>& env_init,
        util::StringPairUnorderedMap<std::shared_ptr<AgentVector>>& model_state,
        const std::string& input,
        Simulation* sim_instance) {
        const std::string extension = util::filesystem::getFileExt(input);

        if (extension == "xml") {
            return new XMLStateReader(model_name, env_desc, env_init, model_state, input, sim_instance);
        } else if (extension == "json") {
            return new JSONStateReader(model_name, env_desc, env_init, model_state, input, sim_instance);
        }
        THROW UnsupportedFileType("File '%s' is not a type which can be read "
            "by StateReaderFactory::createReader().",
            input.c_str());
    }
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_STATEREADERFACTORY_H_
