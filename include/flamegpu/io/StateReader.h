#ifndef INCLUDE_FLAMEGPU_IO_STATEREADER_H_
#define INCLUDE_FLAMEGPU_IO_STATEREADER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/util/StringPair.h"
#include "flamegpu/util/StringUint32Pair.h"

namespace flamegpu {

class AgentVector;

namespace io {

/**
 * Abstract representation of a class for importing model data (agent population data, environment properties, run configuration) from file
 * @see XMLStateReader The XML implementation of a StateReader
 * @see JSONStateReader The JSON implementation of a StateReader
 */
class StateReader {
 public:
    /**
     * Constructs a reader capable of reading model state from a specific format (this class is abstract)
     * Environment properties will be read into the Simulation instance pointed to by 'sim_instance_id'
     * Agent data will be read into 'model_state'
     * @param _model_name Name from the model description hierarchy of the model to be loaded
     * @param _env_desc Environment description for validating property data on load
     * @param _env_init Dictionary of loaded values map:<{name, index}, value>
     * @param _model_state Map of AgentVector to load the agent data into per agent, key should be agent name
     * @param input Filename of the input file (This will be used to determine which reader to return)
     * @param _sim_instance Instance of the simulation (for configuration data IO)
     */
    StateReader(
        const std::string& _model_name,
        const std::unordered_map<std::string, EnvironmentDescription::PropData>& _env_desc,
        std::unordered_map<std::string, util::Any>& _env_init,
        util::StringPairUnorderedMap<std::shared_ptr<AgentVector>>& _model_state,
        const std::string& input,
        Simulation* _sim_instance)
    : model_state(_model_state)
    , inputFile(input)
    , model_name(_model_name)
    , env_desc(_env_desc)
    , env_init(_env_init)
    , sim_instance(_sim_instance) {}
    /**
     * Virtual destructor for correct inheritance behaviour
     */
    virtual ~StateReader() {}

    // -----------------------------------------------------------------------
    //  The interface
    // -----------------------------------------------------------------------
    /**
     * Actually perform the file load
     * @return Returns a return code
     * @todo: This should probably be the same return code between subclasses, and seems redundant with our exceptions as should never return fail.
     */
    virtual int parse() = 0;

 protected:
    util::StringPairUnorderedMap<std::shared_ptr<AgentVector>>& model_state;
    std::string inputFile;
    const std::string model_name;
    const std::unordered_map<std::string, EnvironmentDescription::PropData> &env_desc;
    std::unordered_map<std::string, util::Any>& env_init;
    Simulation *sim_instance;
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_STATEREADER_H_
