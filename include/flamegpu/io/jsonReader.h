#ifndef INCLUDE_FLAMEGPU_IO_JSONREADER_H_
#define INCLUDE_FLAMEGPU_IO_JSONREADER_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "flamegpu/io/statereader.h"
#include "flamegpu/model/ModelDescription.h"

// Derived classes
class jsonReader : public StateReader {
 public:
    /**
     * Constructs a reader capable of reading model state from JSON files
     * Environment properties will be read into the Simulation instance pointed to by 'sim_instance_id'
     * Agent data will be read into 'model_state'
     * @param model_name Name from the model description hierarchy of the model to be loaded
     * @param sim_instance_id Instance is from the Simulation instance to load the environment properties into
     * @param model_state Map of AgentPopulation to load the agent data into per agent, key should be agent name
     * @param input_file Filename of the input file (This will be used to determine which reader to return)
     * @param sim_instance Instance of the Simulation object (This is used for setting/getting config)
     */
    jsonReader(
        const std::string &model_name,
        const unsigned int &sim_instance_id,
        const std::unordered_map<std::string,
        std::shared_ptr<AgentPopulation>> &model_state,
        const std::string &input_file,
        Simulation *sim_instance);
    /**
     * Actual performs the XML parsing to load the model state
     * @return Always 0
     * @throws RapidJSONError If parsing of the input file fails
     */
    int parse();
};

#endif  // INCLUDE_FLAMEGPU_IO_JSONREADER_H_
