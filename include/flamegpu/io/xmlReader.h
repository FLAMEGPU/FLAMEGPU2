#ifndef INCLUDE_FLAMEGPU_IO_XMLREADER_H_
#define INCLUDE_FLAMEGPU_IO_XMLREADER_H_

/**
 * @file xmlreader.h
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "flamegpu/io/statereader.h"
#include "flamegpu/model/ModelDescription.h"

// Derived classes
class xmlReader : public StateReader {
 public:
    /**
     * Constructs a reader capable of reading model state from XML files
     * Environment properties will be read into the Simulation instance pointed to by 'sim_instance_id'
     * Agent data will be read into 'model_state'
     * @param model_name Name from the model description hierarchy of the model to be loaded
     * @param env_desc Environment description for validating property data on load
     * @param env_init Dictionary of loaded values map:<{name, index}, value>
     * @param model_state Map of AgentPopulation to load the agent data into per agent, key should be agent name
     * @param input_file Filename of the input file (This will be used to determine which reader to return)
     */
    xmlReader(
        const std::string &model_name,
        const std::unordered_map<std::string, EnvironmentDescription::PropData> &env_desc,
        std::unordered_map<std::pair<std::string, unsigned int>, Any> &env_init,
        const std::unordered_map<std::string,
        std::shared_ptr<AgentPopulation>> &model_state,
        const std::string &input_file,
        Simulation *sim_instance);
    /**
     * Actual performs the XML parsing to load the model state
     * @return Always tinyxml2::XML_SUCCESS
     * @throws TinyXMLError If parsing of the input file fails
     */
    int parse();
};

#endif  // INCLUDE_FLAMEGPU_IO_XMLREADER_H_
