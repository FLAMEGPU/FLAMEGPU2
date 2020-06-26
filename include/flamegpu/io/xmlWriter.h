#ifndef INCLUDE_FLAMEGPU_IO_XMLWRITER_H_
#define INCLUDE_FLAMEGPU_IO_XMLWRITER_H_

/**
 * @file xmlwriter.h
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#include <memory>
#include <string>
#include <unordered_map>

#include "flamegpu/io/statewriter.h"
#include "flamegpu/model/ModelDescription.h"

// Derived classes
class xmlWriter : public StateWriter {
 public:
    /**
     * Returns a writer capable of writing model state to an XML file
     * Environment properties from the Simulation instance pointed to by 'sim_instance_id' will be used 
     * Agent data will be read from 'model_state'
     * @param model_name Name from the model description hierarchy of the model to be exported
     * @param sim_instance_id Instance is from the Simulation instance to export the environment properties from
     * @param model_state Map of AgentPopulation to read the agent data from per agent, key should be agent name
     * @param iterations The value from the step counter at the time of export.
     * @param output_file Filename of the input file (This will be used to determine which reader to return)
     */
    xmlWriter(const std::string &model_name, const unsigned int &sim_instance_id, const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model_state, const unsigned int &iterations, const std::string &output_file);
    /**
     * Actually perform the writing to file
     * @return Always tinyxml2::XML_SUCCESS
     * @throws TinyXMLError If export of the model state fails
     */
    int writeStates();
};

#endif  // INCLUDE_FLAMEGPU_IO_XMLWRITER_H_
