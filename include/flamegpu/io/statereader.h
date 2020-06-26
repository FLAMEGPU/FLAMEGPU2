#ifndef INCLUDE_FLAMEGPU_IO_STATEREADER_H_
#define INCLUDE_FLAMEGPU_IO_STATEREADER_H_

/**
 * @file statereader.h
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#include <memory>
#include <string>
#include <unordered_map>

#include "flamegpu/model/ModelDescription.h"

// Base class
class StateReader {
 public:
    /**
     * Constructs a reader capable of reading model state from a specific format (this class is abstract)
     * Environment properties will be read into the Simulation instance pointed to by 'sim_instance_id'
     * Agent data will be read into 'model_state'
     * @param _model_name Name from the model description hierarchy of the model to be loaded
     * @param _sim_instance_id Instance is from the Simulation instance to load the environment properties into
     * @param _model_state Map of AgentPopulation to load the agent data into per agent, key should be agent name
     * @param input Filename of the input file (This will be used to determine which reader to return)
     */
    StateReader(const std::string &_model_name, const unsigned int &_sim_instance_id, const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &_model_state, const std::string &input)
    : model_state(_model_state)
    , inputFile(input)
    , model_name(_model_name)
    , sim_instance_id(_sim_instance_id)  {}
    ~StateReader() {}

    // -----------------------------------------------------------------------
    //  The interface
    // -----------------------------------------------------------------------
    /**
     * Actually perform the file load
     * @return Returns a return code
     * @todo: This should probably be the same return code between subclasses, and seems redundant with our exceptions as should never return fail.
     */
    virtual int parse() = 0;

    // void setFileName(const char* input) {    inputFile = std::string(input); }

    // void setModelDesc(const ModelDescription &model_desc) {    model_description_ = model_desc; }
/*
    StateReader& create(const ModelDescription &model, const char *input) {
        string extension = getFileExt(input);
        StateReader *object_to_return = nullptr;

        if (extension == "xml") {
            object_to_return = new xmlReader(model);
        }
        if (extension == "bin") {
            object_to_return = new binReader(model);
        }

        return *object_to_return;
    }
*/

 protected:
    const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model_state;
    std::string inputFile;
    const std::string model_name;
    const unsigned int sim_instance_id;
};

#endif  // INCLUDE_FLAMEGPU_IO_STATEREADER_H_
