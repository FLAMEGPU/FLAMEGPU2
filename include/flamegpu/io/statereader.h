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
    // -----------------------------------------------------------------------
    //  Constructors and Destructor
    // -----------------------------------------------------------------------
    StateReader(const std::string &_model_name, const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &_model_state, const std::string &input)
    : model_state(_model_state)
    , inputFile(input)
    , model_name(_model_name) {}
    ~StateReader() {}

    // -----------------------------------------------------------------------
    //  The interface
    // -----------------------------------------------------------------------

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
};

#endif  // INCLUDE_FLAMEGPU_IO_STATEREADER_H_
