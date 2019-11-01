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

#include <string>
#include "flamegpu/model/ModelDescription.h"

// Base class
class StateReader {
 public:
    // -----------------------------------------------------------------------
    //  Constructors and Destructor
    // -----------------------------------------------------------------------
    StateReader(const ModelDescription &model, const char* input): model_description_(model), inputFile(std::string(input)) {}
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
    ModelDescription model_description_;
    std::string inputFile;
};

#endif // INCLUDE_FLAMEGPU_IO_STATEREADER_H_
