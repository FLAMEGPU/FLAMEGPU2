#ifndef INCLUDE_FLAMEGPU_IO_STATEWRITER_H_
#define INCLUDE_FLAMEGPU_IO_STATEWRITER_H_

/**
 * @file statewriter.h
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#include <string>

#include <flamegpu/exception/FGPUException.h>
#include "flamegpu/model/ModelDescription.h"

class StateWriter {
 public:
    // -----------------------------------------------------------------------
    //  Constructors and Destructor
    // -----------------------------------------------------------------------

    StateWriter(const ModelDescription &model, const char* output) : model_description_(model), outputFile(std::string(output)) {}
    ~StateWriter() {}

    // -----------------------------------------------------------------------
    //  The interface
    // -----------------------------------------------------------------------

    virtual int writeStates() = 0;

 protected:
    ModelDescription model_description_;
    std::string outputFile;
};


#endif // INCLUDE_FLAMEGPU_IO_STATEWRITER_H_
