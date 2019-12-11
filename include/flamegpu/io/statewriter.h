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

#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/model/ModelDescription.h"

class StateWriter {
 public:
    // -----------------------------------------------------------------------
    //  Constructors and Destructor
    // -----------------------------------------------------------------------

    StateWriter(const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &_model_state, const char* output) : model_state(_model_state), outputFile(std::string(output)) {}
    ~StateWriter() {}

    // -----------------------------------------------------------------------
    //  The interface
    // -----------------------------------------------------------------------

    virtual int writeStates() = 0;

 protected:
    const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> model_state;
    std::string outputFile;
};

#endif  // INCLUDE_FLAMEGPU_IO_STATEWRITER_H_
