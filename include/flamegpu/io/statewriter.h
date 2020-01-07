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

#include <memory>
#include <string>
#include <unordered_map>

#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/model/ModelDescription.h"

class StateWriter {
 public:
    // -----------------------------------------------------------------------
    //  Constructors and Destructor
    // -----------------------------------------------------------------------

    StateWriter(const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &_model_state, const unsigned int &_iterations, const char* output_file)
        : model_state(_model_state), iterations(_iterations), outputFile(std::string(output_file)) {}
    ~StateWriter() {}

    // -----------------------------------------------------------------------
    //  The interface
    // -----------------------------------------------------------------------

    virtual int writeStates() = 0;

 protected:
    const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> model_state;
    unsigned int iterations;
    std::string outputFile;
};

#endif  // INCLUDE_FLAMEGPU_IO_STATEWRITER_H_
