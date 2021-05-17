#ifndef INCLUDE_FLAMEGPU_IO_STATEWRITER_H_
#define INCLUDE_FLAMEGPU_IO_STATEWRITER_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/util/StringPair.h"

namespace flamegpu {

class AgentVector;

/**
 * Abstract representation of a class for exporting model data (agent population data, environment properties, run configuration) to file
 * @see XMLWriter The XML implementation of a StateWriter
 * @see JSONWriter The JSON implementation of a StateWriter
 */
class StateWriter {
 public:
    /**
     * Returns a writer capable of writing model state to a specific format (this class is abstract)
     * Environment properties from the Simulation instance pointed to by 'sim_instance_id' will be used 
     * Agent data will be read from 'model_state'
     * @param _model_name Name from the model description hierarchy of the model to be exported
     * @param _sim_instance_id Instance is from the Simulation instance to export the environment properties fromo
     * @param _model_state Map of AgentVector to read the agent data from per agent, key should be agent name
     * @param _iterations The value from the step counter at the time of export.
     * @param output_file Filename of the input file (This will be used to determine which reader to return)
     * @param _sim_instance Instance of the simulation (for configuration data IO)
     */
    StateWriter(const std::string &_model_name,
        const unsigned int &_sim_instance_id,
        const util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> &_model_state,
        const unsigned int &_iterations,
        const std::string &output_file,
        const Simulation *_sim_instance)
    : model_state(_model_state)
    , iterations(_iterations)
    , outputFile(output_file)
    , model_name(_model_name)
    , sim_instance_id(_sim_instance_id)
    , sim_instance(_sim_instance) {}
    /**
     * Virtual destructor for correct inheritance behaviour
     */
    virtual ~StateWriter() {}

    // -----------------------------------------------------------------------
    //  The interface
    // -----------------------------------------------------------------------
    /**
     * Actually perform the file export
     * @param prettyPrint Whether to include indentation and line breaks to aide human reading
     * @return Returns a return code
     * @todo: This should probably be the same return code between subclasses, and seems redundant with our exceptions as should never return fail.
     */
    virtual int writeStates(bool prettyPrint) = 0;

 protected:
    const util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> model_state{};
    unsigned int iterations;
    std::string outputFile;
    const std::string model_name;
    const unsigned int sim_instance_id;
    const Simulation *sim_instance;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_STATEWRITER_H_
