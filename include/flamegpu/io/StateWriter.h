#ifndef INCLUDE_FLAMEGPU_IO_STATEWRITER_H_
#define INCLUDE_FLAMEGPU_IO_STATEWRITER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/util/StringPair.h"

namespace flamegpu {
namespace detail {
class EnvironmentManager;
class CUDAMacroEnvironment;
}  // namespace detail
class AgentVector;
class Simulation;

namespace io {

/**
 * Abstract representation of a class for exporting model data (agent population data, environment properties, run configuration) to file
 * @see XMLStateWriter The XML implementation of a StateWriter
 * @see JSONStateWriter The JSON implementation of a StateWriter
 */
class StateWriter {
 public:
    /**
     * Returns a writer capable of writing model state to a specific format (this class is abstract)
     * Agent data will be read from 'model_state'
     * @param _model_name Name from the model description hierarchy of the model to be exported
     * @param _env_manager Environment manager containing env property data for this sim instance
     * @param _model_state Map of AgentVector to read the agent data from per agent, key should be agent name
     * @param _macro_env Macro environment of the model
     * @param _iterations The value from the step counter at the time of export.
     * @param output_file Filename of the input file (This will be used to determine which reader to return)
     * @param _sim_instance Instance of the simulation (for configuration data IO)
     */
    StateWriter(std::string _model_name,
        std::shared_ptr<detail::EnvironmentManager> _env_manager,
        const util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> &_model_state,
        std::shared_ptr<const detail::CUDAMacroEnvironment> _macro_env,
        const unsigned int _iterations,
        std::string output_file,
        const Simulation *_sim_instance)
    : model_state(_model_state)
    , iterations(_iterations)
    , outputFile(std::move(output_file))
    , model_name(std::move(_model_name))
    , env_manager(std::move(_env_manager))
    , macro_env(std::move(_macro_env))
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
    const std::shared_ptr<detail::EnvironmentManager> env_manager;
    const std::shared_ptr<const detail::CUDAMacroEnvironment> macro_env;
    const Simulation *sim_instance;
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_STATEWRITER_H_
