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
     * Virtual destructor for correct inheritance behaviour
     */
    virtual ~StateWriter() {}
    /**
     * Returns true if beginWrite() has been called and writing is active
     */
    virtual bool isWriting() = 0;
    /**
     * Starts writing to the specified file in the specified mode
     * @param output_file Filename of the input file (This will be used to determine which reader to return)
     * @param pretty_print Whether the output file should be "pretty" or minified.
     * @throws Throws exception if beginWrite() is called a second time before endWrite() has been called
     * @see endWrite()
     */
    virtual void beginWrite(const std::string &output_file, bool pretty_print) = 0;
    /**
     * Saves the current file and ends the writing state
     * @see beginWrite(const std::string &, bool)
     * @throws Throws exception if file IO fails
     * @throws Throws exception if beginWrite() has not been called so writing is not active
     */
    virtual void endWrite() = 0;

    // -----------------------------------------------------------------------
    //  The Easy Interface
    // -----------------------------------------------------------------------
    /**
     * Export the full simulation state
     * @param sim_instance Instance of the Simulation object (This is used for setting/getting config)
     * @param iterations The value from the step counter at the time of export.
     * @param env_manager Environment manager containing env property data for this sim instance
     * @param macro_env Macro environment of the model
     * @param agents_map Map of AgentVector to read the agent data from per agent, key should be agent name
     * @throws Exceptions will be thrown on failure, specific to the subclass/problem
     */
    void writeFullModelState(
        const Simulation *sim_instance,
        unsigned int iterations,
        const std::shared_ptr<const detail::EnvironmentManager>& env_manager,
        const std::shared_ptr<const detail::CUDAMacroEnvironment>& macro_env,
        const util::StringPairUnorderedMap<std::shared_ptr<const AgentVector>>& agents_map) {
        writeConfig(sim_instance);
        writeStats(iterations);
        writeEnvironment(env_manager);
        writeMacroEnvironment(macro_env);
        writeAgents(agents_map);
    }
    // -----------------------------------------------------------------------
    //  The Advanced Interface
    // -----------------------------------------------------------------------
    virtual void writeConfig(const Simulation *sim_instance) = 0;
    virtual void writeStats(unsigned int iterations) = 0;
    virtual void writeEnvironment(const std::shared_ptr<const detail::EnvironmentManager>& env_manager) = 0;
    /**
     * Write the macro environment block
     * @param macro_env The macro environment to pull properties from
     * @param filter If provided, only named properties will be written. Note, if filter contains missing properties it will fail
     */
    virtual void writeMacroEnvironment(const std::shared_ptr<const detail::CUDAMacroEnvironment>& macro_env, std::initializer_list<std::string> filter = {}) = 0;
    virtual void writeAgents(const util::StringPairUnorderedMap<std::shared_ptr<const AgentVector>>& agents_map) = 0;
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_STATEWRITER_H_
