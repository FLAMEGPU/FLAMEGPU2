#ifndef INCLUDE_FLAMEGPU_IO_STATEREADER_H_
#define INCLUDE_FLAMEGPU_IO_STATEREADER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <any>

#include "flamegpu/util/StringPair.h"
#include "flamegpu/simulation/Simulation.h"
#include "flamegpu/simulation/CUDASimulation.h"

namespace flamegpu {

class AgentVector;

namespace io {

/**
 * Abstract representation of a class for importing model data (agent population data, environment properties, run configuration) from file
 * @see XMLStateReader The XML implementation of a StateReader
 * @see JSONStateReader The JSON implementation of a StateReader
 */
class StateReader {
 public:
    /**
     * Virtual destructor for correct inheritance behaviour
     */
    virtual ~StateReader() {}

    /**
     * Loads the file to an internal data-structure
     * @param input_file Path to file to be read
     * @param model Model description to ensure file loaded is suitable
     * @param verbosity Verbosity level to use during load
     */
    virtual void parse(const std::string &input_file, const std::shared_ptr<const ModelData> &model, Verbosity verbosity) = 0;

    // -----------------------------------------------------------------------
    //  The Easy Interface
    // -----------------------------------------------------------------------
    /**
     * Grab the full simulation state from the input file
     * @note CUDASimulation Config is not included and should be requested separately
     */
    void getFullModelState(
        Simulation::Config &s_cfg,
        std::unordered_map<std::string, detail::Any> &environment_init,
        std::unordered_map<std::string, std::vector<char>> &macro_environment_init,
        util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> &agents_init);

    // -----------------------------------------------------------------------
    //  The Advanced Interface
    // -----------------------------------------------------------------------
    /**
     * Overwrite the provided simulation config with the one loaded from file
     * @param cfg The config struct to be overwritten
     * @throws If the parsed file did not contain a simulation config or a file has not been parsed
     */
    void getSimulationConfig(Simulation::Config &cfg);
    /**
    * Overwrite the provided CUDA config with the one loaded from file
    * @param cfg The config struct to be overwritten
    * @throws If the parsed file did not contain a CUDA config or a file has not been parsed
    */
    void getCUDAConfig(CUDASimulation::Config &cfg);
    void getEnvironment(std::unordered_map<std::string, detail::Any> &environment_init);
    void getMacroEnvironment(std::unordered_map<std::string, std::vector<char>> &macro_environment_init);
    void getAgents(util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> &agents_init);

 protected:
    std::string input_filepath;

    void resetCache();

    std::unordered_map<std::string, std::any> simulation_config;
    std::unordered_map<std::string, std::any> cuda_config;
    std::unordered_map<std::string, detail::Any> env_init;
    std::unordered_map<std::string, std::vector<char>> macro_env_init;
    util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> agents_map;
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_STATEREADER_H_
