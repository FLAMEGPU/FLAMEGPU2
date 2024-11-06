#include "flamegpu/io/StateReader.h"

#include <vector>
#include <unordered_map>
#include <string>

namespace flamegpu {
namespace io {

void StateReader::resetCache() {
    simulation_config.clear();
    cuda_config.clear();
    env_init.clear();
    macro_env_init.clear();
    agents_map.clear();
}
void StateReader::getFullModelState(
    Simulation::Config &s_cfg,
    std::unordered_map<std::string, detail::Any> &environment_init,
    std::unordered_map<std::string, std::vector<char>> &macro_environment_init,
    util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> &agents_init) {
    getSimulationConfig(s_cfg);
    getEnvironment(environment_init);
    getMacroEnvironment(macro_environment_init);
    getAgents(agents_init);
}

#define MAP_GET(out, map, name, typ) out.name = map.find(#name) == map.end()?out.name:std::any_cast<typ>(map.at(#name))

void StateReader::getSimulationConfig(Simulation::Config &cfg) {
    if (input_filepath.empty()) {
        THROW exception::InvalidOperation("Input file has not been parsed, in StateReader::getSimulationConfig()");
    }
    // if (!simulation_config) {
    //     THROW exception::InvalidInputFile("Input file %s did not contain an simulation config, in StateReader::getSimulationConfig()", input_filepath.c_str());
    // }
    // Set all the items manually
    MAP_GET(cfg, simulation_config, input_file, std::string);
    MAP_GET(cfg, simulation_config, step_log_file, std::string);
    MAP_GET(cfg, simulation_config, exit_log_file, std::string);
    MAP_GET(cfg, simulation_config, common_log_file, std::string);
    MAP_GET(cfg, simulation_config, truncate_log_files, bool);
    MAP_GET(cfg, simulation_config, random_seed, uint64_t);
    MAP_GET(cfg, simulation_config, steps, unsigned int);
    MAP_GET(cfg, simulation_config, verbosity, Verbosity);
    MAP_GET(cfg, simulation_config, timing, bool);
    MAP_GET(cfg, simulation_config, silence_unknown_args, bool);
    MAP_GET(cfg, simulation_config, telemetry, bool);
#ifdef FLAMEGPU_VISUALISATION
    MAP_GET(cfg, simulation_config, console_mode, bool);
#endif
}
void StateReader::getCUDAConfig(CUDASimulation::Config &cfg) {
    if (input_filepath.empty()) {
        THROW exception::InvalidOperation("Input file has not been parsed, in StateReader::getCUDAConfig()");
    }
    // if (!cuda_config) {
    //     THROW exception::InvalidInputFile("Input file %s did not contain an CUDA config, in StateReader::getCUDAConfig()", input_filepath.c_str());
    // }
    // Set all the items manually
    MAP_GET(cfg, cuda_config, device_id, int);
    MAP_GET(cfg, cuda_config, inLayerConcurrency, bool);
}
void StateReader::getEnvironment(std::unordered_map<std::string, detail::Any> &environment_init) {
    if (input_filepath.empty()) {
        THROW exception::InvalidOperation("Input file has not been parsed, in StateReader::getEnvironment()");
    }
    // if (env_init.empty()) {
    //     THROW exception::InvalidInputFile("Input file %s did not contain any environment properties, in StateReader::getEnvironment()", input_filepath.c_str());
    // }
    for (const auto& [key, val] : env_init) {
        environment_init.erase(key);
        environment_init.emplace(key, val);
    }
}
void StateReader::getMacroEnvironment(std::unordered_map<std::string, std::vector<char>> &macro_environment_init) {
    if (input_filepath.empty()) {
        THROW exception::InvalidOperation("Input file has not been parsed, in StateReader::getEnvironment()");
    }
    // if (macro_env_init.empty()) {
    //     THROW exception::InvalidInputFile("Input file %s did not contain any macro environment properties, in StateReader::getMacroEnvironment()", input_filepath.c_str());
    // }
    for (const auto& [key, val] : macro_env_init) {
        macro_environment_init.insert_or_assign(key, val);
    }
}
void StateReader::getAgents(util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> &agents_init) {
    if (input_filepath.empty()) {
        THROW exception::InvalidOperation("Input file has not been parsed, in StateReader::getEnvironment()");
    }
    // if (agents_map.empty()) {
    //     THROW exception::InvalidInputFile("Input file %s did not contain any agents, in StateReader::getMacroEnvironment()", input_filepath.c_str());
    // }
    for (const auto& [key, val] : agents_map) {
        agents_init.insert_or_assign(key, val);
    }
}

}  // namespace io
}  // namespace flamegpu
