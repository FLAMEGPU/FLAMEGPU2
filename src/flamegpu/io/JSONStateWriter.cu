#include "flamegpu/io/JSONStateWriter.h"

#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <set>
#include <algorithm>
#include <memory>
#include <map>
#include <functional>

#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/simulation/AgentVector.h"
#include "flamegpu/simulation/CUDASimulation.h"
#include "flamegpu/util/StringPair.h"
#include "flamegpu/simulation/detail/EnvironmentManager.cuh"
#include "flamegpu/simulation/detail/CUDAMacroEnvironment.h"

namespace flamegpu {
namespace io {
JSONStateWriter::JSONStateWriter()
    : StateWriter() {}
void JSONStateWriter::beginWrite(const std::string &output_file, bool pretty_print) {
    this->outputPath = output_file;
    if (isWriting()) {
        THROW exception::UnknownInternalError("Writing already active, in JSONStateWriter::beginWrite()");
    }
    outStream.open(output_file, std::ios::out|std::ios::binary|std::ios::trunc);
    if (!outStream.is_open()) {
        THROW exception::InvalidFilePath("Failed to open file '%s', in JSONStateWriter::beginWrite()", output_file.c_str());
    }
    if (pretty_print) {
        outStream << std::setw(4);
    }
    // Reset json cache
    j = {};
    // Clear flags
    this->config_written = false;
    this->stats_written = false;
    this->environment_written = false;
    this->macro_environment_written = false;
    this->agents_written = false;
}
void JSONStateWriter::endWrite() {
    if (!isWriting()) {
        THROW exception::UnknownInternalError("Writing not active, in JSONStateWriter::endWrite()");
    }

    outStream << j;
    outStream.close();
}

void JSONStateWriter::writeConfig(const Simulation *sim_instance) {
    if (!isWriting()) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeConfig(), in JSONStateWriter::writeConfig()");
    } else if (config_written) {
        THROW exception::UnknownInternalError("writeConfig() can only be called once per write session, in JSONStateWriter::writeConfig()");
    }

    // Simulation config
    if (sim_instance) {
        const auto& sim_cfg = sim_instance->getSimulationConfig();
        j["config"]["simulation"]["input_file"] = sim_cfg.input_file;
        j["config"]["simulation"]["step_log_file"] = sim_cfg.step_log_file;
        j["config"]["simulation"]["exit_log_file"] = sim_cfg.exit_log_file;
        j["config"]["simulation"]["common_log_file"] = sim_cfg.common_log_file;
        j["config"]["simulation"]["truncate_log_files"] = sim_cfg.truncate_log_files;
        j["config"]["simulation"]["random_seed"] = sim_cfg.random_seed;
        j["config"]["simulation"]["steps"] = sim_cfg.steps;
        j["config"]["simulation"]["verbosity"] = static_cast<unsigned int>(sim_cfg.verbosity);
        j["config"]["simulation"]["timing"] = sim_cfg.timing;
#ifdef FLAMEGPU_VISUALISATION
        j["config"]["simulation"]["console_mode"] = sim_cfg.console_mode;
#endif
    }
    
    // CUDA config
    if (auto* cudamodel_instance = dynamic_cast<const CUDASimulation*>(sim_instance)) {
        const auto& cuda_cfg = cudamodel_instance->getCUDAConfig();
        j["config"]["cuda"]["device_id"] = cuda_cfg.device_id;
        j["config"]["cuda"]["inLayerConcurrency"] = cuda_cfg.inLayerConcurrency;
    }
    config_written = true;
}
void JSONStateWriter::writeStats(unsigned int iterations) {
    if (!isWriting()) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeIterations(), in JSONStateWriter::writeIterations()");
    } else if (stats_written) {
        THROW exception::UnknownInternalError("writeIterations() can only be called once per write session, in JSONStateWriter::writeIterations()");
    }

    // General runtime stats (e.g. we could add timing data in future)
    j["stats"]["step_count"] = iterations;
    // in future could also support random seed, run args etc

    stats_written = true;
}
void JSONStateWriter::writeEnvironment(const std::shared_ptr<const detail::EnvironmentManager>& env_manager) {
    if (!isWriting()) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeEnvironment(), in JSONStateWriter::writeEnvironment()");
    } else if (environment_written) {
        THROW exception::UnknownInternalError("writeEnvironment() can only be called once per write session, in JSONStateWriter::writeEnvironment()");
    }

    // Environment properties
    auto &j_env = j["environment"];
    if (env_manager) {
        const char *env_buffer = reinterpret_cast<const char *>(env_manager->getHostBuffer());
        // for each environment property
        for (auto &a : env_manager->getPropertiesMap()) {
            if (a.second.elements == 1) {
                if (a.second.type == std::type_index(typeid(float))) {
                    j_env[a.first] = *reinterpret_cast<const float*>(env_buffer + a.second.offset);
                } else if (a.second.type == std::type_index(typeid(double))) {
                    j_env[a.first] = *reinterpret_cast<const double*>(env_buffer + a.second.offset);
                } else if (a.second.type == std::type_index(typeid(int64_t))) {
                    j_env[a.first] = *reinterpret_cast<const int64_t*>(env_buffer + a.second.offset);
                } else if (a.second.type == std::type_index(typeid(uint64_t))) {
                    j_env[a.first] = *reinterpret_cast<const uint64_t*>(env_buffer + a.second.offset);
                } else if (a.second.type == std::type_index(typeid(int32_t))) {
                    j_env[a.first] = *reinterpret_cast<const int32_t*>(env_buffer + a.second.offset);
                } else if (a.second.type == std::type_index(typeid(uint32_t))) {
                    j_env[a.first] = *reinterpret_cast<const uint32_t*>(env_buffer + a.second.offset);
                } else if (a.second.type == std::type_index(typeid(int16_t))) {
                    j_env[a.first] = *reinterpret_cast<const int16_t*>(env_buffer + a.second.offset);
                } else if (a.second.type == std::type_index(typeid(uint16_t))) {
                    j_env[a.first] = *reinterpret_cast<const uint16_t*>(env_buffer + a.second.offset);
                } else if (a.second.type == std::type_index(typeid(int8_t))) {
                    j_env[a.first] = static_cast<int32_t>(*reinterpret_cast<const int8_t*>(env_buffer + a.second.offset));  // Char outputs weird if being used as an integer
                } else if (a.second.type == std::type_index(typeid(uint8_t))) {
                    j_env[a.first] = static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(env_buffer + a.second.offset));  // Char outputs weird if being used as an integer
                } else {
                    THROW exception::JSONError("Model contains environment property '%s' of unsupported type '%s', "
                        "in JSONStateWriter::writeEnvironment()\n", a.first.c_str(), a.second.type.name());
                }
            }
            j_env[a.first] = {};
            // Loop through elements, to construct array
            for (unsigned int el = 0; el < a.second.elements; ++el) {
                if (a.second.type == std::type_index(typeid(float))) {
                    j_env[a.first].emplace_back(*reinterpret_cast<const float*>(env_buffer + a.second.offset + (el * sizeof(float))));
                } else if (a.second.type == std::type_index(typeid(double))) {
                    j_env[a.first].emplace_back(*reinterpret_cast<const double*>(env_buffer + a.second.offset + (el * sizeof(double))));
                } else if (a.second.type == std::type_index(typeid(int64_t))) {
                    j_env[a.first].emplace_back(*reinterpret_cast<const int64_t*>(env_buffer + a.second.offset + (el * sizeof(int64_t))));
                } else if (a.second.type == std::type_index(typeid(uint64_t))) {
                    j_env[a.first].emplace_back(*reinterpret_cast<const uint64_t*>(env_buffer + a.second.offset + (el * sizeof(uint64_t))));
                } else if (a.second.type == std::type_index(typeid(int32_t))) {
                    j_env[a.first].emplace_back(*reinterpret_cast<const int32_t*>(env_buffer + a.second.offset + (el * sizeof(int32_t))));
                } else if (a.second.type == std::type_index(typeid(uint32_t))) {
                    j_env[a.first].emplace_back(*reinterpret_cast<const uint32_t*>(env_buffer + a.second.offset + (el * sizeof(uint32_t))));
                } else if (a.second.type == std::type_index(typeid(int16_t))) {
                    j_env[a.first].emplace_back(*reinterpret_cast<const int16_t*>(env_buffer + a.second.offset + (el * sizeof(int16_t))));
                } else if (a.second.type == std::type_index(typeid(uint16_t))) {
                    j_env[a.first].emplace_back(*reinterpret_cast<const uint16_t*>(env_buffer + a.second.offset + (el * sizeof(uint16_t))));
                } else if (a.second.type == std::type_index(typeid(int8_t))) {
                    j_env[a.first].emplace_back(static_cast<int32_t>(*reinterpret_cast<const int8_t*>(env_buffer + a.second.offset + (el * sizeof(int8_t)))));  // Char outputs weird if being used as an integer
                } else if (a.second.type == std::type_index(typeid(uint8_t))) {
                    j_env[a.first].emplace_back(static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(env_buffer + a.second.offset + (el * sizeof(uint8_t)))));  // Char outputs weird if being used as an integer
                } else {
                    THROW exception::JSONError("Model contains environment property '%s' of unsupported type '%s', "
                        "in JSONStateWriter::writeEnvironment()\n", a.first.c_str(), a.second.type.name());
                }
            }
        }
    }

    environment_written = true;
}
void JSONStateWriter::writeMacroEnvironment(const std::shared_ptr<const detail::CUDAMacroEnvironment>& macro_env, std::initializer_list<std::string> filter) {
    if (!isWriting()) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeMacroEnvironment(), in JSONStateWriter::writeMacroEnvironment()");
    } else if (macro_environment_written) {
        THROW exception::UnknownInternalError("writeMacroEnvironment() can only be called once per write session, in JSONStateWriter::writeMacroEnvironment()");
    }

    // Macro Environment
    auto& j_menv = j["macro_environment"];
    if (macro_env) {
        const std::map<std::string, detail::CUDAMacroEnvironment::MacroEnvProp>& m_properties = macro_env->getPropertiesMap();
        for (const auto &_filter : filter) {
            if (m_properties.find(_filter) == m_properties.end()) {
                THROW exception::InvalidEnvProperty("Macro property '%s' specified in filter does not exist, in JSONStateWriter::writeMacroEnvironment()", _filter.c_str());
            }
        }
        std::set<std::string> filter_set = filter;
        // Calculate largest buffer in map
        size_t max_len = 0;
        for (const auto& [_, prop] : m_properties) {
            max_len = std::max(max_len, std::accumulate(prop.elements.begin(), prop.elements.end(), 1, std::multiplies<unsigned int>()) * prop.type_size);
        }
        if (max_len) {
            // Allocate temp buffer
            char* const t_buffer = static_cast<char*>(malloc(max_len));
            // Write out each array (all are written out as 1D arrays for simplicity given variable dimensions)
            for (const auto& [name, prop] : m_properties) {
                if (!filter_set.empty() && filter_set.find(name) == filter_set.end())
                    continue;
                // Copy data
                const size_t element_ct = std::accumulate(prop.elements.begin(), prop.elements.end(), 1, std::multiplies<unsigned int>());
                gpuErrchk(cudaMemcpy(t_buffer, prop.d_ptr, element_ct * prop.type_size, cudaMemcpyDeviceToHost));
                j_menv[name] = {};
                for (size_t i = 0; i < element_ct; ++i) {
                    if (prop.type == std::type_index(typeid(float))) {
                        j_menv[name].emplace_back(*reinterpret_cast<const float*>(t_buffer + i * sizeof(float)));
                    } else if (prop.type == std::type_index(typeid(double))) {
                        j_menv[name].emplace_back(*reinterpret_cast<const double*>(t_buffer + i * sizeof(double)));
                    } else if (prop.type == std::type_index(typeid(int64_t))) {
                        j_menv[name].emplace_back(*reinterpret_cast<const int64_t*>(t_buffer + i * sizeof(int64_t)));
                    } else if (prop.type == std::type_index(typeid(uint64_t))) {
                        j_menv[name].emplace_back(*reinterpret_cast<const uint64_t*>(t_buffer + i * sizeof(uint64_t)));
                    } else if (prop.type == std::type_index(typeid(int32_t))) {
                        j_menv[name].emplace_back(*reinterpret_cast<const int32_t*>(t_buffer + i * sizeof(int32_t)));
                    } else if (prop.type == std::type_index(typeid(uint32_t))) {
                        j_menv[name].emplace_back(*reinterpret_cast<const uint32_t*>(t_buffer + i * sizeof(uint32_t)));
                    } else if (prop.type == std::type_index(typeid(int16_t))) {
                        j_menv[name].emplace_back(*reinterpret_cast<const int16_t*>(t_buffer + i * sizeof(int16_t)));
                    } else if (prop.type == std::type_index(typeid(uint16_t))) {
                        j_menv[name].emplace_back(*reinterpret_cast<const uint16_t*>(t_buffer + i * sizeof(uint16_t)));
                    } else if (prop.type == std::type_index(typeid(int8_t))) {
                        j_menv[name].emplace_back(static_cast<int32_t>(*reinterpret_cast<const int8_t*>(t_buffer + i * sizeof(int8_t))));  // Char outputs weird if being used as an integer
                    } else if (prop.type == std::type_index(typeid(uint8_t))) {
                        j_menv[name].emplace_back(static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(t_buffer + i * sizeof(uint8_t))));  // Char outputs weird if being used as an integer
                    } else {
                        THROW exception::JSONError("Model contains macro environment property '%s' of unsupported type '%s', "
                            "in JSONStateWriter::writeFullModelState()\n", name.c_str(), prop.type.name());
                    }
                }
            }
            // Release temp buffer
            free(t_buffer);
        }
    }

    macro_environment_written = true;
}
void JSONStateWriter::writeAgents(const util::StringPairUnorderedMap<std::shared_ptr<const AgentVector>>& agents_map) {
    if (!isWriting()) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeAgents(), in JSONStateWriter::writeAgents()");
    } else if (agents_written) {
        THROW exception::UnknownInternalError("writeAgents() can only be called once per write session, in JSONStateWriter::writeAgents()");
    }

    // AgentStates
    auto& j_agt = j["agents"];
    // Build a set of agent names
    std::set<std::string> agent_names;
    for (const auto& [key, _] : agents_map) {
        agent_names.emplace(key.first);
    }
    // Process agents one at a time by iterating the map once per agent type
    for (const auto &agt : agent_names) {
        auto& j_agt_t = j_agt[agt];
        for (const auto &agent : agents_map) {
            const std::string &agent_name = agent.first.first;
            if (agent_name != agt)
                continue;
            const std::string &state_name = agent.first.second;
            const VariableMap &agent_vars = agent.second->getVariableMetaData();
            // States
            const unsigned int populationSize = agent.second->size();
            // Only log states with agents
            if (populationSize) {
                j_agt_t[state_name] = {};
                for (unsigned int i = 0; i < populationSize; ++i) {
                    AgentVector::CAgent instance = agent.second->at(i);
                    nlohmann::ordered_json t_agt;
                    // for each variable
                    for (auto var : agent_vars) {
                        // Set name
                        const std::string variable_name = var.first;
                        t_agt[variable_name] = {};
                        // Loop through elements, to construct array
                        for (unsigned int el = 0; el < var.second.elements; ++el) {
                            if (var.second.type == std::type_index(typeid(float))) {
                                t_agt.emplace_back(instance.getVariable<float>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(double))) {
                                t_agt.emplace_back(instance.getVariable<double>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(int64_t))) {
                                t_agt.emplace_back(instance.getVariable<int64_t>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(uint64_t))) {
                                t_agt.emplace_back(instance.getVariable<uint64_t>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(int32_t))) {
                                t_agt.emplace_back(instance.getVariable<int32_t>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(uint32_t))) {
                                t_agt.emplace_back(instance.getVariable<uint32_t>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(int16_t))) {
                                t_agt.emplace_back(instance.getVariable<int16_t>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(uint16_t))) {
                                t_agt.emplace_back(instance.getVariable<uint16_t>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(int8_t))) {
                                t_agt.emplace_back(instance.getVariable<int8_t>(variable_name, el));  // Char outputs weird if being used as an integer
                            } else if (var.second.type == std::type_index(typeid(uint8_t))) {
                                t_agt.emplace_back(instance.getVariable<uint8_t>(variable_name, el));  // Char outputs weird if being used as an integer
                            } else {
                                THROW exception::JSONError("Agent '%s' contains variable '%s' of unsupported type '%s', "
                                    "in JSONStateWriter::writeAgents()\n", agent.first.first.c_str(), variable_name.c_str(), var.second.type.name());
                            }
                        }
                    }
                    j_agt_t[state_name].push_back(t_agt);
                }
            }
        }
    }

    agents_written = true;
}
}  // namespace io
}  // namespace flamegpu
