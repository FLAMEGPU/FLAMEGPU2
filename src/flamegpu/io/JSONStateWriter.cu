#include "flamegpu/io/JSONStateWriter.h"

#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include <set>

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
    if (writer) {
        THROW exception::UnknownInternalError("Writing already active, in JSONStateWriter::beginWrite()");
    }
    buffer = rapidjson::StringBuffer();
    if (pretty_print) {
        auto t_writer = std::make_unique<rapidjson::PrettyWriter<rapidjson::StringBuffer, rapidjson::UTF8<>, rapidjson::UTF8<>, rapidjson::CrtAllocator, rapidjson::kWriteNanAndInfFlag>>(buffer);
        t_writer->SetIndent('\t', 1);
        writer = std::move(t_writer);
    } else {
        writer = std::make_unique<rapidjson::Writer<rapidjson::StringBuffer, rapidjson::UTF8<>, rapidjson::UTF8<>, rapidjson::CrtAllocator, rapidjson::kWriteNanAndInfFlag>>(buffer);
    }
    // Begin Json file
    writer->StartObject();

    // Clear flags
    this->config_written = false;
    this->stats_written = false;
    this->environment_written = false;
    this->macro_environment_written = false;
    this->agents_written = false;
}
void JSONStateWriter::endWrite() {
    if (!writer) {
        THROW exception::UnknownInternalError("Writing not active, in XMLStateWriter::endWrite()");
    }

    // End Json file
    writer->EndObject();

    std::ofstream out(outputPath, std::ofstream::trunc);
    out << buffer.GetString();
    out.close();

    writer.reset();
    buffer.Clear();
}

void JSONStateWriter::writeConfig(const Simulation *sim_instance) {
    if (!writer) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeConfig(), in JSONStateWriter::writeConfig()");
    } else if (config_written) {
        THROW exception::UnknownInternalError("writeConfig() can only be called once per write session, in JSONStateWriter::writeConfig()");
    }

    // General simulation config/properties
    writer->Key("config");
    writer->StartObject();
    // Simulation config
    if (sim_instance) {
        writer->Key("simulation");
        writer->StartObject();
        {
            const auto& sim_cfg = sim_instance->getSimulationConfig();
            // Input file
            writer->Key("input_file");
            writer->String(sim_cfg.input_file.c_str());
            // Step log file
            writer->Key("step_log_file");
            writer->String(sim_cfg.step_log_file.c_str());
            // Exit log file
            writer->Key("exit_log_file");
            writer->String(sim_cfg.exit_log_file.c_str());
            // Common log file
            writer->Key("common_log_file");
            writer->String(sim_cfg.common_log_file.c_str());
            // Truncate log files
            writer->Key("truncate_log_files");
            writer->Bool(sim_cfg.truncate_log_files);
            // Random seed
            writer->Key("random_seed");
            writer->Uint64(sim_cfg.random_seed);
            // Steps
            writer->Key("steps");
            writer->Uint(sim_cfg.steps);
            // Verbose output
            writer->Key("verbosity");
            writer->Uint(static_cast<unsigned int>(sim_cfg.verbosity));
            // Timing Output
            writer->Key("timing");
            writer->Bool(sim_cfg.timing);
#ifdef FLAMEGPU_VISUALISATION
            // Console mode
            writer->Key("console_mode");
            writer->Bool(sim_cfg.console_mode);
#endif
        }
        writer->EndObject();

        // CUDA config
        if (auto* cudamodel_instance = dynamic_cast<const CUDASimulation*>(sim_instance)) {
            writer->Key("cuda");
            writer->StartObject();
            {
                const auto& cuda_cfg = cudamodel_instance->getCUDAConfig();
                // device_id
                writer->Key("device_id");
                writer->Uint(cuda_cfg.device_id);
                // inLayerConcurrency
                writer->Key("inLayerConcurrency");
                writer->Bool(cuda_cfg.inLayerConcurrency);
            }
            writer->EndObject();
        }
    }
    writer->EndObject();
    config_written = true;
}
void JSONStateWriter::writeStats(unsigned int iterations) {
    if (!writer) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeIterations(), in JSONStateWriter::writeIterations()");
    } else if (stats_written) {
        THROW exception::UnknownInternalError("writeIterations() can only be called once per write session, in JSONStateWriter::writeIterations()");
    }

    // General runtime stats (e.g. we could add timing data in future)
    writer->Key("stats");
    writer->StartObject();
    {
        // Steps
        writer->Key("step_count");
        writer->Uint(iterations);
        // in future could also support random seed, run args etc
    }
    writer->EndObject();

    stats_written = true;
}
void JSONStateWriter::writeEnvironment(const std::shared_ptr<const detail::EnvironmentManager>& env_manager) {
    if (!writer) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeEnvironment(), in JSONStateWriter::writeEnvironment()");
    } else if (environment_written) {
        THROW exception::UnknownInternalError("writeEnvironment() can only be called once per write session, in JSONStateWriter::writeEnvironment()");
    }

    // Environment properties
    writer->Key("environment");
    writer->StartObject();
    if (env_manager) {
        const char *env_buffer = reinterpret_cast<const char *>(env_manager->getHostBuffer());
        // for each environment property
        for (auto &a : env_manager->getPropertiesMap()) {
            // Set name
            writer->Key(a.first.c_str());
            // Output value
            if (a.second.elements > 1) {
                // Value is an array
                writer->StartArray();
            }
            // Loop through elements, to construct array
            for (unsigned int el = 0; el < a.second.elements; ++el) {
                if (a.second.type == std::type_index(typeid(float))) {
                    writer->Double(*reinterpret_cast<const float*>(env_buffer + a.second.offset + (el * sizeof(float))));
                } else if (a.second.type == std::type_index(typeid(double))) {
                    writer->Double(*reinterpret_cast<const double*>(env_buffer + a.second.offset + (el * sizeof(double))));
                } else if (a.second.type == std::type_index(typeid(int64_t))) {
                    writer->Int64(*reinterpret_cast<const int64_t*>(env_buffer + a.second.offset + (el * sizeof(int64_t))));
                } else if (a.second.type == std::type_index(typeid(uint64_t))) {
                    writer->Uint64(*reinterpret_cast<const uint64_t*>(env_buffer + a.second.offset + (el * sizeof(uint64_t))));
                } else if (a.second.type == std::type_index(typeid(int32_t))) {
                    writer->Int(*reinterpret_cast<const int32_t*>(env_buffer + a.second.offset + (el * sizeof(int32_t))));
                } else if (a.second.type == std::type_index(typeid(uint32_t))) {
                    writer->Uint(*reinterpret_cast<const uint32_t*>(env_buffer + a.second.offset + (el * sizeof(uint32_t))));
                } else if (a.second.type == std::type_index(typeid(int16_t))) {
                    writer->Int(*reinterpret_cast<const int16_t*>(env_buffer + a.second.offset + (el * sizeof(int16_t))));
                } else if (a.second.type == std::type_index(typeid(uint16_t))) {
                    writer->Uint(*reinterpret_cast<const uint16_t*>(env_buffer + a.second.offset + (el * sizeof(uint16_t))));
                } else if (a.second.type == std::type_index(typeid(int8_t))) {
                    writer->Int(static_cast<int32_t>(*reinterpret_cast<const int8_t*>(env_buffer + a.second.offset + (el * sizeof(int8_t)))));  // Char outputs weird if being used as an integer
                } else if (a.second.type == std::type_index(typeid(uint8_t))) {
                    writer->Uint(static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(env_buffer + a.second.offset + (el * sizeof(uint8_t)))));  // Char outputs weird if being used as an integer
                } else {
                    THROW exception::RapidJSONError("Model contains environment property '%s' of unsupported type '%s', "
                        "in JSONStateWriter::writeEnvironment()\n", a.first.c_str(), a.second.type.name());
                }
            }
            if (a.second.elements > 1) {
                // Value is an array
                writer->EndArray();
            }
        }
    }
    writer->EndObject();

    environment_written = true;
}
void JSONStateWriter::writeMacroEnvironment(const std::shared_ptr<const detail::CUDAMacroEnvironment>& macro_env, std::initializer_list<std::string> filter) {
    if (!writer) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeMacroEnvironment(), in JSONStateWriter::writeMacroEnvironment()");
    } else if (macro_environment_written) {
        THROW exception::UnknownInternalError("writeMacroEnvironment() can only be called once per write session, in JSONStateWriter::writeMacroEnvironment()");
    }

    // Macro Environment
    writer->Key("macro_environment");
    writer->StartObject();
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
                writer->Key(name.c_str());
                writer->StartArray();
                for (size_t i = 0; i < element_ct; ++i) {
                    if (prop.type == std::type_index(typeid(float))) {
                        writer->Double(*reinterpret_cast<const float*>(t_buffer + i * sizeof(float)));
                    } else if (prop.type == std::type_index(typeid(double))) {
                        writer->Double(*reinterpret_cast<const double*>(t_buffer + i * sizeof(double)));
                    } else if (prop.type == std::type_index(typeid(int64_t))) {
                        writer->Int64(*reinterpret_cast<const int64_t*>(t_buffer + i * sizeof(int64_t)));
                    } else if (prop.type == std::type_index(typeid(uint64_t))) {
                        writer->Uint64(*reinterpret_cast<const uint64_t*>(t_buffer + i * sizeof(uint64_t)));
                    } else if (prop.type == std::type_index(typeid(int32_t))) {
                        writer->Int(*reinterpret_cast<const int32_t*>(t_buffer + i * sizeof(int32_t)));
                    } else if (prop.type == std::type_index(typeid(uint32_t))) {
                        writer->Uint(*reinterpret_cast<const uint32_t*>(t_buffer + i * sizeof(uint32_t)));
                    } else if (prop.type == std::type_index(typeid(int16_t))) {
                        writer->Int(*reinterpret_cast<const int16_t*>(t_buffer + i * sizeof(int16_t)));
                    } else if (prop.type == std::type_index(typeid(uint16_t))) {
                        writer->Uint(*reinterpret_cast<const uint16_t*>(t_buffer + i * sizeof(uint16_t)));
                    } else if (prop.type == std::type_index(typeid(int8_t))) {
                        writer->Int(static_cast<int32_t>(*reinterpret_cast<const int8_t*>(t_buffer + i * sizeof(int8_t))));  // Char outputs weird if being used as an integer
                    } else if (prop.type == std::type_index(typeid(uint8_t))) {
                        writer->Uint(static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(t_buffer + i * sizeof(uint8_t))));  // Char outputs weird if being used as an integer
                    } else {
                        THROW exception::RapidJSONError("Model contains macro environment property '%s' of unsupported type '%s', "
                            "in JSONStateWriter::writeFullModelState()\n", name.c_str(), prop.type.name());
                    }
                }
                writer->EndArray();
            }
            // Release temp buffer
            free(t_buffer);
        }
    }
    writer->EndObject();

    macro_environment_written = true;
}
void JSONStateWriter::writeAgents(const util::StringPairUnorderedMap<std::shared_ptr<const AgentVector>>& agents_map) {
    if (!writer) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeAgents(), in JSONStateWriter::writeAgents()");
    } else if (agents_written) {
        THROW exception::UnknownInternalError("writeAgents() can only be called once per write session, in JSONStateWriter::writeAgents()");
    }

    // AgentStates
    writer->Key("agents");
    writer->StartObject();
    // Build a set of agent names
    std::set<std::string> agent_names;
    for (const auto& [key, _] : agents_map) {
        agent_names.emplace(key.first);
    }
    // Process agents one at a time by iterating the map once per agent type
    for (const auto &agt : agent_names) {
        writer->Key(agt.c_str());
        writer->StartObject();
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
                writer->Key(state_name.c_str());
                writer->StartArray();
                for (unsigned int i = 0; i < populationSize; ++i) {
                    writer->StartObject();
                    AgentVector::CAgent instance = agent.second->at(i);
                    // for each variable
                    for (auto var : agent_vars) {
                        // Set name
                        const std::string variable_name = var.first;
                        writer->Key(variable_name.c_str());
                        // Output value
                        if (var.second.elements > 1) {
                            // Value is an array
                            writer->StartArray();
                        }
                        // Loop through elements, to construct array
                        for (unsigned int el = 0; el < var.second.elements; ++el) {
                            if (var.second.type == std::type_index(typeid(float))) {
                                writer->Double(instance.getVariable<float>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(double))) {
                                writer->Double(instance.getVariable<double>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(int64_t))) {
                                writer->Int64(instance.getVariable<int64_t>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(uint64_t))) {
                                writer->Uint64(instance.getVariable<uint64_t>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(int32_t))) {
                                writer->Int(instance.getVariable<int32_t>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(uint32_t))) {
                                writer->Uint(instance.getVariable<uint32_t>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(int16_t))) {
                                writer->Int(instance.getVariable<int16_t>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(uint16_t))) {
                                writer->Uint(instance.getVariable<uint16_t>(variable_name, el));
                            } else if (var.second.type == std::type_index(typeid(int8_t))) {
                                writer->Int(instance.getVariable<int8_t>(variable_name, el));  // Char outputs weird if being used as an integer
                            } else if (var.second.type == std::type_index(typeid(uint8_t))) {
                                writer->Uint(instance.getVariable<uint8_t>(variable_name, el));  // Char outputs weird if being used as an integer
                            } else {
                                THROW exception::RapidJSONError("Agent '%s' contains variable '%s' of unsupported type '%s', "
                                    "in JSONStateWriter::writeAgents()\n", agent.first.first.c_str(), variable_name.c_str(), var.second.type.name());
                            }
                        }
                        if (var.second.elements > 1) {
                            // Value is an array
                            writer->EndArray();
                        }
                    }
                    writer->EndObject();
                }
                writer->EndArray();
            }
        }
        writer->EndObject();
    }
    writer->EndObject();

    agents_written = true;
}
}  // namespace io
}  // namespace flamegpu
