#include "flamegpu/io/JSONStateWriter.h"

#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <iostream>
#include <fstream>
#include <string>

#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/pop/AgentVector.h"
#include "flamegpu/gpu/CUDASimulation.h"
#include "flamegpu/util/StringPair.h"

namespace flamegpu {
namespace io {

JSONStateWriter::JSONStateWriter(
    const std::string &model_name,
    const std::shared_ptr<EnvironmentManager> &env_manager,
    const util::StringPairUnorderedMap<std::shared_ptr<AgentVector>>&model,
    const unsigned int iterations,
    const std::string &output_file,
    const Simulation *_sim_instance)
    : StateWriter(model_name, env_manager, model, iterations, output_file, _sim_instance) {}

template<typename T>
void JSONStateWriter::doWrite(T &writer) {
    // Begin json output object
    writer.StartObject();

    // General simulation config/properties
    writer.Key("config");
    writer.StartObject();
    {
        // Simulation config
        if (sim_instance) {
            writer.Key("simulation");
            writer.StartObject();
            {
                const auto &sim_cfg = sim_instance->getSimulationConfig();
                // Input file
                writer.Key("input_file");
                writer.String(sim_cfg.input_file.c_str());
                // Step log file
                writer.Key("step_log_file");
                writer.String(sim_cfg.step_log_file.c_str());
                // Exit log file
                writer.Key("exit_log_file");
                writer.String(sim_cfg.exit_log_file.c_str());
                // Common log file
                writer.Key("common_log_file");
                writer.String(sim_cfg.common_log_file.c_str());
                // Truncate log files
                writer.Key("truncate_log_files");
                writer.Bool(sim_cfg.truncate_log_files);
                // Random seed
                writer.Key("random_seed");
                writer.Uint64(sim_cfg.random_seed);
                // Steps
                writer.Key("steps");
                writer.Uint(sim_cfg.steps);
                // Verbose output
                writer.Key("verbosity");
                writer.Uint(sim_cfg.verbosity);
                // Timing Output
                writer.Key("timing");
                writer.Bool(sim_cfg.timing);
#ifdef VISUALISATION
                // Console mode
                writer.Key("console_mode");
                writer.Bool(sim_cfg.console_mode);
#endif
            }
            writer.EndObject();
        }

        // CUDA config
        if (auto *cudamodel_instance = dynamic_cast<const CUDASimulation*>(sim_instance)) {
            writer.Key("cuda");
            writer.StartObject();
            {
                const auto &cuda_cfg = cudamodel_instance->getCUDAConfig();
                // device_id
                writer.Key("device_id");
                writer.Uint(cuda_cfg.device_id);
                // inLayerConcurrency
                writer.Key("inLayerConcurrency");
                writer.Bool(cuda_cfg.inLayerConcurrency);
            }
            writer.EndObject();
        }
    }
    writer.EndObject();

    // General runtime stats (e.g. we could add timing data in future)
    writer.Key("stats");
    writer.StartObject();
    {
        // Steps
        writer.Key("step_count");
        writer.Uint(iterations);
        // in future could also support random seed, run args etc
    }
    writer.EndObject();

    // Environment properties
    writer.Key("environment");
    writer.StartObject();
    if (env_manager) {
        const char *env_buffer = reinterpret_cast<const char *>(env_manager->getHostBuffer());
        // for each environment property
        for (auto &a : env_manager->getPropertiesMap()) {
            // Set name
            writer.Key(a.first.c_str());
            // Output value
            if (a.second.elements > 1) {
                // Value is an array
                writer.StartArray();
            }
            // Loop through elements, to construct array
            for (unsigned int el = 0; el < a.second.elements; ++el) {
                if (a.second.type == std::type_index(typeid(float))) {
                    writer.Double(*reinterpret_cast<const float*>(env_buffer + a.second.offset + (el * sizeof(float))));
                } else if (a.second.type == std::type_index(typeid(double))) {
                    writer.Double(*reinterpret_cast<const double*>(env_buffer + a.second.offset + (el * sizeof(double))));
                } else if (a.second.type == std::type_index(typeid(int64_t))) {
                    writer.Int64(*reinterpret_cast<const int64_t*>(env_buffer + a.second.offset + (el * sizeof(int64_t))));
                } else if (a.second.type == std::type_index(typeid(uint64_t))) {
                    writer.Uint64(*reinterpret_cast<const uint64_t*>(env_buffer + a.second.offset + (el * sizeof(uint64_t))));
                } else if (a.second.type == std::type_index(typeid(int32_t))) {
                    writer.Int(*reinterpret_cast<const int32_t*>(env_buffer + a.second.offset + (el * sizeof(int32_t))));
                } else if (a.second.type == std::type_index(typeid(uint32_t))) {
                    writer.Uint(*reinterpret_cast<const uint32_t*>(env_buffer + a.second.offset + (el * sizeof(uint32_t))));
                } else if (a.second.type == std::type_index(typeid(int16_t))) {
                    writer.Int(*reinterpret_cast<const int16_t*>(env_buffer + a.second.offset + (el * sizeof(int16_t))));
                } else if (a.second.type == std::type_index(typeid(uint16_t))) {
                    writer.Uint(*reinterpret_cast<const uint16_t*>(env_buffer + a.second.offset + (el * sizeof(uint16_t))));
                } else if (a.second.type == std::type_index(typeid(int8_t))) {
                    writer.Int(static_cast<int32_t>(*reinterpret_cast<const int8_t*>(env_buffer + a.second.offset + (el * sizeof(int8_t)))));  // Char outputs weird if being used as an integer
                } else if (a.second.type == std::type_index(typeid(uint8_t))) {
                    writer.Uint(static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(env_buffer + a.second.offset + (el * sizeof(uint8_t)))));  // Char outputs weird if being used as an integer
                } else {
                    THROW exception::RapidJSONError("Model contains environment property '%s' of unsupported type '%s', "
                        "in JSONStateWriter::writeStates()\n", a.first.c_str(), a.second.type.name());
                }
            }
            if (a.second.elements > 1) {
                // Value is an array
                writer.EndArray();
            }
        }
    }
    writer.EndObject();

    // AgentStates
    writer.Key("agents");
    writer.StartObject();
    for (const auto &agent : model_state) {
        const std::string &agent_name = agent.first.first;
        const std::string &state_name = agent.first.second;
        writer.Key(agent_name.c_str());
        writer.StartObject();
        const VariableMap &agent_vars = agent.second->getVariableMetaData();
        // States
        const unsigned int populationSize = agent.second->size();
        // Only log states with agents
        if (populationSize) {
            writer.Key(state_name.c_str());
            writer.StartArray();
            for (unsigned int i = 0; i < populationSize; ++i) {
                writer.StartObject();
                AgentVector::Agent instance = agent.second->at(i);
                // for each variable
                for (auto var : agent_vars) {
                    // Set name
                    const std::string variable_name = var.first;
                    writer.Key(variable_name.c_str());
                    // Output value
                    if (var.second.elements > 1) {
                        // Value is an array
                        writer.StartArray();
                    }
                    // Loop through elements, to construct array
                    for (unsigned int el = 0; el < var.second.elements; ++el) {
                        if (var.second.type == std::type_index(typeid(float))) {
                            writer.Double(instance.getVariable<float>(variable_name, el));
                        } else if (var.second.type == std::type_index(typeid(double))) {
                            writer.Double(instance.getVariable<double>(variable_name, el));
                        } else if (var.second.type == std::type_index(typeid(int64_t))) {
                            writer.Int64(instance.getVariable<int64_t>(variable_name, el));
                        } else if (var.second.type == std::type_index(typeid(uint64_t))) {
                            writer.Uint64(instance.getVariable<uint64_t>(variable_name, el));
                        } else if (var.second.type == std::type_index(typeid(int32_t))) {
                            writer.Int(instance.getVariable<int32_t>(variable_name, el));
                        } else if (var.second.type == std::type_index(typeid(uint32_t))) {
                            writer.Uint(instance.getVariable<uint32_t>(variable_name, el));
                        } else if (var.second.type == std::type_index(typeid(int16_t))) {
                            writer.Int(instance.getVariable<int16_t>(variable_name, el));
                        } else if (var.second.type == std::type_index(typeid(uint16_t))) {
                            writer.Uint(instance.getVariable<uint16_t>(variable_name, el));
                        } else if (var.second.type == std::type_index(typeid(int8_t))) {
                            writer.Int(instance.getVariable<int8_t>(variable_name, el));  // Char outputs weird if being used as an integer
                        } else if (var.second.type == std::type_index(typeid(uint8_t))) {
                            writer.Uint(instance.getVariable<uint8_t>(variable_name, el));  // Char outputs weird if being used as an integer
                        } else {
                            THROW exception::RapidJSONError("Agent '%s' contains variable '%s' of unsupported type '%s', "
                                "in JSONStateWriter::writeStates()\n", agent.first.first.c_str(), variable_name.c_str(), var.second.type.name());
                        }
                    }
                    if (var.second.elements > 1) {
                        // Value is an array
                        writer.EndArray();
                    }
                }
                writer.EndObject();
            }
            writer.EndArray();
        }
        writer.EndObject();
    }
    writer.EndObject();

    // End Json file
    writer.EndObject();
}

int JSONStateWriter::writeStates(bool prettyPrint) {
    rapidjson::StringBuffer s;
    if (prettyPrint) {
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer = rapidjson::PrettyWriter<rapidjson::StringBuffer>(s);
        writer.SetIndent('\t', 1);
        doWrite(writer);
    } else {
        rapidjson::Writer<rapidjson::StringBuffer> writer = rapidjson::Writer<rapidjson::StringBuffer>(s);
        doWrite(writer);
    }

    // Perform output
    std::ofstream out(outputFile);
    out << s.GetString();
    out.close();

    return 0;
}

}  // namespace io
}  // namespace flamegpu
