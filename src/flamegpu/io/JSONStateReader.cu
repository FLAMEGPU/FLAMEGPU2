#include "flamegpu/io/JSONStateReader.h"

#include <stack>
#include <fstream>
#include <string>
#include <unordered_map>
#include <cerrno>
#include <numeric>
#include <cstdio>
#include <vector>
#include <functional>
#include <memory>

#include <nlohmann/json.hpp>

#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/simulation/AgentVector.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/model/EnvironmentData.h"
#include "flamegpu/simulation/CUDASimulation.h"
#include "flamegpu/util/StringPair.h"

namespace flamegpu {
namespace io {

/**
 * This is the main sax style parser for the json state
 * It stores it's current position within the hierarchy with mode, lastKey and current_variable_array_index
 */
class JSONStateReader_impl : public nlohmann::json_sax<nlohmann::json> {
    enum Mode{ Nop, Root, Config, Stats, SimCfg, CUDACfg, Environment, MacroEnvironment, Agents, Agent, State, AgentInstance, VariableArray };
    std::stack<Mode> mode;
    std::string lastKey;
    std::string filename;
    const std::shared_ptr<const ModelData>& model;
    std::unordered_map<std::string, detail::Any> &env_init;
    std::unordered_map<std::string, std::vector<char>> &macro_env_init;
    util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> &agents_map;
    Verbosity verbosity;
    /**
     * Tracks current position reading variable array
     */
    unsigned int current_variable_array_index = 0;
    /**
     * Set when we enter an agent
     */
    std::string current_agent;
    /**
     * Set when we enter a state
     */
    std::string current_state;

 public:
    JSONStateReader_impl(const std::string &_filename,
        const std::shared_ptr<const ModelData> &_model,
        std::unordered_map<std::string, detail::Any> &_env_init,
        std::unordered_map<std::string, std::vector<char>> & _macro_env_init,
        util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> &_agents_map,
        Verbosity _verbosity)
        : filename(_filename)
        , model(_model)
        , env_init(_env_init)
        , macro_env_init(_macro_env_init)
        , agents_map(_agents_map)
        , verbosity(_verbosity) { }

    template<typename T>
    bool processValue(const T val) {
        Mode isArray = Nop;
        if (mode.top() == VariableArray) {
            isArray = mode.top();
            mode.pop();
        }
        if (mode.top() == Environment) {
            const auto it = model->environment->properties.find(lastKey);
            if (it == model->environment->properties.end()) {
                THROW exception::JSONError("Input file contains unrecognised environment property '%s',"
                    "in JSONStateReader::parse()\n", lastKey.c_str());
            }
            if (current_variable_array_index == 0) {
                // New property, create buffer with default value and add to map
                if (!env_init.emplace(lastKey, detail::Any(it->second.data)).second) {
                    THROW exception::JSONError("Input file contains environment property '%s' multiple times, "
                        "in JSONStateReader::parse()\n", lastKey.c_str());
                }
            } else if (current_variable_array_index >= it->second.data.elements) {
                THROW exception::JSONError("Input file contains environment property '%s' with %u elements expected %u,"
                    "in JSONStateReader::parse()\n", lastKey.c_str(), current_variable_array_index, it->second.data.elements);
            }
            // Retrieve the linked any and replace the value
            const auto ei_it = env_init.find(lastKey);
            const std::type_index val_type = it->second.data.type;
            if (val_type == std::type_index(typeid(float))) {
                static_cast<float*>(const_cast<void*>(ei_it->second.ptr))[current_variable_array_index++] = static_cast<float>(val);
            } else if (val_type == std::type_index(typeid(double))) {
                static_cast<double*>(const_cast<void*>(ei_it->second.ptr))[current_variable_array_index++] = static_cast<double>(val);
            } else if (val_type == std::type_index(typeid(int64_t))) {
                static_cast<int64_t*>(const_cast<void*>(ei_it->second.ptr))[current_variable_array_index++] = static_cast<int64_t>(val);
            } else if (val_type == std::type_index(typeid(uint64_t))) {
                static_cast<uint64_t*>(const_cast<void*>(ei_it->second.ptr))[current_variable_array_index++] = static_cast<uint64_t>(val);
            } else if (val_type == std::type_index(typeid(int32_t))) {
                static_cast<int32_t*>(const_cast<void*>(ei_it->second.ptr))[current_variable_array_index++] = static_cast<int32_t>(val);
            } else if (val_type == std::type_index(typeid(uint32_t))) {
                static_cast<uint32_t*>(const_cast<void*>(ei_it->second.ptr))[current_variable_array_index++] = static_cast<uint32_t>(val);
            } else if (val_type == std::type_index(typeid(int16_t))) {
                static_cast<int16_t*>(const_cast<void*>(ei_it->second.ptr))[current_variable_array_index++] = static_cast<int16_t>(val);
            } else if (val_type == std::type_index(typeid(uint16_t))) {
                static_cast<uint16_t*>(const_cast<void*>(ei_it->second.ptr))[current_variable_array_index++] = static_cast<uint16_t>(val);
            } else if (val_type == std::type_index(typeid(int8_t))) {
                static_cast<int8_t*>(const_cast<void*>(ei_it->second.ptr))[current_variable_array_index++] = static_cast<int8_t>(val);
            } else if (val_type == std::type_index(typeid(uint8_t))) {
                static_cast<uint8_t*>(const_cast<void*>(ei_it->second.ptr))[current_variable_array_index++] = static_cast<uint8_t>(val);
            } else {
                THROW exception::JSONError("Model contains environment property '%s' of unsupported type '%s', "
                    "in JSONStateReader::parse()\n", lastKey.c_str(), val_type.name());
            }
        } else if (mode.top() == MacroEnvironment) {
            const auto it = model->environment->macro_properties.find(lastKey);
            if (it == model->environment->macro_properties.end()) {
                THROW exception::JSONError("Input file contains unrecognised macro environment property '%s',"
                    "in JSONStateReader::parse()\n", lastKey.c_str());
            }
            const unsigned int macro_prop_elements = std::accumulate(it->second.elements.begin(), it->second.elements.end(), 1, std::multiplies<unsigned int>());
            if (current_variable_array_index == 0) {
                // New property, create buffer with default value and add to map
                if (!macro_env_init.emplace(lastKey, std::vector<char>(macro_prop_elements * it->second.type_size)).second) {
                    THROW exception::JSONError("Input file contains environment property '%s' multiple times, "
                        "in JSONStateReader::parse()\n", lastKey.c_str());
                }
            } else if (current_variable_array_index >= macro_prop_elements) {
                THROW exception::JSONError("Input file contains environment property '%s' with %u elements expected %u,"
                    "in JSONStateReader::parse()\n", lastKey.c_str(), current_variable_array_index, macro_prop_elements);
            }
            // Retrieve the linked any and replace the value
            auto &mei = macro_env_init.at(lastKey);
            const std::type_index val_type = it->second.type;
            if (val_type == std::type_index(typeid(float))) {
                static_cast<float*>(static_cast<void*>(mei.data()))[current_variable_array_index++] = static_cast<float>(val);
            } else if (val_type == std::type_index(typeid(double))) {
                static_cast<double*>(static_cast<void*>(mei.data()))[current_variable_array_index++] = static_cast<double>(val);
            } else if (val_type == std::type_index(typeid(int64_t))) {
                static_cast<int64_t*>(static_cast<void*>(mei.data()))[current_variable_array_index++] = static_cast<int64_t>(val);
            } else if (val_type == std::type_index(typeid(uint64_t))) {
                static_cast<uint64_t*>(static_cast<void*>(mei.data()))[current_variable_array_index++] = static_cast<uint64_t>(val);
            } else if (val_type == std::type_index(typeid(int32_t))) {
                static_cast<int32_t*>(static_cast<void*>(mei.data()))[current_variable_array_index++] = static_cast<int32_t>(val);
            } else if (val_type == std::type_index(typeid(uint32_t))) {
                static_cast<uint32_t*>(static_cast<void*>(mei.data()))[current_variable_array_index++] = static_cast<uint32_t>(val);
            } else if (val_type == std::type_index(typeid(int16_t))) {
                static_cast<int16_t*>(static_cast<void*>(mei.data()))[current_variable_array_index++] = static_cast<int16_t>(val);
            } else if (val_type == std::type_index(typeid(uint16_t))) {
                static_cast<uint16_t*>(static_cast<void*>(mei.data()))[current_variable_array_index++] = static_cast<uint16_t>(val);
            } else if (val_type == std::type_index(typeid(int8_t))) {
                static_cast<int8_t*>(static_cast<void*>(mei.data()))[current_variable_array_index++] = static_cast<int8_t>(val);
            } else if (val_type == std::type_index(typeid(uint8_t))) {
                static_cast<uint8_t*>(static_cast<void*>(mei.data()))[current_variable_array_index++] = static_cast<uint8_t>(val);
            } else {
                THROW exception::JSONError("Model contains macro environment property '%s' of unsupported type '%s', "
                    "in JSONStateReader::parse()\n", lastKey.c_str(), val_type.name());
            }
        } else if (mode.top() == AgentInstance) {
            const std::shared_ptr<AgentVector> &pop = agents_map.at({current_agent, current_state});
            AgentVector::Agent instance = pop->back();
            char *data = static_cast<char*>(const_cast<void*>(std::static_pointer_cast<const AgentVector>(pop)->data(lastKey)));
            const VariableMap& agentVariables = pop->getVariableMetaData();
            const auto var_data = agentVariables.at(lastKey);
            const size_t v_size = var_data.type_size * var_data.elements;
            const std::type_index val_type = var_data.type;
            if (val_type == std::type_index(typeid(float))) {
                const float t = static_cast<float>(val);
                memcpy(data + ((pop->size() - 1) * v_size) + (var_data.type_size * current_variable_array_index++), &t, var_data.type_size);
            } else if (val_type == std::type_index(typeid(double))) {
                const double t = static_cast<double>(val);
                memcpy(data + ((pop->size() - 1) * v_size) + (var_data.type_size * current_variable_array_index++), &t, var_data.type_size);
            } else if (val_type == std::type_index(typeid(int64_t))) {
                const int64_t t = static_cast<int64_t>(val);
                memcpy(data + ((pop->size() - 1) * v_size) + (var_data.type_size * current_variable_array_index++), &t, var_data.type_size);
            } else if (val_type == std::type_index(typeid(uint64_t))) {
                const uint64_t t = static_cast<uint64_t>(val);
                memcpy(data + ((pop->size() - 1) * v_size) + (var_data.type_size * current_variable_array_index++), &t, var_data.type_size);
            } else if (val_type == std::type_index(typeid(int32_t))) {
                const int32_t t = static_cast<int32_t>(val);
                memcpy(data + ((pop->size() - 1) * v_size) + (var_data.type_size * current_variable_array_index++), &t, var_data.type_size);
            } else if (val_type == std::type_index(typeid(uint32_t))) {
                const uint32_t t = static_cast<uint32_t>(val);
                memcpy(data + ((pop->size() - 1) * v_size) + (var_data.type_size * current_variable_array_index++), &t, var_data.type_size);
            } else if (val_type == std::type_index(typeid(int16_t))) {
                const int16_t t = static_cast<int16_t>(val);
                memcpy(data + ((pop->size() - 1) * v_size) + (var_data.type_size * current_variable_array_index++), &t, var_data.type_size);
            } else if (val_type == std::type_index(typeid(uint16_t))) {
                const uint16_t t = static_cast<uint16_t>(val);
                memcpy(data + ((pop->size() - 1) * v_size) + (var_data.type_size * current_variable_array_index++), &t, var_data.type_size);
            } else if (val_type == std::type_index(typeid(int8_t))) {
                const int8_t t = static_cast<int8_t>(val);
                memcpy(data + ((pop->size() - 1) * v_size) + (var_data.type_size * current_variable_array_index++), &t, var_data.type_size);
            } else if (val_type == std::type_index(typeid(uint8_t))) {
                const uint8_t t = static_cast<uint8_t>(val);
                memcpy(data + ((pop->size() - 1) * v_size) + (var_data.type_size * current_variable_array_index++), &t, var_data.type_size);
            } else {
                THROW exception::JSONError("Model contains agent variable '%s:%s' of unsupported type '%s', "
                    "in JSONStateReader::parse()\n", current_agent.c_str(), lastKey.c_str(), val_type.name());
            }
        }  else if (mode.top() == CUDACfg || mode.top() == SimCfg || mode.top() == Stats) {
            // Not useful
            // Cfg are loaded by counter
        } else {
            THROW exception::JSONError("Unexpected value whilst parsing input file '%s'.\n", filename.c_str());
        }
        if (isArray == VariableArray) {
            mode.push(isArray);
        } else {
            current_variable_array_index = 0;  // Didn't actually want to increment it above, because not in an array
        }
        return true;
    }
    bool null() {
        bool is_array = false;
        if (mode.top() == VariableArray) {
            mode.pop();
            is_array = true;
        }
        // Emit a warning if the verbosity level is high enough
        if (verbosity > Verbosity::Quiet) {
            if (mode.top() == Environment) {
                fprintf(stderr, "Warning: JSON Environment property '%s' contains NULL, this has been interpreted as NaN (but may represent Inf).\n", lastKey.c_str());
            } else if (mode.top() == MacroEnvironment) {
                fprintf(stderr, "Warning: JSON MacroEnvironment property '%s' contains NULL, this has been interpreted as NaN (but may represent Inf).\n", lastKey.c_str());
            } else if (mode.top() == AgentInstance) {
                fprintf(stderr, "Warning: JSON Agent '%s' variable '%s' contains NULL, this has been interpreted as NaN (but may represent Inf).\n", current_agent.c_str(), lastKey.c_str());
            } else {
                fprintf(stderr, "Warning: JSON state item '%s' contains NULL, this has been interpreted as NaN (but may represent Inf).\n", lastKey.c_str());
            }
        }
        if (is_array) {
            mode.push(VariableArray);
        }
        return processValue<number_float_t>(std::numeric_limits<number_float_t>::quiet_NaN());
    }
    bool boolean(bool b) { return processValue<bool>(b); }
    bool number_integer(number_integer_t i) { return processValue<number_integer_t>(i); }
    bool number_unsigned(number_unsigned_t u) { return processValue<number_unsigned_t>(u); }
    bool number_float(number_float_t d, const string_t&) { return processValue<number_float_t>(d); }

    bool string(string_t &) {
        // String is only possible in config, and config is not processed by this handler
        if (mode.top() == SimCfg || mode.top() == CUDACfg) {
            return true;
        }
        THROW exception::JSONError("Unexpected string whilst parsing input file '%s'.\n", filename.c_str());
    }
    bool binary(binary_t&) {
        THROW exception::JSONError("Unexpected binary value whilst parsing input file '%s'.\n", filename.c_str());
    }
    bool start_object(size_t) {
        if (mode.empty()) {
            mode.push(Root);
        } else if (mode.top() == Root) {
            if (lastKey == "config") {
                mode.push(Config);
            } else if (lastKey == "stats") {
                mode.push(Stats);
            } else if (lastKey == "environment") {
                mode.push(Environment);
            } else if (lastKey == "macro_environment") {
                mode.push(MacroEnvironment);
            } else if (lastKey == "agents") {
                mode.push(Agents);
            } else {
                THROW exception::JSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        } else if (mode.top() == Config) {
            if (lastKey == "simulation") {
                mode.push(SimCfg);
            } else if (lastKey == "cuda") {
                mode.push(CUDACfg);
            } else {
                THROW exception::JSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        } else if (mode.top() == Agents) {
            current_agent = lastKey;
            mode.push(Agent);
        } else if (mode.top() == State) {
            mode.push(AgentInstance);
            auto f = agents_map.find({ current_agent, current_state });
            if (f == agents_map.end()) {
                THROW exception::JSONError("Input file '%s' contains data for agent:state combination '%s:%s' not found in model description hierarchy.\n", filename.c_str(), current_agent.c_str(), current_state.c_str());
            }
            f->second->push_back();
        } else {
            THROW exception::JSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
    bool key(string_t &str) {
        lastKey = str;
        return true;
    }
    bool end_object() {
        mode.pop();
        return true;
    }
    bool start_array(size_t) {
        if (current_variable_array_index != 0) {
            THROW exception::JSONError("Array start when current_variable_array_index !=0, in file '%s'. This should never happen.\n", filename.c_str());
        }
        if (mode.top() == AgentInstance) {
            mode.push(VariableArray);
        } else if (mode.top() == Environment) {
            mode.push(VariableArray);
        } else if (mode.top() == MacroEnvironment) {
            mode.push(VariableArray);
        } else if (mode.top() == Agent) {
            current_state = lastKey;
            mode.push(State);
        } else {
            THROW exception::JSONError("Unexpected array start whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
    bool end_array() {
        if (mode.top() == VariableArray) {
            mode.pop();
            if (mode.top() == Environment) {
                // Confirm env array had correct number of elements
                const auto prop = model->environment->properties.at(lastKey);
                if (current_variable_array_index != prop.data.elements) {
                    THROW exception::JSONError("Input file contains environment property '%s' with %u elements expected %u,"
                        "in JSONStateReader::parse()\n", lastKey.c_str(), current_variable_array_index, prop.data.elements);
                }
            } else if (mode.top() == MacroEnvironment) {
                // Confirm macro env array had correct number of elements
                const auto macro_prop = model->environment->macro_properties.at(lastKey);
                const unsigned int macro_prop_elements = std::accumulate(macro_prop.elements.begin(), macro_prop.elements.end(), 1, std::multiplies<unsigned int>());
                if (current_variable_array_index != macro_prop_elements) {
                    THROW exception::JSONError("Input file contains environment macro property '%s' with %u elements expected %u,"
                        "in JSONStateReader::parse()\n", lastKey.c_str(), current_variable_array_index, macro_prop_elements);
                }
            }
            current_variable_array_index = 0;
        } else {
            mode.pop();
        }
        return true;
    }
    // called when a parse error occurs; byte position, the last token, and an exception is passed
    bool parse_error(std::size_t /*position*/, const std::string& /*last_token*/, const nlohmann::json::exception& ex) {
        THROW exception::JSONError(ex.what());
    }
};
/**
 * This is a trivial parser, it builds a map of the number of agents in each state
 * This allows the agent statelists to be preallocated
 * It also reads the config blocks, so that device can be init before we do environment
 */
class JSONStateReader_agentsize_counter : public nlohmann::json_sax<nlohmann::json> {
    enum Mode{ Nop, Root, Config, Stats, SimCfg, CUDACfg, Environment, MacroEnvironment, Agents, Agent, State, AgentInstance, VariableArray };
    std::stack<Mode> mode;
    std::string lastKey;
    unsigned int currentIndex = 0;
    std::string filename;
    std::string current_agent = "";
    std::string current_state = "";
    util::StringPairUnorderedMap<unsigned int> agentstate_counts;
    std::unordered_map<std::string, std::any> &simulation_config;
    std::unordered_map<std::string, std::any> &cuda_config;
    Verbosity verbosity;

 public:
     util::StringPairUnorderedMap<unsigned int> getAgentCounts() const {
        return agentstate_counts;
    }
    explicit JSONStateReader_agentsize_counter(const std::string &_filename,
        std::unordered_map<std::string, std::any> &_simulation_config,
        std::unordered_map<std::string, std::any> &_cuda_config,
        Verbosity _verbosity)
        : filename(_filename)
        , simulation_config(_simulation_config)
        , cuda_config(_cuda_config)
        , verbosity(_verbosity) { }

    template<typename T>
    bool processValue(const T val) {
        Mode isArray = Nop;
        if (mode.top() == VariableArray) {
            isArray = mode.top();
            mode.pop();
        }
        if (mode.top() == SimCfg) {
            if (lastKey == "truncate_log_files") {
                simulation_config.emplace(lastKey, static_cast<bool>(val));
            } else if (lastKey == "random_seed") {
                simulation_config.emplace(lastKey, static_cast<uint64_t>(val));
            } else if (lastKey == "steps") {
                simulation_config.emplace(lastKey, static_cast<unsigned int>(val));
            } else if (lastKey == "timing") {
                simulation_config.emplace(lastKey, static_cast<bool>(val));
            } else if (lastKey == "verbosity") {
                simulation_config.emplace(lastKey, static_cast<flamegpu::Verbosity>(static_cast<int>(val)));
            } else if (lastKey == "console_mode") {
#ifdef FLAMEGPU_VISUALISATION
                simulation_config.emplace(lastKey, static_cast<bool>(val));
#else
                fprintf(stderr, "Warning: Cannot configure 'console_mode' with input file '%s', FLAMEGPU2 library has not been built with visualisation support enabled.\n", filename.c_str());
#endif
            } else {
                THROW exception::JSONError("Unexpected simulation config item '%s' in input file '%s'.\n", lastKey.c_str(), filename.c_str());
            }
        } else if (mode.top() == CUDACfg) {
            if (lastKey == "device_id") {
                cuda_config.emplace(lastKey, static_cast<int>(val));
            } else if (lastKey == "inLayerConcurrency") {
                cuda_config.emplace(lastKey, static_cast<bool>(val));
            } else {
                THROW exception::JSONError("Unexpected CUDA config item '%s' in input file '%s'.\n", lastKey.c_str(), filename.c_str());
            }
        }  else {
            // Not useful
            // Everything else is loaded by main handler
        }
        if (isArray == VariableArray) {
            mode.push(isArray);
        }
        return true;
    }
    bool null() { return processValue<number_float_t>(std::numeric_limits<number_float_t>::quiet_NaN()); }
    bool boolean(bool b) { return processValue<bool>(b); }
    bool number_integer(number_integer_t i) { return processValue<number_integer_t>(i); }
    bool number_unsigned(number_unsigned_t u) { return processValue<number_unsigned_t>(u); }
    bool number_float(number_float_t d, const string_t&) { return processValue<number_float_t>(d); }

    bool string(string_t& str) {
        if (mode.top() == SimCfg) {
            if (lastKey == "input_file") {
                if (filename != str && str[0] != '\0')
                    if (verbosity > Verbosity::Quiet)
                        fprintf(stderr, "Warning: Input file '%s' refers to second input file '%s', this will not be loaded.\n", filename.c_str(), str.c_str());
                // sim_instance->SimulationConfig().input_file = str;
            } else if (lastKey == "step_log_file" ||
                       lastKey == "exit_log_file" ||
                       lastKey == "common_log_file") {
                simulation_config.emplace(lastKey, str);
            }
        }
        return true;
    }
    bool binary(binary_t&) {
        THROW exception::JSONError("Unexpected binary value whilst parsing input file '%s'.\n", filename.c_str());
    }
    bool start_object(size_t) {
        if (mode.empty()) {
            mode.push(Root);
        } else if (mode.top() == Root) {
            if (lastKey == "config") {
                mode.push(Config);
            } else if (lastKey == "stats") {
                mode.push(Stats);
            } else if (lastKey == "environment") {
                mode.push(Environment);
            } else if (lastKey == "macro_environment") {
                mode.push(MacroEnvironment);
            } else if (lastKey == "agents") {
                mode.push(Agents);
            } else {
                THROW exception::JSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        } else if (mode.top() == Config) {
            if (lastKey == "simulation") {
                mode.push(SimCfg);
            } else if (lastKey == "cuda") {
                mode.push(CUDACfg);
            } else {
                THROW exception::JSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        }  else if (mode.top() == Agents) {
            current_agent = lastKey;
            mode.push(Agent);
        } else if (mode.top() == State) {
            agentstate_counts[{current_agent, current_state}]++;
            mode.push(AgentInstance);
        } else {
            THROW exception::JSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
    bool key(string_t &str) {
        lastKey = str;
        return true;
    }
    bool end_object() {
        mode.pop();
        return true;
    }
    bool start_array(size_t) {
        if (currentIndex != 0) {
            THROW exception::JSONError("Array start when current_variable_array_index !=0, in file '%s'. This should never happen.\n", filename.c_str());
        }
        if (mode.top() == AgentInstance) {
            mode.push(VariableArray);
        } else if (mode.top() == Environment) {
            mode.push(VariableArray);
        } else if (mode.top() == MacroEnvironment) {
            mode.push(VariableArray);
        } else if (mode.top() == Agent) {
            current_state = lastKey;
            mode.push(State);
        } else {
            THROW exception::JSONError("Unexpected array start whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
    bool end_array() {
        if (mode.top() == VariableArray) {
            currentIndex = 0;
        }
        mode.pop();
        return true;
    }
    // called when a parse error occurs; byte position, the last token, and an exception is passed
    bool parse_error(std::size_t /*position*/, const std::string& /*last_token*/, const nlohmann::json::exception& ex) {
        THROW exception::JSONError(ex.what());
    }
};

void JSONStateReader::parse(const std::string &input_file, const std::shared_ptr<const ModelData> &model, Verbosity verbosity) {
    resetCache();

    std::ifstream in(input_file, std::ios::in | std::ios::binary);
    if (!in) {
        THROW exception::JSONError("Unable to open file '%s' for reading, in JSONStateReader::parse().", input_file.c_str());
    }
    JSONStateReader_agentsize_counter agentcounter(input_file, simulation_config, cuda_config, verbosity);
    if (!nlohmann::json::sax_parse(in, &agentcounter)) {
        THROW exception::JSONError("Parsing input file '%s' failed, in JSONStateReader::parse()\n", input_file.c_str());
    }
    const util::StringPairUnorderedMap<unsigned int> agentCounts = agentcounter.getAgentCounts();
    // Use this to preallocate the agent statelists
    for (auto &it : agentCounts) {
        const auto& agent = model->agents.find(it.first.first);
        if (agent == model->agents.end() || agent->second->states.find(it.first.second) == agent->second->states.end()) {
            THROW exception::InvalidAgentState("Agent '%s' with state '%s', found in input file '%s', is not part of the model description hierarchy, "
                "in JSONStateReader::parse()\n Ensure the input file is for the correct model.\n", it.first.first.c_str(), it.first.second.c_str(), input_file.c_str());
        }
        auto [_it, _] = agents_map.emplace(it.first, std::make_shared<AgentVector>(*agent->second));
        _it->second->reserve(it.second);
    }
    // Reset the stream
    in.clear();
    in.seekg(0);
    // Read in the file data
    JSONStateReader_impl handler(input_file, model, env_init, macro_env_init, agents_map, verbosity);
    if (!nlohmann::json::sax_parse(in, &handler)) {
        THROW exception::JSONError("Parsing input file '%s' failed, in JSONStateReader::parse()\n", input_file.c_str());
    }
    // Mark input as loaded
    this->input_filepath = input_file;
}

}  // namespace io
}  // namespace flamegpu
