#include "flamegpu/io/JSONStateReader.h"

#include <rapidjson/stream.h>
#include <rapidjson/reader.h>
#include <rapidjson/error/en.h>
#include <stack>
#include <fstream>
#include <string>
#include <unordered_map>
#include <cerrno>

#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/pop/AgentVector.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/gpu/CUDASimulation.h"
#include "flamegpu/util/StringPair.h"

namespace flamegpu {
namespace io {

JSONStateReader::JSONStateReader(
    const std::string &model_name,
    const std::unordered_map<std::string, EnvironmentDescription::PropData> &env_desc,
    util::StringUint32PairUnorderedMap<util::Any> &env_init,
    util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> &model_state,
    const std::string &input,
    Simulation *sim_instance)
    : StateReader(model_name, env_desc, env_init, model_state, input, sim_instance) {}
/**
 * This is the main sax style parser for the json state
 * It stores it's current position within the hierarchy with mode, lastKey and current_variable_array_index
 */
class JSONStateReader_impl : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, JSONStateReader_impl>  {
    enum Mode{ Nop, Root, Config, Stats, SimCfg, CUDACfg, Environment, Agents, Agent, State, AgentInstance, VariableArray };
    std::stack<Mode> mode;
    std::string lastKey;
    std::string filename;
    const std::unordered_map<std::string, EnvironmentDescription::PropData> env_desc;
    util::StringUint32PairUnorderedMap<util::Any> &env_init;
    /**
     * Used for setting agent values
     */
    util::StringPairUnorderedMap<std::shared_ptr<AgentVector>>&model_state;
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
        const std::unordered_map<std::string, EnvironmentDescription::PropData> &_env_desc,
        util::StringUint32PairUnorderedMap<util::Any> &_env_init,
        util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> &_model_state)
        : filename(_filename)
        , env_desc(_env_desc)
        , env_init(_env_init)
        , model_state(_model_state) { }
    template<typename T>
    bool processValue(const T&val) {
        Mode isArray = Nop;
        if (mode.top() == VariableArray) {
            isArray = mode.top();
            mode.pop();
        }
        if (mode.top() == Environment) {
            const auto it = env_desc.find(lastKey);
            if (it == env_desc.end()) {
                THROW exception::RapidJSONError("Input file contains unrecognised environment property '%s',"
                    "in JSONStateReader::parse()\n", lastKey.c_str());
            }
            if (env_init.find(make_pair(lastKey, current_variable_array_index)) != env_init.end()) {
                THROW exception::RapidJSONError("Input file contains environment property '%s' multiple times, "
                    "in JSONStateReader::parse()\n", lastKey.c_str());
            }
            const std::type_index val_type = it->second.data.type;
            if (val_type == std::type_index(typeid(float))) {
                const float t = static_cast<float>(val);
                env_init.emplace(make_pair(lastKey, current_variable_array_index++), util::Any(&t, sizeof(float), val_type, 1));
            } else if (val_type == std::type_index(typeid(double))) {
                const double t = static_cast<double>(val);
                env_init.emplace(make_pair(lastKey, current_variable_array_index++), util::Any(&t, sizeof(double), val_type, 1));
            } else if (val_type == std::type_index(typeid(int64_t))) {
                const int64_t t = static_cast<int64_t>(val);
                env_init.emplace(make_pair(lastKey, current_variable_array_index++), util::Any(&t, sizeof(int64_t), val_type, 1));
            } else if (val_type == std::type_index(typeid(uint64_t))) {
                const uint64_t t = static_cast<uint64_t>(val);
                env_init.emplace(make_pair(lastKey, current_variable_array_index++), util::Any(&t, sizeof(uint64_t), val_type, 1));
            } else if (val_type == std::type_index(typeid(int32_t))) {
                const int32_t t = static_cast<int32_t>(val);
                env_init.emplace(make_pair(lastKey, current_variable_array_index++), util::Any(&t, sizeof(int32_t), val_type, 1));
            } else if (val_type == std::type_index(typeid(uint32_t))) {
                const uint32_t t = static_cast<uint32_t>(val);
                env_init.emplace(make_pair(lastKey, current_variable_array_index++), util::Any(&t, sizeof(uint32_t), val_type, 1));
            } else if (val_type == std::type_index(typeid(int16_t))) {
                const int16_t t = static_cast<int16_t>(val);
                env_init.emplace(make_pair(lastKey, current_variable_array_index++), util::Any(&t, sizeof(int16_t), val_type, 1));
            } else if (val_type == std::type_index(typeid(uint16_t))) {
                const uint16_t t = static_cast<uint16_t>(val);
                env_init.emplace(make_pair(lastKey, current_variable_array_index++), util::Any(&t, sizeof(uint16_t), val_type, 1));
            } else if (val_type == std::type_index(typeid(int8_t))) {
                const int8_t t = static_cast<int8_t>(val);
                env_init.emplace(make_pair(lastKey, current_variable_array_index++), util::Any(&t, sizeof(int8_t), val_type, 1));
            } else if (val_type == std::type_index(typeid(uint8_t))) {
                const uint8_t t = static_cast<uint8_t>(val);
                env_init.emplace(make_pair(lastKey, current_variable_array_index++), util::Any(&t, sizeof(uint8_t), val_type, 1));
            } else {
                THROW exception::RapidJSONError("Model contains environment property '%s' of unsupported type '%s', "
                    "in JSONStateReader::parse()\n", lastKey.c_str(), val_type.name());
            }
        } else if (mode.top() == AgentInstance) {
            const std::shared_ptr<AgentVector> &pop = model_state.at({current_agent, current_state});
            AgentVector::Agent instance = pop->back();
            char *data = static_cast<char*>(const_cast<void*>(static_cast<std::shared_ptr<const AgentVector>>(pop)->data(lastKey)));
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
                THROW exception::RapidJSONError("Model contains agent variable '%s:%s' of unsupported type '%s', "
                    "in JSONStateReader::parse()\n", current_agent.c_str(), lastKey.c_str(), val_type.name());
            }
        }  else if (mode.top() == CUDACfg || mode.top() == SimCfg || mode.top() == Stats) {
            // Not useful
            // Cfg are loaded by counter
        } else {
            THROW exception::RapidJSONError("Unexpected value whilst parsing input file '%s'.\n", filename.c_str());
        }
        if (isArray == VariableArray) {
            mode.push(isArray);
        } else {
            current_variable_array_index = 0;  // Didn't actually want to increment it above, because not in an array
        }
        return true;
    }
    bool Null() { return true; }
    bool Bool(bool b) { return processValue<bool>(b); }
    bool Int(int i) { return processValue<int32_t>(i); }
    bool Uint(unsigned u) { return processValue<uint32_t>(u); }
    bool Int64(int64_t i) { return processValue<int64_t>(i); }
    bool Uint64(uint64_t u) { return processValue<uint64_t>(u); }
    bool Double(double d) { return processValue<double>(d); }
    bool String(const char*, rapidjson::SizeType, bool) {
        // String is only possible in config, and config is not processed by this handler
        if (mode.top() == SimCfg || mode.top() == CUDACfg) {
            return true;
        }
        THROW exception::RapidJSONError("Unexpected string whilst parsing input file '%s'.\n", filename.c_str());
    }
    bool StartObject() {
        if (mode.empty()) {
            mode.push(Root);
        } else if (mode.top() == Root) {
            if (lastKey == "config") {
                mode.push(Config);
            } else if (lastKey == "stats") {
                mode.push(Stats);
            } else if (lastKey == "environment") {
                mode.push(Environment);
            } else if (lastKey == "agents") {
                mode.push(Agents);
            } else {
                THROW exception::RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        } else if (mode.top() == Config) {
            if (lastKey == "simulation") {
                mode.push(SimCfg);
            } else if (lastKey == "cuda") {
                mode.push(CUDACfg);
            } else {
                THROW exception::RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        } else if (mode.top() == Agents) {
            current_agent = lastKey;
            mode.push(Agent);
        } else if (mode.top() == State) {
            mode.push(AgentInstance);
            auto f = model_state.find({ current_agent, current_state });
            if (f == model_state.end()) {
                THROW exception::RapidJSONError("Input file '%s' contains data for agent:state combination '%s:%s' not found in model description hierarchy.\n", filename.c_str());
            }
            f->second->push_back();
        } else {
            THROW exception::RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
    bool Key(const char* str, rapidjson::SizeType, bool) {
        lastKey = str;
        return true;
    }
    bool EndObject(rapidjson::SizeType) {
        mode.pop();
        return true;
    }
    bool StartArray() {
        if (current_variable_array_index != 0) {
            THROW exception::RapidJSONError("Array start when current_variable_array_index !=0, in file '%s'. This should never happen.\n", filename.c_str());
        }
        if (mode.top() == AgentInstance) {
            mode.push(VariableArray);
        } else if (mode.top() == Environment) {
            mode.push(VariableArray);
        } else if (mode.top() == Agent) {
            current_state = lastKey;
            mode.push(State);
        } else {
            THROW exception::RapidJSONError("Unexpected array start whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
    bool EndArray(rapidjson::SizeType) {
        if (mode.top() == VariableArray) {
            current_variable_array_index = 0;
        }
        mode.pop();
        return true;
    }
};
/**
 * This is a trivial parser, it builds a map of the number of agents in each state
 * This allows the agent statelists to be preallocated
 * It also reads the config blocks, so that device can be init before we do environment
 */
class JSONStateReader_agentsize_counter : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, JSONStateReader_impl>  {
    enum Mode{ Nop, Root, Config, Stats, SimCfg, CUDACfg, Environment, Agents, Agent, State, AgentInstance, VariableArray };
    std::stack<Mode> mode;
    std::string lastKey;
    unsigned int currentIndex = 0;
    std::string filename;
    std::string current_agent = "";
    std::string current_state = "";
    util::StringPairUnorderedMap<unsigned int> agentstate_counts;
    Simulation *sim_instance;
    CUDASimulation *cudamodel_instance;

 public:
     util::StringPairUnorderedMap<unsigned int> getAgentCounts() const {
        return agentstate_counts;
    }
    explicit JSONStateReader_agentsize_counter(const std::string &_filename, Simulation *_sim_instance)
        : filename(_filename)
        , sim_instance(_sim_instance)
        , cudamodel_instance(dynamic_cast<CUDASimulation*>(_sim_instance)) { }

    template<typename T>
    bool processValue(const T&val) {
        Mode isArray = Nop;
        if (mode.top() == VariableArray) {
            isArray = mode.top();
            mode.pop();
        }
        if (mode.top() == SimCfg) {
            if (sim_instance) {
                if (lastKey == "random_seed") {
                    sim_instance->SimulationConfig().random_seed = static_cast<unsigned int>(val);
                } else if (lastKey == "steps") {
                    sim_instance->SimulationConfig().steps = static_cast<unsigned int>(val);
                } else if (lastKey == "timing") {
                    sim_instance->SimulationConfig().timing = static_cast<bool>(val);
                } else if (lastKey == "verbose") {
                    sim_instance->SimulationConfig().verbose = static_cast<bool>(val);
                } else if (lastKey == "console_mode") {
#ifdef VISUALISATION
                    sim_instance->SimulationConfig().console_mode = static_cast<bool>(val);
#else
                    if (static_cast<bool>(val) == false) {
                        fprintf(stderr, "Warning: Cannot disable 'console_mode' with input file '%s', FLAMEGPU2 library has not been built with visualisation support enabled.\n", filename.c_str());
                    }
#endif
                } else {
                    THROW exception::RapidJSONError("Unexpected simulation config item '%s' in input file '%s'.\n", lastKey.c_str(), filename.c_str());
                }
            }
        } else if (mode.top() == CUDACfg) {
            if (cudamodel_instance) {
                if (lastKey == "device_id") {
                    cudamodel_instance->CUDAConfig().device_id = static_cast<unsigned int>(val);
                } else {
                    THROW exception::RapidJSONError("Unexpected CUDA config item '%s' in input file '%s'.\n", lastKey.c_str(), filename.c_str());
                }
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
    bool Null() { return true; }
    bool Bool(bool b) { return processValue<bool>(b); }
    bool Int(int i) { return processValue<int32_t>(i); }
    bool Uint(unsigned u) { return processValue<uint32_t>(u); }
    bool Int64(int64_t i) { return processValue<int64_t>(i); }
    bool Uint64(uint64_t u) { return processValue<uint64_t>(u); }
    bool Double(double d) { return processValue<double>(d); }
    bool String(const char*str, rapidjson::SizeType, bool) {
        if (mode.top() == SimCfg) {
            if (sim_instance) {
                if (lastKey == "input_file") {
                    if (filename != str && str[0] != '\0')
                        printf("Warning: Input file '%s' refers to second input file '%s', this will not be loaded.\n", filename.c_str(), str);
                    // sim_instance->SimulationConfig().input_file = str;
                }
            }
        }
        return true;
    }
    bool StartObject() {
        if (mode.empty()) {
            mode.push(Root);
        } else if (mode.top() == Root) {
            if (lastKey == "config") {
                mode.push(Config);
            } else if (lastKey == "stats") {
                mode.push(Stats);
            } else if (lastKey == "environment") {
                mode.push(Environment);
            } else if (lastKey == "agents") {
                mode.push(Agents);
            } else {
                THROW exception::RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        } else if (mode.top() == Config) {
            if (lastKey == "simulation") {
                mode.push(SimCfg);
            } else if (lastKey == "cuda") {
                mode.push(CUDACfg);
            } else {
                THROW exception::RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        }  else if (mode.top() == Agents) {
            current_agent = lastKey;
            mode.push(Agent);
        } else if (mode.top() == State) {
            agentstate_counts[{current_agent, current_state}]++;
            mode.push(AgentInstance);
        } else {
            THROW exception::RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
    bool Key(const char* str, rapidjson::SizeType, bool) {
        lastKey = str;
        return true;
    }
    bool EndObject(rapidjson::SizeType) {
        mode.pop();
        return true;
    }
    bool StartArray() {
        if (currentIndex != 0) {
            THROW exception::RapidJSONError("Array start when current_variable_array_index !=0, in file '%s'. This should never happen.\n", filename.c_str());
        }
        if (mode.top() == AgentInstance) {
            mode.push(VariableArray);
        } else if (mode.top() == Environment) {
            mode.push(VariableArray);
        } else if (mode.top() == Agent) {
            current_state = lastKey;
            mode.push(State);
        } else {
            THROW exception::RapidJSONError("Unexpected array start whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
    bool EndArray(rapidjson::SizeType) {
        if (mode.top() == VariableArray) {
            currentIndex = 0;
        }
        mode.pop();
        return true;
    }
};

int JSONStateReader::parse() {
    std::ifstream in(inputFile, std::ios::in | std::ios::binary);
    if (!in) {
        THROW exception::RapidJSONError("Unable to open file '%s' for reading.\n", inputFile.c_str());
    }
    JSONStateReader_agentsize_counter agentcounter(inputFile, sim_instance);
    JSONStateReader_impl handler(inputFile, env_desc, env_init, model_state);
    std::string filestring = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    rapidjson::StringStream filess(filestring.c_str());
    rapidjson::Reader reader;
    // First parse the file and simply count the size of agent list
    rapidjson::ParseResult pr1 = reader.Parse(filess, agentcounter);
    if (pr1.Code() != rapidjson::ParseErrorCode::kParseErrorNone) {
        THROW exception::RapidJSONError("Whilst parsing input file '%s', RapidJSON returned error: %s\n", inputFile.c_str(), rapidjson::GetParseError_En(pr1.Code()));
    }
    const util::StringPairUnorderedMap<unsigned int> agentCounts = agentcounter.getAgentCounts();
    // Use this to preallocate the agent statelists
    for (auto &agt : agentCounts) {
        auto f = model_state.find(agt.first);
        if (f!= model_state.end())
            f->second->reserve(agt.second);
    }
    // Reset the string stream
    filess = rapidjson::StringStream(filestring.c_str());
    // Read in the file data
    rapidjson::ParseResult pr2 = reader.Parse(filess, handler);
    if (pr2.Code() != rapidjson::ParseErrorCode::kParseErrorNone) {
        THROW exception::RapidJSONError("Whilst parsing input file '%s', RapidJSON returned error: %s\n", inputFile.c_str(), rapidjson::GetParseError_En(pr1.Code()));
    }
    return 0;
}

}  // namespace io
}  // namespace flamegpu
