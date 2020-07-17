#include "flamegpu/io/jsonReader.h"

#include <rapidjson/stream.h>
#include <rapidjson/reader.h>
#include <stack>
#include <fstream>
#include <string>
#include <unordered_map>
#include <cerrno>

#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/gpu/CUDAAgentModel.h"

jsonReader::jsonReader(
    const std::string &model_name,
    const unsigned int &sim_instance_id,
    const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model_state,
    const std::string &input,
    Simulation *sim_instance)
    : StateReader(model_name, sim_instance_id, model_state, input, sim_instance) {}
/**
 * This is the main sax style parser for the json state
 * It stores it's current position within the hierarchy with mode, lastKey and current_variable_array_index
 */
class jsonReader_impl : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, jsonReader_impl>  {
    enum Mode{ Nop, Root, Config, Stats, SimCfg, CUDACfg, Environment, Agents, Agent, State, AgentInstance, VariableArray };
    std::stack<Mode> mode;
    std::string lastKey;
    std::string filename;
    EnvironmentManager &env_manager;
    unsigned int sim_instance_id;
    /**
     * Used for setting agent values
     */
    const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model_state;
    /**
     * Tracks current position reading variable array
     */
    unsigned int current_variable_array_index = 0;
    /**
     * Set when we enter an agent instance
     */
    unsigned int current_agent_index = 0;
    /**
     * Set when we enter an agent
     */
    std::string current_agent;
    /**
     * Set when we enter a state
     */
    std::string current_state;

 public:
    jsonReader_impl(const std::string &_filename, EnvironmentManager &em, unsigned int _sim_instance_id,
        const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &_model_state)
        : filename(_filename)
        , env_manager(em)
        , sim_instance_id(_sim_instance_id)
        , model_state(_model_state) { }
    template<typename T>
    bool processValue(const T&val) {
        Mode isArray = Nop;
        if (mode.top() == VariableArray) {
            isArray = mode.top();
            mode.pop();
        }
        if (mode.top() == Environment) {
            const EnvironmentManager::NamePair np = { sim_instance_id , lastKey };
            const std::type_index val_type = env_manager.type(np);
            if (val_type == std::type_index(typeid(float))) {
                env_manager.set<float>(np, current_variable_array_index++, static_cast<float>(val));
            } else if (val_type == std::type_index(typeid(double))) {
                env_manager.set<double>(np, current_variable_array_index++, static_cast<double>(val));
            } else if (val_type == std::type_index(typeid(int64_t))) {
                env_manager.set<int64_t>(np, current_variable_array_index++, static_cast<int64_t>(val));
            } else if (val_type == std::type_index(typeid(uint64_t))) {
                env_manager.set<uint64_t>(np, current_variable_array_index++, static_cast<uint64_t>(val));
            } else if (val_type == std::type_index(typeid(int32_t))) {
                env_manager.set<int32_t>(np, current_variable_array_index++, static_cast<int32_t>(val));
            } else if (val_type == std::type_index(typeid(uint32_t))) {
                env_manager.set<uint32_t>(np, current_variable_array_index++, static_cast<uint32_t>(val));
            } else if (val_type == std::type_index(typeid(int16_t))) {
                env_manager.set<int16_t>(np, current_variable_array_index++, static_cast<int16_t>(val));
            } else if (val_type == std::type_index(typeid(uint16_t))) {
                env_manager.set<uint16_t>(np, current_variable_array_index++, static_cast<uint16_t>(val));
            } else if (val_type == std::type_index(typeid(int8_t))) {
                env_manager.set<int8_t>(np, current_variable_array_index++, static_cast<int8_t>(val));
            } else if (val_type == std::type_index(typeid(uint8_t))) {
                env_manager.set<uint8_t>(np, current_variable_array_index++, static_cast<uint8_t>(val));
            } else {
                THROW RapidJSONError("Model contains environment property '%s' of unsupported type '%s', "
                    "in jsonReader::parse()\n", lastKey.c_str(), val_type.name());
            }
        } else if (mode.top() == AgentInstance) {
            const std::shared_ptr<AgentPopulation> &pop = model_state.at(current_agent);
            ::AgentInstance instance = pop->getInstanceAt(current_agent_index, current_state);
            const auto &agentVariables = model_state.at(current_agent)->getAgentDescription().variables;
            const std::type_index val_type = agentVariables.at(lastKey).type;
            if (val_type == std::type_index(typeid(float))) {
                instance.setVariable<float>(lastKey, current_variable_array_index++, static_cast<float>(val));
            } else if (val_type == std::type_index(typeid(double))) {
                instance.setVariable<double>(lastKey, current_variable_array_index++, static_cast<double>(val));
            } else if (val_type == std::type_index(typeid(int64_t))) {
                instance.setVariable<int64_t>(lastKey, current_variable_array_index++, static_cast<int64_t>(val));
            } else if (val_type == std::type_index(typeid(uint64_t))) {
                instance.setVariable<uint64_t>(lastKey, current_variable_array_index++, static_cast<uint64_t>(val));
            } else if (val_type == std::type_index(typeid(int32_t))) {
                instance.setVariable<int32_t>(lastKey, current_variable_array_index++, static_cast<int32_t>(val));
            } else if (val_type == std::type_index(typeid(uint32_t))) {
                instance.setVariable<uint32_t>(lastKey, current_variable_array_index++, static_cast<uint32_t>(val));
            } else if (val_type == std::type_index(typeid(int16_t))) {
                instance.setVariable<int16_t>(lastKey, current_variable_array_index++, static_cast<int16_t>(val));
            } else if (val_type == std::type_index(typeid(uint16_t))) {
                instance.setVariable<uint16_t>(lastKey, current_variable_array_index++, static_cast<uint16_t>(val));
            } else if (val_type == std::type_index(typeid(int8_t))) {
                instance.setVariable<int8_t>(lastKey, current_variable_array_index++, static_cast<int8_t>(val));
            } else if (val_type == std::type_index(typeid(uint8_t))) {
                instance.setVariable<uint8_t>(lastKey, current_variable_array_index++, static_cast<uint8_t>(val));
            } else {
                THROW RapidJSONError("Model contains environment property '%s' of unsupported type '%s', "
                    "in jsonReader::parse()\n", lastKey.c_str(), val_type.name());
            }
        }  else if (mode.top() == CUDACfg || mode.top() == SimCfg || mode.top() == Stats) {
            // Not useful
            // Cfg are loaded by counter
        } else {
            THROW RapidJSONError("Unexpected value whilst parsing input file '%s'.\n", filename.c_str());
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
        THROW RapidJSONError("Unexpected string whilst parsing input file '%s'.\n", filename.c_str());
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
                THROW RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        } else if (mode.top() == Config) {
            if (lastKey == "simulation") {
                mode.push(SimCfg);
            } else if (lastKey == "cuda") {
                mode.push(CUDACfg);
            } else {
                THROW RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        } else if (mode.top() == Agents) {
            current_agent = lastKey;
            mode.push(Agent);
        } else if (mode.top() == State) {
            mode.push(AgentInstance);
            // Currently agent pop is annoying, we need to call this to actually create the next agent
            // We can't hold an empty AgentInstance, so we will recover it later with current_agent_index
            model_state.at(current_agent)->getNextInstance(current_state);
        } else {
            THROW RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
    bool Key(const char* str, rapidjson::SizeType, bool) {
        lastKey = str;
        return true;
    }
    bool EndObject(rapidjson::SizeType) {
        if (mode.top() == AgentInstance) {
            current_agent_index++;
        }
        mode.pop();
        return true;
    }
    bool StartArray() {
        if (current_variable_array_index != 0) {
            THROW RapidJSONError("Array start when current_variable_array_index !=0, in file '%s'. This should never happen.\n", filename.c_str());
        }
        if (mode.top() == AgentInstance) {
            mode.push(VariableArray);
        } else if (mode.top() == Environment) {
            mode.push(VariableArray);
        } else if (mode.top() == Agent) {
            current_state = lastKey;
            mode.push(State);
        } else {
            THROW RapidJSONError("Unexpected array start whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
    bool EndArray(rapidjson::SizeType) {
        if (mode.top() == VariableArray) {
            current_variable_array_index = 0;
        } else if (mode.top() == State) {
            current_agent_index = 0;
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
class jsonReader_agentsize_counter : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, jsonReader_impl>  {
    enum Mode{ Nop, Root, Config, Stats, SimCfg, CUDACfg, Environment, Agents, Agent, State, AgentInstance, VariableArray };
    std::stack<Mode> mode;
    std::string lastKey;
    unsigned int currentIndex = 0;
    std::string filename;
    std::string current_agent = "";
    std::string current_state = "";
    struct StringPairHash {
        size_t operator()(const std::pair<std::string, std::string>& k) const {
            return std::hash<std::string>()(k.first) ^
                (std::hash<std::string>()(k.second) << 1);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, unsigned int, StringPairHash> agentstate_counts;
    Simulation *sim_instance;
    CUDAAgentModel *cudamodel_instance;

 public:
    std::unordered_map<std::string, unsigned int> getAgentCounts() const {
        std::unordered_map<std::string, unsigned int> rtn;
        for (const auto &asc : agentstate_counts) {
            rtn[asc.first.first] = rtn[asc.first.first] > asc.second ? rtn.at(asc.first.first) : asc.second;
        }
        return rtn;
    }
    explicit jsonReader_agentsize_counter(const std::string &_filename, Simulation *_sim_instance)
        : filename(_filename)
        , sim_instance(_sim_instance)
        , cudamodel_instance(dynamic_cast<CUDAAgentModel*>(_sim_instance)) { }

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
                    THROW RapidJSONError("Unexpected simulation config item '%s' in input file '%s'.\n", lastKey.c_str(), filename.c_str());
                }
            }
        } else if (mode.top() == CUDACfg) {
            if (cudamodel_instance) {
                if (lastKey == "device_id") {
                    cudamodel_instance->CUDAConfig().device_id = static_cast<unsigned int>(val);
                } else {
                    THROW RapidJSONError("Unexpected CUDA config item '%s' in input file '%s'.\n", lastKey.c_str(), filename.c_str());
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
                    if (filename != str)
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
                THROW RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        } else if (mode.top() == Config) {
            if (lastKey == "simulation") {
                mode.push(SimCfg);
            } else if (lastKey == "cuda") {
                mode.push(CUDACfg);
            } else {
                THROW RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        }  else if (mode.top() == Agents) {
            current_agent = lastKey;
            mode.push(Agent);
        } else if (mode.top() == State) {
            agentstate_counts[{current_agent, current_state}]++;
            mode.push(AgentInstance);
        } else {
            THROW RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
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
            THROW RapidJSONError("Array start when current_variable_array_index !=0, in file '%s'. This should never happen.\n", filename.c_str());
        }
        if (mode.top() == AgentInstance) {
            mode.push(VariableArray);
        } else if (mode.top() == Environment) {
            mode.push(VariableArray);
        } else if (mode.top() == Agent) {
            current_state = lastKey;
            mode.push(State);
        } else {
            THROW RapidJSONError("Unexpected array start whilst parsing input file '%s'.\n", filename.c_str());
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

int jsonReader::parse() {
    std::ifstream in(inputFile, std::ios::in | std::ios::binary);
    if (!in) {
        THROW RapidJSONError("Unable to open file '%s' for reading.\n", inputFile.c_str());
    }
    jsonReader_agentsize_counter agentcounter(inputFile, sim_instance);
    jsonReader_impl handler(inputFile, EnvironmentManager::getInstance(), sim_instance_id, model_state);
    std::string filestring = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    rapidjson::StringStream filess(filestring.c_str());
    rapidjson::Reader reader;
    // First parse the file and simply count the size of agent list
    reader.Parse(filess, agentcounter);
    const auto agentCounts = agentcounter.getAgentCounts();
    // Use this to preallocate the agent statelists
    for (auto &agt : agentCounts) {
        if (agt.second > AgentPopulation::DEFAULT_POPULATION_SIZE) {
            model_state.at(agt.first)->setStateListCapacity(agt.second);
        }
    }
    // Reset the string stream
    filess = rapidjson::StringStream(filestring.c_str());
    // Read in the file data
    reader.Parse(filess, handler);
    return 0;
}
