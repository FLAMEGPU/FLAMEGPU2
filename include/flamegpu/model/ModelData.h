#ifndef INCLUDE_FLAMEGPU_MODEL_MODELDATA_H_
#define INCLUDE_FLAMEGPU_MODEL_MODELDATA_H_

#include <unordered_map>
#include <memory>
#include <typeindex>
#include <set>

#include "flamegpu/runtime/AgentFunction.h"
#include "flamegpu/runtime/flamegpu_host_api_macros.h"  // Todo replace with std/cub style fns (see AgentFunction.h)

class EnvironmentDescription;
class AgentDescription;
class MessageDescription;
class AgentFunctionDescription;

struct AgentData;
struct MessageData;
struct AgentFunctionData;
struct LayerData;

struct ModelData {
    static const std::string DEFAULT_STATE;  // "default"
    friend class ModelDescription;
    typedef unsigned int size_type;
    struct Variable {
        /**
         * Cannot explicitly specify template args of constructor, so we take redundant arg for implicit template
         */
        template<typename T>
        Variable(size_type _elements, T)
            : type(typeid(T)), type_size(sizeof(T)), elements(elements) { }
        const std::type_index type;
        const size_t type_size;
        const unsigned int elements;
    };
    typedef std::unordered_map<const std::string, Variable> VariableMap;
    typedef std::unordered_map<const std::string, std::shared_ptr<AgentData>> AgentMap;
    typedef std::unordered_map<const std::string, std::shared_ptr<MessageData>> MessageMap;
    typedef std::list<std::shared_ptr<LayerData>> LayerList;
    typedef std::set<FLAMEGPU_INIT_FUNCTION_POINTER> InitFunctionSet;
    typedef std::set<FLAMEGPU_STEP_FUNCTION_POINTER> StepFunctionSet;
    typedef std::set<FLAMEGPU_EXIT_FUNCTION_POINTER> ExitFunctionSet;
    typedef std::set<FLAMEGPU_EXIT_CONDITION_POINTER> ExitConditionSet;

    AgentMap agents;
    MessageMap messages;
    LayerList layers;
    InitFunctionSet initFunctions;
    StepFunctionSet stepFunctions;
    ExitFunctionSet exitFunctions;
    ExitConditionSet exitConditions;
    std::unique_ptr<EnvironmentDescription> environment; // TODO: Move this to same Data:Description format
    std::string name;

 protected:
    ModelData(const std::string &model_name);
};
struct AgentData {
    typedef std::unordered_map<const std::string, std::shared_ptr<AgentFunctionData>> FunctionMap;

    FunctionMap functions;
    ModelData::VariableMap variables;
    std::set<std::string> states;
    std::string initial_state;
    std::unique_ptr<AgentDescription> description;
    std::string name;

 protected:
    AgentData(const std::string &agent_name);
};
struct MessageData {

    ModelData::VariableMap variables;
    std::unique_ptr<MessageDescription> description;
    std::string name;

 protected:
    MessageData(const std::string &message_name);
};
struct AgentFunctionData {

    AgentFunctionWrapper *func;

    std::string initial_state = ModelData::DEFAULT_STATE;
    std::string end_state = ModelData::DEFAULT_STATE;

    std::weak_ptr<MessageData> message_input;
    std::weak_ptr<MessageData> message_output;

    std::weak_ptr<AgentData> agent_output;
    
    bool has_agent_death = false;

    std::weak_ptr<AgentDescription> owner;
    std::unique_ptr<AgentFunctionDescription> description;
    std::string name;

 protected:
    AgentFunctionData(const std::string &function_name, AgentFunctionWrapper *agent_function);
};
struct LayerData {
    std::set<std::weak_ptr<AgentFunctionData>> agent_functions;
    std::set<std::weak_ptr<AgentFunctionData>> host_functions;

 protected:
     LayerData() = default;
};
#endif  // INCLUDE_FLAMEGPU_MODEL_MODELDATA_H_
