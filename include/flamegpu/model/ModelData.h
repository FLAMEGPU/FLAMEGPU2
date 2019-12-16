#ifndef INCLUDE_FLAMEGPU_MODEL_MODELDATA_H_
#define INCLUDE_FLAMEGPU_MODEL_MODELDATA_H_

#include <unordered_map>
#include <list>
#include <memory>
#include <typeindex>
#include <set>
#include <string>

#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/runtime/AgentFunction.h"
#include "flamegpu/runtime/flamegpu_host_api_macros.h"  // Todo replace with std/cub style fns (see AgentFunction.h)
#include "flamegpu/pop/MemoryVector.h"

class EnvironmentDescription;
class AgentDescription;
class MessageDescription;
class AgentFunctionDescription;
class LayerDescription;

struct AgentData;
struct MessageData;
struct AgentFunctionData;
struct LayerData;

struct ModelData : std::enable_shared_from_this<ModelData>{
    static const char *DEFAULT_STATE;  // "default"
    friend class ModelDescription;
    typedef unsigned int size_type;
    struct Variable {
        /**
         * Cannot explicitly specify template args of constructor, so we take redundant arg for implicit template
         */
        template<typename T>
        Variable(size_type _elements, T)
            : type(typeid(T)), type_size(sizeof(T)), elements(_elements), memory_vector(new MemoryVector<T>()) { }
        const std::type_index type;
        const size_t type_size;
        const unsigned int elements;
        /**
         * Holds the type so we can dynamically create them with clone()
         */
        const std::unique_ptr<GenericMemoryVector> memory_vector;

        Variable(const Variable &other)
            :type(other.type), type_size(other.type_size), elements(other.elements), memory_vector(other.memory_vector->clone()) { }
    };
    typedef std::unordered_map<std::string, Variable> VariableMap;
    typedef std::unordered_map<std::string, std::shared_ptr<AgentData>> AgentMap;
    typedef std::unordered_map<std::string, std::shared_ptr<MessageData>> MessageMap;
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
    std::unique_ptr<EnvironmentDescription> environment;  // TODO: Move this to same Data:Description format
    std::string name;

    std::shared_ptr<ModelData> clone() const;

 protected:
     /**
      * This should only be called via clone();
      */
    explicit ModelData(const ModelData &other);
    explicit ModelData(const std::string &model_name);
};
struct AgentData : std::enable_shared_from_this<AgentData> {
    friend class ModelDescription;
    friend struct ModelData;
    friend class AgentPopulation;
    typedef std::unordered_map<std::string, std::shared_ptr<AgentFunctionData>> FunctionMap;

    FunctionMap functions;
    ModelData::VariableMap variables;
    std::set<std::string> states;
    std::string initial_state;
    unsigned int agent_outputs;  // Number of functions that have agent output of this agent type
    std::shared_ptr<AgentDescription> description;
    std::string name;
    bool keepDefaultState;

    bool isOutputOnDevice() const;  // Convenience wrapper for agent_outputs

    bool operator==(const AgentData& rhs) const;
    bool operator!=(const AgentData& rhs) const;
    AgentData(const AgentData &other) = delete;
 protected:
    AgentData(ModelData *const model, const AgentData &other);
    AgentData(ModelData *const model, const std::string &agent_name);
};
struct MessageData {
    friend class ModelDescription;
    friend class ModelData;

    ModelData::VariableMap variables;
    std::unique_ptr<MessageDescription> description;
    std::string name;

    MessageData(const MessageData &other) = delete;
 protected:
     MessageData(ModelData *const, const MessageData &other);
    MessageData(ModelData *const, const std::string &message_name);
};
struct AgentFunctionData {
    friend class AgentDescription;
    friend ModelData;

    AgentFunctionWrapper *func;

    std::string initial_state;
    std::string end_state;

    std::weak_ptr<MessageData> message_input;
    std::weak_ptr<MessageData> message_output;
    bool message_output_optional;

    std::weak_ptr<AgentData> agent_output;

    bool has_agent_death = false;

    std::weak_ptr<AgentData> parent;
    std::unique_ptr<AgentFunctionDescription> description;
    std::string name;

    bool operator==(const AgentFunctionData& rhs) const;
    bool operator!=(const AgentFunctionData& rhs) const;

    AgentFunctionData(const AgentFunctionData &other) = delete;

 protected:
    AgentFunctionData(ModelData *const model, std::shared_ptr<AgentData> _parent, const AgentFunctionData &other);
    AgentFunctionData(std::shared_ptr<AgentData> _parent, const std::string &function_name, AgentFunctionWrapper *agent_function);
};
struct LayerData {
    friend class ModelDescription;
    friend struct ModelData;

    std::set<std::shared_ptr<AgentFunctionData>> agent_functions;
    std::set<FLAMEGPU_HOST_FUNCTION_POINTER> host_functions;

    std::unique_ptr<LayerDescription> description;
    std::string name;

    ModelData::size_type index;

    LayerData(const LayerData &other) = delete;
 protected:
     LayerData(ModelData *const model, const LayerData &other);
    LayerData(ModelData *const model, const std::string &name, const ModelData::size_type &index);
};



#endif  // INCLUDE_FLAMEGPU_MODEL_MODELDATA_H_
