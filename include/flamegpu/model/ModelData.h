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

/**
 * This is the internal data store for ModelDescription
 * Users should only access that data stored within via an instance of ModelDescription
 */
struct ModelData : std::enable_shared_from_this<ModelData>{
    /**
     * Default state, all agents and agent functions begin in/with this state
     */
    static const char *DEFAULT_STATE;  // "default"
    /**
     * Description needs full access
     */
    friend class ModelDescription;
    /**
     * Common size type used in the definition of models
     */
    typedef unsigned int size_type;
    /**
     * Common variable definition type
     * Used internally by AgentData and MessageData
     */
    struct Variable {
        /**
         * Constructs a new variable
         * @param _elements The number of elements, this will be 1 unless the variable is an array
         * @tparam T The type of the variable, it's size and std::type_index are derived from this
         * @note Cannot explicitly specify template args of constructor, so we take redundant arg for implicit template
         */
        template<typename T>
        Variable(size_type _elements, T)
            : type(typeid(T)), type_size(sizeof(T)), elements(_elements), memory_vector(new MemoryVector<T>()) { }
        /**
         * Unique identifier of the variables type as returned by std::type_index(typeid())
         */
        const std::type_index type;
        /**
         * Size of the type in bytes as returned by sizeof() (e.g. float == 4 bytes)
         */
        const size_t type_size;
        /**
         * The number of elements, this will be 1 unless the variable is an array
         */
        const unsigned int elements;
        /**
         * Holds the variables memory vector type so we can dynamically create them with clone()
         */
        const std::unique_ptr<GenericMemoryVector> memory_vector;
        /**
         * Copy constructor
         */
        Variable(const Variable &other)
            :type(other.type), type_size(other.type_size), elements(other.elements), memory_vector(other.memory_vector->clone()) { }
    };
    /**
     * Map of name:variable definition
     * map<string, Variable>
     */
    typedef std::unordered_map<std::string, Variable> VariableMap;
    /**
     * Map of name:agent definition
     * map<string, AgentData>
     */
    typedef std::unordered_map<std::string, std::shared_ptr<AgentData>> AgentMap;
    /**
     * Map of name:message definition
     * map<string, MessageData>
     */
    typedef std::unordered_map<std::string, std::shared_ptr<MessageData>> MessageMap;
    /**
     * List of layer definitions
     * list<LayerData>
     */
    typedef std::list<std::shared_ptr<LayerData>> LayerList;
    /**
     * Set of Init function pointers
     * set<FLAMEGPU_INIT_FUNCTION_POINTER>
     */
    typedef std::set<FLAMEGPU_INIT_FUNCTION_POINTER> InitFunctionSet;
    /**
     * Set of Step function pointers
     * set<FLAMEGPU_STEP_FUNCTION_POINTER>
     */
    typedef std::set<FLAMEGPU_STEP_FUNCTION_POINTER> StepFunctionSet;
    /**
     * Set of Exit function pointers
     * set<FLAMEGPU_EXIT_FUNCTION_POINTER>
     */
    typedef std::set<FLAMEGPU_EXIT_FUNCTION_POINTER> ExitFunctionSet;
    /**
     * Set of Exit condition pointers
     * set<FLAMEGPU_EXIT_CONDITION_POINTER>
     */
    typedef std::set<FLAMEGPU_EXIT_CONDITION_POINTER> ExitConditionSet;

    /**
     * Holds all of the model's agent definitions
     */
    AgentMap agents;
    /**
     * Holds all of the model's message definitions
     */
    MessageMap messages;
    /**
     * Holds all of the model's layer definitions
     */
    LayerList layers;
    /**
     * Holds pointers to all of the init functions used by the model
     */
    InitFunctionSet initFunctions;
    /**
     * Holds pointers to all of the step functions used by the model
     */
    StepFunctionSet stepFunctions;
    /**
     * Holds pointers to all of the exit functions used by the model
     */
    ExitFunctionSet exitFunctions;
    /**
     * Holds pointers to all of the exit conditions used by the model
     */
    ExitConditionSet exitConditions;
    /**
     * Holds all of the model's environment property definitions
     */
    std::unique_ptr<EnvironmentDescription> environment;  // TODO: Move this to same Data:Description format
    /**
     * The name of the model
     * This must be unique among Simulation (e.g. CUDAAgentModel) instances
     */
    std::string name;
    /**
     * Creates a copy of the entire model definition hierarchy
     * This is called when a ModelDescription is passed to a Simulation (e.g. CUDAAgentModel)
     */
    std::shared_ptr<ModelData> clone() const;
    /**
     * Equality operator, checks whether ModelData hierarchies are functionally the same
     * @returns True when models are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const ModelData& rhs) const;
    /**
     * Equality operator, checks whether ModelData hierarchies are functionally different
     * @returns True when models are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const ModelData& rhs) const;

 protected:
     /**
      * Copy constructor
      * This should only be called via clone();
      */
    explicit ModelData(const ModelData &other);
    /**
     * Normal constructor
     * This should only be called by ModelDescription
     */
    explicit ModelData(const std::string &model_name);
};

/**
 * This is the internal data store for AgentDescription
 * Users should only access that data stored within via an instance of AgentDescription
 */
struct AgentData : std::enable_shared_from_this<AgentData> {
    friend class ModelDescription;
    friend struct ModelData;
    friend class AgentPopulation;
    /**
     * Map of name:agent function definition
     * map<string, AgentFunctionData>
     */
    typedef std::unordered_map<std::string, std::shared_ptr<AgentFunctionData>> FunctionMap;

    /**
     * Holds all of the agent's function definitions
     */
    FunctionMap functions;
    /**
     * Holds all of the agent's variable definitions
     */
    ModelData::VariableMap variables;
    /**
     * Holds all of the agent's possible states
     */
    std::set<std::string> states;
    /**
     * The initial state of newly created agents
     * @note must be found within member set states
     */
    std::string initial_state;
    /**
     * The number of functions that have agent output of this agent type
     * This value is modified by AgentFunctionDescription
     */
    unsigned int agent_outputs;
    /**
     * Description class which provides convenient accessors
     * This may be null if the instance has been cloned
     */
    std::shared_ptr<AgentDescription> description;
    /**
     * Name of the agent, used to refer to the agent in many functions
     */
    std::string name;
    /**
     * Internal value used to track whether the user has requested the default state as a state
     */
    bool keepDefaultState;
    /**
     * Returns true if this type of agent is created by any agent functions
     */
    bool isOutputOnDevice() const;
    /**
     * Equality operator, checks whether AgentData hierarchies are functionally the same
     * @returns True when agents are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const AgentData& rhs) const;
    /**
     * Equality operator, checks whether AgentData hierarchies are functionally different
     * @returns True when agents are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const AgentData& rhs) const;
    /**
     * Default copy constructor, not implemented
     */
    AgentData(const AgentData &other) = delete;
    /**
     * Returns a constant copy of this agent's hierarchy
     * Does not copy description, sets it to nullptr instead
     */
    std::shared_ptr<const AgentData> clone() const;

 protected:
    /**
     * Copy constructor
     * This is unsafe, should only be used internally, use clone() instead
     * This does not setup functions map 
     */
    AgentData(ModelData *const model, const AgentData &other);
    /**
     * Normal constructor, only to be called by ModelDescription
     */
    AgentData(ModelData *const model, const std::string &agent_name);
};

/**
 * This is the internal data store for MessageDescription
 * Users should only access that data stored within via an instance of MessageDescription
 */
struct MessageData {
    friend class ModelDescription;
    friend struct ModelData;

    /**
     * Holds all of the message's variable definitions
     */
    ModelData::VariableMap variables;
    /**
     * Description class which provides convenient accessors
     */
    std::unique_ptr<MessageDescription> description;
    /**
     * Name of the message, used to refer to the message in many functions
     */
    std::string name;
    /**
     * Equality operator, checks whether MessageData hierarchies are functionally the same
     * @returns True when messages are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const MessageData& rhs) const;
    /**
     * Equality operator, checks whether MessageData hierarchies are functionally different
     * @returns True when messages are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const MessageData& rhs) const;
    /**
     * Default copy constructor, not implemented
     */
    MessageData(const MessageData &other) = delete;

 protected:
    /**
     * Copy constructor
     * This is unsafe, should only be used internally, use clone() instead
     */
    MessageData(ModelData *const, const MessageData &other);
    /**
     * Normal constructor, only to be called by ModelDescription
     */
    MessageData(ModelData *const, const std::string &message_name);
};

/**
 * This is the internal data store for AgentFunctionDescription
 * Users should only access that data stored within via an instance of AgentFunctionDescription
 */
struct AgentFunctionData {
    friend class AgentDescription;
    friend std::shared_ptr<const AgentData> AgentData::clone() const;
    friend struct ModelData;

    /**
     * The cuda kernel entry point for executing the agent function
     * @see void agent_function_wrapper(Curve::NamespaceHash, Curve::NamespaceHash, Curve::NamespaceHash, Curve::NamespaceHash, const int, const unsigned int, const unsigned int)
     */
    AgentFunctionWrapper *func;
    /**
     * Agent's must be in this state to execute this function
     */
    std::string initial_state;
    /**
     * Agent's which execute this function leave in this state
     */
    std::string end_state;
    /**
     * If set, this type of message is input to the function
     */
    std::weak_ptr<MessageData> message_input;
    /**
     * If set, this type of message is output by the function
     */
    std::weak_ptr<MessageData> message_output;
    /**
     * If set, message outputs from this function are optional
     */
    bool message_output_optional;
    /**
     * If set, this is the agent type which is output by the function
     */
    std::weak_ptr<AgentData> agent_output;
    /**
     * This must be marked to true if the agent function can return DEAD
     * Enabling this tells FLAMEGPU to sort agents to remove those which have died from the population
     */
    bool has_agent_death = false;
    /**
     * The agent which this function is a member of
     */
    std::weak_ptr<AgentData> parent;
    /**
     * Description class which provides convenient accessors
     */
    std::unique_ptr<AgentFunctionDescription> description;
    /**
     * Name of the agent function, used to refer to the agent function in many functions
     */
    std::string name;
    /**
     * Equality operator, checks whether AgentFunctionData hierarchies are functionally the same
     * @returns True when agent functions are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const AgentFunctionData& rhs) const;
    /**
     * Equality operator, checks whether AgentFunctionData hierarchies are functionally different
     * @returns True when agent functions are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const AgentFunctionData& rhs) const;
    /**
     * Default copy constructor, not implemented
     */
    AgentFunctionData(const AgentFunctionData &other) = delete;

 protected:
    /**
     * Copy constructor
     * This is unsafe, should only be used internally, use clone() instead
     */
    AgentFunctionData(ModelData *const model, std::shared_ptr<AgentData> _parent, const AgentFunctionData &other);
    /**
     * Normal constructor, only to be called by AgentDescription
     */
    AgentFunctionData(std::shared_ptr<AgentData> _parent, const std::string &function_name, AgentFunctionWrapper *agent_function);
};

/**
 * This is the internal data store for LayerDescription
 * Users should only access that data stored within via an instance of LayerDescription
 */
struct LayerData {
    friend class ModelDescription;
    friend struct ModelData;

    /**
     * Set of Agent Functions
     * set<AgentFunctionData>
     */
    std::set<std::shared_ptr<AgentFunctionData>> agent_functions;
    /**
     * Set of host function pointers
     * set<FLAMEGPU_HOST_FUNCTION_POINTER>
     */
    std::set<FLAMEGPU_HOST_FUNCTION_POINTER> host_functions;
    /**
     * Description class which provides convenient accessors
     */
    std::unique_ptr<LayerDescription> description;
    /**
     * Name of the agent function, used to refer to the agent function in many functions
     */
    std::string name;
    /**
     * Index of the layer in the stack
     * (Eventually this will be replaced when we move to a more durable mode of layers, e.g. dependency analysis)
     */
    ModelData::size_type index;
    /**
     * Equality operator, checks whether LayerData hierarchies are functionally the same
     * @returns True when layers are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const LayerData& rhs) const;
    /**
     * Equality operator, checks whether LayerData hierarchies are functionally different
     * @returns True when layers are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const LayerData& rhs) const;
    /**
     * Default copy constructor, not implemented
     */
    LayerData(const LayerData &other) = delete;

 protected:
    /**
     * Copy constructor
     * This is unsafe, should only be used internally, use clone() instead
     */
    LayerData(ModelData *const model, const LayerData &other);
    /**
     * Normal constructor, only to be called by ModelDescription
     */
    LayerData(ModelData *const model, const std::string &name, const ModelData::size_type &index);
};

#endif  // INCLUDE_FLAMEGPU_MODEL_MODELDATA_H_
