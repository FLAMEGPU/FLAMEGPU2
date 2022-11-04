#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDESCRIPTION_H_

#include <string>
#include <memory>
#include <vector>

#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/runtime/AgentFunction.cuh"
#include "flamegpu/runtime/AgentFunctionCondition.cuh"
#include "flamegpu/model/DependencyNode.h"
#include "flamegpu/model/Variable.h"
#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/runtime/messaging/MessageBruteForce.h"
#include "flamegpu/runtime/detail/curve/curve_rtc.cuh"

#ifdef _MSC_VER
#pragma warning(push, 2)
#include "jitify/jitify.hpp"
#pragma warning(pop)
#else
#include "jitify/jitify.hpp"
#endif

namespace flamegpu {

struct ModelData;
struct AgentFunctionData;
class AgentFunctionDescription;

class CAgentFunctionDescription : public DependencyNode {
    friend struct AgentFunctionData;
    /**
     * Accesses internals to validate function description before adding to layer
     */
    friend void LayerDescription::addAgentFunction(const AgentFunctionDescription&);
    /**
     * Accesses internals to validate function belongs to the same model as the DependencyGraph
     */
    friend class DependencyGraph;
    friend class DependencyNode;

 public:
     /**
      * Constructor, creates an interface to the AgentFunctionData
      * @param data Data store of this function's data
      */
     explicit CAgentFunctionDescription(std::shared_ptr<AgentFunctionData> data);
     explicit CAgentFunctionDescription(std::shared_ptr<const AgentFunctionData> data);
     /**
      * Copy constructor
      * Creates a new interface to the same AgentFunctionData/ModelData
      */
     CAgentFunctionDescription(const CAgentFunctionDescription& other_function) = default;
     CAgentFunctionDescription(CAgentFunctionDescription&& other_function) = default;
     /**
      * Assignment operator
      * Assigns this interface to the same AgentFunctionData/ModelData
      */
     CAgentFunctionDescription& operator=(const CAgentFunctionDescription& other_function) = default;
     CAgentFunctionDescription& operator=(CAgentFunctionDescription&& other_function) = default;
     /**
      * Equality operator, checks whether AgentFunctionDescription hierarchies are functionally the same
      * @param rhs right hand side
      * @returns True when functions are the same
      * @note Instead compare pointers if you wish to check that they are the same instance
      */
     bool operator==(const CAgentFunctionDescription& rhs) const;
     /**
      * Equality operator, checks whether AgentFunctionDescription hierarchies are functionally different
      * @param rhs right hand side
      * @returns True when functions are not the same
      * @note Instead compare pointers if you wish to check that they are not the same instance
      */
     bool operator!=(const CAgentFunctionDescription& rhs) const;

     /**
      * @return The function's name
      */
     std::string getName() const;
     /**
      * @return The state which agents must be in to execute this agent function
      */
     std::string getInitialState() const;
     /**
      * @return The state which agents executing this function enter
      */
     std::string getEndState() const;
     /**
      * @return An immutable reference to the message input of this agent function
      * @see AgentFunctionDescription::MessageInput() for the mutable version
      * @throw exception::OutOfBoundsException If the message input has not been set
      */
     const MessageBruteForce::Description& getMessageInput() const;
     /**
      * @return An immutable reference to the message output of this agent function
      * @see AgentFunctionDescription::MessageOutput() for the mutable version
      * @throw exception::OutOfBoundsException If the message output has not been set
      */
     const MessageBruteForce::Description& getMessageOutput() const;
     /**
      * @return True if message output from this agent function is optional
      */
     bool getMessageOutputOptional() const;
     /**
      * @return A immutable interface to the agent output of this agent function
      * @throw exception::OutOfBoundsException If the agent output has not been set
      */
     CAgentDescription getAgentOutput() const;
     /**
      * @return The state output agents will be output in
      * @throw exception::OutOfBoundsException If the agent output has not been set
      */
     std::string getAgentOutputState() const;
     /**
      * @return True if this agent function can kill agents
      */
     bool getAllowAgentDeath() const;
     /**
      * @return True if setMessageInput() has been called successfully
      * @see AgentFunctionDescription::setMessageInput(const std::string &)
      * @see AgentFunctionDescription::setMessageInput(MessageDescription &)
      */
     bool hasMessageInput() const;
     /**
      * @return True if setMessageOutput() has been called successfully
      * @see AgentFunctionDescription::setMessageOutput(const std::string &)
      * @see AgentFunctionDescription::setMessageOutput(MessageDescription &)
      */
     bool hasMessageOutput() const;
     /**
      * @return True if setAgentOutput() has been called successfully
      * @see AgentFunctionDescription::setAgentOutput(const std::string &)
      * @see AgentFunctionDescription::setAgentOutput(AgentDescription &)
      */
     bool hasAgentOutput() const;
     /**
      * @return True if setFunctionCondition() has been called successfully
      * @see AgentFunctionDescription::setFunctionCondition(AgentFunctionCondition)
      */
     bool hasFunctionCondition() const;
     /**
      * @return The cuda kernel entry point for executing the agent function
      */
     AgentFunctionWrapper* getFunctionPtr() const;
     /**
      * @return The cuda kernel entry point for executing the agent function condition
      */
     AgentFunctionConditionWrapper* getConditionPtr() const;
     /**
      * @return True if the function is a runtime time specified function
      */
     bool isRTC() const;

 protected:
     /**
      * The class which stores all of the function's data.
      */
     std::shared_ptr<AgentFunctionData> function;
};


/**
 * Within the model hierarchy, this class represents an agent function for a FLAMEGPU model
 * This class is used to configure external elements of agent functions, such as inputs and outputs
 * @see AgentDescription::newFunction(const std::string&, AgentFunction) For creating instances of this class
 */
class AgentFunctionDescription : public CAgentFunctionDescription {
 public:
    /**
     * Constructors
     */
    explicit AgentFunctionDescription(std::shared_ptr<AgentFunctionData> data);
    /**
     * Copy constructor
     */
    AgentFunctionDescription(const AgentFunctionDescription& other_agent) = default;
    AgentFunctionDescription(AgentFunctionDescription&& other_agent) = default;
    /**
     * Assignment operator
     */
    AgentFunctionDescription& operator=(const AgentFunctionDescription &other_function) = default;
    AgentFunctionDescription& operator=(AgentFunctionDescription &&other_function) = default;
    /**
     * Equality operator, checks whether AgentFunctionDescription hierarchies are functionally the same
     * @returns True when agent functions are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const CAgentFunctionDescription& rhs) const;
    /**
     * Equality operator, checks whether AgentFunctionDescription hierarchies are functionally different
     * @returns True when agent functions are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const CAgentFunctionDescription& rhs) const;

    /**
     * Sets the initial state which agents must be in to execute this function
     * @param init_state Name of the desired state
     * @throws exception::InvalidStateName If the named state is not found within the agent
     * @see AgentFunctionDescription::setEndState(const std::string &)
     */
    void setInitialState(const std::string &init_state);
    /**
     * Sets the end state which agents enter after executing this function
     * @param end_state Name of the desired state
     * @throws exception::InvalidStateName If the named state is not found within the agent
     * @see AgentFunctionDescription::setInitialState(const std::string &)
     */
    void setEndState(const std::string &end_state);
    /**
     * Sets the message type that can be read during this agent function
     * This is optional, and only one type of message can be read per agent function
     * @param message_name Name of the message type to be input
     * @throws exception::InvalidMessageName If a message with the same name is not found within the model's hierarchy
     * @throws exception::InvalidMessageName If the same message is already bound to the message output of this agent function
     * @see AgentFunctionDescription::setMessageInput(MessageDescription &)
     */
    void setMessageInput(const std::string &message_name);
    /**
     * Sets the message type that can be read during this agent function
     * This is optional, and only one type of message can be read per agent function
     * @param message Type of message to be input
     * @throws exception::DifferentModel If the message is not from this model hierarchy
     * @throws exception::InvalidMessageName If a message with the same name is not found within the model's hierarchy
     * @throws exception::InvalidMessageName If the same message is already bound to the message output of this agent function
     * @see AgentFunctionDescription::setMessageInput(const std::string &)
     */
    void setMessageInput(MessageBruteForce::Description &message);
    /**
     * Sets the message type that can be output during this agent function
     * This is optional, and only one type of message can be output per agent function
     * @param message_name Name of the message type to be output
     * @throws exception::InvalidMessageName If a message with the same name is not found within the model's hierarchy
     * @throws exception::InvalidMessageName If the same message is already bound to the message input of this agent function
     * @see AgentFunctionDescription::setMessageOutput(MessageDescription &)
     * @see AgentFunctionDescription::setMessageOutputOptional(bool) To configure whether all agents must output messages
     */
    void setMessageOutput(const std::string &message_name);
    /**
     * Sets the message type that can be output during this agent function
     * This is optional, and only one type of message can be output per agent function
     * @param message Type of message to be output
     * @throws exception::DifferentModel If the message is not from this model hierarchy
     * @throws exception::InvalidMessageName If a message with the same name is not found within the model's hierarchy
     * @throws exception::InvalidMessageName If the same message is already bound to the message input of this agent function
     * @see AgentFunctionDescription::setMessageInput(const std::string &)
     * @see AgentFunctionDescription::setMessageOutputOptional(bool) To configure whether all agents must output messages
     */
    void setMessageOutput(MessageBruteForce::Description &message);
    /**
     * Configures whether message output from this agent function is optional
     * (e.g. whether all agents must output a message each time the function is called)
     * If the function has no message output, this can be ignored
     * @param output_is_optional True if not all agents executing this function will output messages
     * @note Defaults to false
     */
    void setMessageOutputOptional(bool output_is_optional);
    /**
     * Sets the agent type that can be output during this agent function
     * This is optional, and only one type of agent can be output per agent function
     * @param agent_name Name of the agent type to be output
     * @param state The stage agents should be created in
     * @throws exception::InvalidAgentName If an agent with the same name is not found within the model's hierarchy
     * @throws exception::InvalidStateName If the named agent does not contain the named state
     * @see AgentFunctionDescription::setAgentOutput(AgentDescription &)
     */
    void setAgentOutput(const std::string &agent_name, const std::string state = ModelData::DEFAULT_STATE);
    /**
     * Sets the agent type that can be output during this agent function
     * This is optional, and only one type of agent can be output per agent function
     * @param agent Type of agent to be output
     * @param state The stage agents should be created in
     * @throws exception::DifferentModel If the agent is not from this model hierarchy
     * @throws exception::InvalidAgentName If an agent with the same name is not found within the model's hierarchy
     * @throws exception::InvalidStateName If the named agent does not contain the named state
     * @see AgentFunctionDescription::setAgentOutput(AgentDescription &)
     */
    void setAgentOutput(AgentDescription &agent, const std::string state = ModelData::DEFAULT_STATE);
    /**
     * Configures whether agents can die during execution of this function
     * (e.g. by returning AGENT_STATUS::DEAD from the agent function)
     * @param has_death True if some agents executing this agent function may die
     * @see AgentFunctionDescription::AllowAgentDeath()
     * @see AgentFunctionDescription::getAllowAgentDeath()
     * @note Defaults to false
     */
    void setAllowAgentDeath(bool has_death);
    /**
     * Sets the function condition for the agent function
     * This is an FLAMEGPU_AGENT_FUNCTION_CONDITION which returns a boolean value (true or false)
     * Only agents which return true perform the attached FLAMEGPU_AGENT_FUNCTION
     * and transition from the initial to end state
     * 
     */
    template<typename AgentFunctionCondition>
    void setFunctionCondition(AgentFunctionCondition);
    /**
     * Sets the RTC function condition for the agent function
     * This is string containing a definition of an FLAMEGPU_AGENT_FUNCTION_CONDITION which returns a boolean value (true or false)
     * Only agents which return true perform the attached FLAMEGPU_AGENT_FUNCTION
     * and transition from the initial to end state.
     * The string will be compiled at runtime.
     * @param func_cond_src Source containing RTC agent function condition source
     * @see setRTCFunctionConditionFile()
     */
    void setRTCFunctionCondition(std::string func_cond_src);
    /**
     * Sets the RTC function condition for the agent function, with an agent function condition loaded from file
     * This is file containing a definition of an FLAMEGPU_AGENT_FUNCTION_CONDITION which returns a boolean value (true or false)
     * Only agents which return true perform the attached FLAMEGPU_AGENT_FUNCTION
     * and transition from the initial to end state.
     * The string will be compiled at runtime.
     * @param file_path Location on disk of file containing RTC agent function condition source
     * @see setRTCFunctionCondition()
     */
    void setRTCFunctionConditionFile(const std::string& file_path);

    /**
     * @return A mutable reference to the message input of this agent function
     * @see AgentFunctionDescription::getMessageInput() for the immutable version
     * @throw exception::OutOfBoundsException If the message input has not been set
     */
    MessageBruteForce::Description &MessageInput();
    /**
     * @return An mutable reference to the message output of this agent function
     * @see AgentFunctionDescription::getMessageOutput() for the immutable version
     * @throw exception::OutOfBoundsException If the message output has not been set
     */
    MessageBruteForce::Description &MessageOutput();
    /**
     * @return A mutable reference to the message output optional configuration flag
     * @see AgentFunctionDescription::getAgentOutputOptional()
     * @see AgentFunctionDescription::setAgentOutputOptional(bool)
     */
    bool &MessageOutputOptional();
    /**
     * @return A mutable reference to the allow agent death configuration flag
     * @see AgentFunctionDescription::getAllowAgentDeath()
     * @see AgentFunctionDescription::setAllowAgentDeath(bool)
     */
    bool &AllowAgentDeath();
    /**
     * @return An interface to the agent output of this agent function
     * @throw exception::OutOfBoundsException If the agent output has not been set
     */
    AgentDescription AgentOutput();
};


template<typename AgentFunction>
AgentFunctionDescription AgentDescription::newFunction(const std::string &function_name, AgentFunction) {
    if (agent->functions.find(function_name) == agent->functions.end()) {
        AgentFunctionWrapper *f = AgentFunction::fnPtr();
        std::string in_t = detail::curve::CurveRTCHost::demangle(AgentFunction::inType().name());
        std::string out_t = detail::curve::CurveRTCHost::demangle(AgentFunction::outType().name());
        if (in_t == "flamegpu::MessageSpatial3D" || in_t == "flamegpu::MessageSpatial2D" || out_t == "flamegpu::MessageSpatial3D" || out_t == "flamegpu::MessageSpatial2D") {
            if (agent->variables.find("_auto_sort_bin_index") == agent->variables.end()) {
                agent->variables.emplace("_auto_sort_bin_index", Variable(1, std::vector<unsigned int> {0}));
            }
        }
        auto rtn = std::shared_ptr<AgentFunctionData>(new AgentFunctionData(this->agent->shared_from_this(), function_name, f, in_t, out_t));
        agent->functions.emplace(function_name, rtn);
        return AgentFunctionDescription(rtn);
    }
    THROW exception::InvalidAgentFunc("Agent ('%s') already contains function '%s', "
        "in AgentDescription::newFunction().",
        agent->name.c_str(), function_name.c_str());
}

template<typename AgentFunctionCondition>
void AgentFunctionDescription::setFunctionCondition(AgentFunctionCondition) {
    function->condition = AgentFunctionCondition::fnPtr();
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDESCRIPTION_H_
