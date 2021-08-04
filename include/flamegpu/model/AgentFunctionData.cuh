
#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDATA_CUH_
#define INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDATA_CUH_

#include <memory>
#include <string>

#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/runtime/AgentFunction.cuh"
#include "flamegpu/runtime/AgentFunctionCondition.cuh"

namespace flamegpu {

class AgentFunctionDescription;
class AgentDescription;

/**
 * This is the internal data store for AgentFunctionDescription
 * Users should only access that data stored within via an instance of AgentFunctionDescription
 */
struct AgentFunctionData {
    friend class AgentDescription;
    /**
     * Cloning an AgentData requires access to AgentFunctionData for building the cloned function map
     */
    friend std::shared_ptr<const AgentData> AgentData::clone() const;
    friend struct ModelData;

    /**
     * The cuda kernel entry point for executing the agent function
     * @see void agent_function_wrapper(detail::curve::Curve::NamespaceHash, detail::curve::Curve::NamespaceHash, detail::curve::Curve::NamespaceHash, detail::curve::Curve::NamespaceHash, const int, const void *, const unsigned int, const unsigned int)
     */
    AgentFunctionWrapper *func;
    /**
     * The string representing the RTC defined agent function
     */
    std::string rtc_source;
    /**
     * The string representing the RTC defined agent function name
     */
    std::string rtc_func_name;
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
    std::weak_ptr<MessageBruteForce::Data> message_input;
    /**
     * If set, this type of message is output by the function
     */
    std::weak_ptr<MessageBruteForce::Data> message_output;
    /**
     * If set, message outputs from this function are optional
     */
    bool message_output_optional;
    /**
     * If set, this is the agent type which is output by the function
     */
    std::weak_ptr<AgentData> agent_output;
    /**
     * If set, this is the agent type which is output by the function
     */
    std::string agent_output_state;
    /**
     * This must be marked to true if the agent function can return DEAD
     * Enabling this tells FLAMEGPU to sort agents to remove those which have died from the population
     */
    bool has_agent_death = false;
    /**
     * The cuda kernel entry point for executing the agent function condition
     * @see void agent_function_condition_wrapper(detail::curve::Curve::NamespaceHash, detail::curve::Curve::NamespaceHash, const int, const unsigned int, const unsigned int)
     */
    AgentFunctionConditionWrapper *condition;
    /**
     * The string representing the RTC defined agent function condition
     */
    std::string rtc_condition_source;
    /**
     * The string representing the RTC defined agent function condition
     */
    std::string rtc_func_condition_name;
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
    bool operator==(const AgentFunctionData &rhs) const;
    /**
     * Equality operator, checks whether AgentFunctionData hierarchies are functionally different
     * @returns True when agent functions are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const AgentFunctionData &rhs) const;
    /**
     * Default copy constructor, not implemented
     */
    AgentFunctionData(const AgentFunctionData &other) = delete;
    /**
     * Input messaging type (as string) specified in FLAMEGPU_AGENT_FUNCTION. Used for type checking in model specification.
     */
    std::string message_in_type;
    /**
     * Output messaging type (as string) specified in FLAMEGPU_AGENT_FUNCTION. Used for type checking in model specification.
     */
    std::string message_out_type;

 protected:
    /**
     * Copy constructor
     * This is unsafe, should only be used internally, use clone() instead
     * @param model Model hierarchy root, used for performing lookup copies (e.g. messages)
     * @param _parent Parent agent description
     * @param other Agent function description being copied
     */
    AgentFunctionData(const std::shared_ptr<const ModelData> &model, std::shared_ptr<AgentData> _parent, const AgentFunctionData &other);
    /**
     * Normal constructor, only to be called by AgentDescription
     * @param _parent Parent agent description
     * @param function_name User defined name of the agent function
     * @param agent_function Pointer to compile time agent function
     * @param in_type String form of the input message type
     * @param out_type String form of the output message type
     */
    AgentFunctionData(std::shared_ptr<AgentData> _parent, const std::string &function_name, AgentFunctionWrapper *agent_function, const std::string &in_type, const std::string &out_type);
    /**
     * Normal constructor for RTC function, only to be called by AgentDescription
     * @param _parent Parent agent description
     * @param function_name User defined name of the agent function
     * @param rtc_function_src Pointer to runtime agent function
     * @param in_type String form of the input message type
     * @param out_type String form of the output message type
     * @param code_func_name Name of the RTC agent function
     */
    AgentFunctionData(std::shared_ptr<AgentData> _parent, const std::string& function_name, const std::string &rtc_function_src, const std::string &in_type, const std::string& out_type, const std::string& code_func_name);
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDATA_CUH_
