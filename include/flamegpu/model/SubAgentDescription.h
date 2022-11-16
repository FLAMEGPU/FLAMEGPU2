#ifndef INCLUDE_FLAMEGPU_MODEL_SUBAGENTDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_SUBAGENTDESCRIPTION_H_

#include <memory>
#include <string>


namespace flamegpu {

struct ModelData;
struct SubModelData;
struct SubAgentData;


class CSubAgentDescription {
    /**
     * Data store class for this description, constructs instances of this class
     */
    friend struct SubAgentData;

 public:
    /**
     * Constructor, creates an interface to the SubAgentData
     * @param data Data store of this subagents's data
     */
    explicit CSubAgentDescription(std::shared_ptr<SubAgentData> data);
    explicit CSubAgentDescription(std::shared_ptr<const SubAgentData> data);
    /**
     * Copy constructor
     * Creates a new interface to the same SubAgentData/ModelData
     */
    CSubAgentDescription(const CSubAgentDescription& other_agent) = default;
    CSubAgentDescription(CSubAgentDescription&& other_agent) = default;
    /**
     * Assignment operator
     * Assigns this interface to the same SubAgentData/ModelData
     */
    CSubAgentDescription& operator=(const CSubAgentDescription& other_agent) = default;
    CSubAgentDescription& operator=(CSubAgentDescription&& other_agent) = default;
    /**
     * Equality operator, checks whether SubAgentDescription hierarchies are functionally the same
     * @param rhs right hand side
     * @returns True when subagents are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const CSubAgentDescription& rhs) const;
    /**
     * Equality operator, checks whether SubAgentDescription hierarchies are functionally different
     * @param rhs right hand side
     * @returns True when subagents are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const CSubAgentDescription& rhs) const;

    /**
     * Returns the master agent state which has been mapped to the name subagent state
     * @param sub_state_name Name of the state in the sub agent to check
     * @return The name of the state within the master agent which is mapped
     * @throws exception::InvalidAgentState If the sub agent state does not exist or has not been mapped yet
     */
    std::string getStateMapping(const std::string& sub_state_name) const;
    /**
     * Returns the master agent variable which has been mapped to the name subagent variable
     * @param sub_variable_name Name of the variable in the sub agent to check
     * @return The name of the variable within the master agent which is mapped
     * @throws exception::InvalidAgentVar If the sub agent variable does not exist or has not been mapped yet
     */
    std::string getVariableMapping(const std::string& sub_variable_name) const;

 protected:
    /**
     * The class which stores all of the agent's data.
     */
    std::shared_ptr<SubAgentData> subagent;
};

/**
 * This class provides an interface to a mapping between a parent and submodel agent
 */
class SubAgentDescription : public CSubAgentDescription {
 public:
    /**
     * Constructor, creates an interface to the SubAgentData
     * @param data Data store of this agent's data
     */
    explicit SubAgentDescription(std::shared_ptr<SubAgentData> data);
    /**
     * Copy constructor
     * Creates a new interface to the same SubAgentData/ModelData
     */
    SubAgentDescription(const SubAgentDescription& other_agent) = default;
    SubAgentDescription(SubAgentDescription&& other_agent) = default;
    /**
     * Assignment operator
     * Assigns this interface to the same SubAgentData/ModelData
     */
    SubAgentDescription& operator=(const SubAgentDescription& other_agent) = default;
    SubAgentDescription& operator=(SubAgentDescription&& other_agent) = default;

    /**
     * Links the named states between the master and sub agent
     * @param sub_state_name Name of the state in the sub models agent
     * @param master_state_name Name of the state in the master models agent
     * @throws exception::InvalidParent If the sub agent or master agent weak_ptrs have expired (this should never happen)
     * @throws exception::InvalidAgentState If the named state does not exist within the bound sub or master agent
     */
    void mapState(const std::string &sub_state_name, const std::string &master_state_name);
    /**
     * Links the named variables between the master and sub agent
     * These variables must have the same type and number of elements (1 unless they're an array variable)
     * @param sub_variable_name Name of the variable in the sub models agent
     * @param master_variable_name Name of the variable in the master models agent
     * @throws exception::InvalidParent If the sub agent or master agent weak_ptrs have expired (this should never happen)
     * @throws exception::InvalidAgentVar If the named variable does not exist within the bound sub or master agent
     * @throws exception::InvalidAgentVar If there is a mismatch between the variables types or number of elements
     */
    void mapVariable(const std::string &sub_variable_name, const std::string &master_variable_name);
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_SUBAGENTDESCRIPTION_H_
