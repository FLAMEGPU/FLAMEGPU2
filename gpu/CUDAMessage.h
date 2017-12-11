/**
* @file CUDAMessage.h
* @authors
* @date
* @brief
*
* @see
* @warning
*/

#ifndef CUDAMESSAGE_H_
#define CUDAMESSAGE_H_


//forward declare classes from other modules

class AgentPopulation;
class AgentFunctionDescription;
class MessageDescription;
class CUDAMessageStateList;

class CUDAMessage
{
public:
    CUDAMessage(const MessageDescription& description);
    virtual ~CUDAMessage(void);

    const MessageDescription& getMessageDescription() const;
    unsigned int getMaximumListSize() const;

    /**
     * @brief Uses the cuRVE runtime to map the variables used by the agent function to the cuRVE library so that can be accessed by name within a n agent function
     */
    void mapRuntimeVariables(const AgentFunctionDescription& func) const;

    /**
     * @brief	Uses the cuRVE runtime to unmap the variables used by the agent function to the cuRVE
     * 			library so that they are unavailable to be accessed by name within an agent function.
     *
     * @param	func	The function.
     */

    void unmapRuntimeVariables(const AgentFunctionDescription& func) const;

protected:


private:
    const MessageDescription& message_description;

	std::unique_ptr<CUDAMessageStateList> list;

    unsigned int max_list_size;

};

#endif
