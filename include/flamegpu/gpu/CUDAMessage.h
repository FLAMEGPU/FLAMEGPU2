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

#include <memory>

// include sub classes
#include <flamegpu/gpu/CUDAMessageList.h>

// forward declare classes from other modules

class AgentPopulation;
class AgentFunctionDescription;
class MessageDescription;


class CUDAMessage
{
 public:
    CUDAMessage(const MessageDescription& description);
    virtual ~CUDAMessage(void);

    const MessageDescription& getMessageDescription() const;
    unsigned int getMaximumListSize() const;

    /**
     * @brief Uses the cuRVE runtime to map the variables used by the agent function to the cuRVE library so that can be accessed by name within a n agent function
     *
     * @param    func    The function.
     */
    void mapRuntimeVariables(const AgentFunctionDescription& func) const;

    /**
     * @brief    Uses the cuRVE runtime to unmap the variables used by the agent function to the cuRVE
     *             library so that they are unavailable to be accessed by name within an agent function.
     *
     * @param    func    The function.
     */
    void unmapRuntimeVariables(const AgentFunctionDescription& func) const;

protected:
    /**
     * @brief Allocates the messagelist memory, called by constructor
     */
    void setInitialMessageList();

    /** @brief    Zero all message variable data. */
    void zeroAllMessageData();


 private:
    const MessageDescription& message_description;

    std::unique_ptr<CUDAMessageList> message_list; // CUDAMessageMap message_list;

    unsigned int max_list_size;

};

#endif
