/**
* @file CUDAMessage.h
* @authors
* @date
* @brief
*
* @see
* @warning
*/

#ifndef INCLUDE_FLAMEGPU_GPU_CUDAMESSAGE_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAMESSAGE_H_

#include <memory>

// include sub classes
#include "flamegpu/gpu/CUDAMessageList.h"

// forward declare classes from other modules

struct AgentFunctionData;
struct MessageData;
class AgentPopulation;
class Curve;

class CUDAMessage {
 public:
    explicit CUDAMessage(const MessageData& description);
    virtual ~CUDAMessage(void);

    const MessageData& getMessageDescription() const;
    unsigned int getMaximumListSize() const;
    unsigned int getMessageCount() const;
    void resize(unsigned int newSize);
    /**
     * @brief Uses the cuRVE runtime to map the variables used by the agent function to the cuRVE library so that can be accessed by name within a n agent function
     *
     * @param    func    The function.
     */
    void mapRuntimeVariables(const AgentFunctionData& func) const;

    /**
     * @brief    Uses the cuRVE runtime to unmap the variables used by the agent function to the cuRVE
     *             library so that they are unavailable to be accessed by name within an agent function.
     *
     * @param    func    The function.
     */
    void unmapRuntimeVariables(const AgentFunctionData& func) const;

 protected:
    /**
     * @brief Allocates the messagelist memory, called by constructor
     */
    void setInitialMessageList();

    /** @brief    Zero all message variable data. */
    void zeroAllMessageData();

 private:
    const MessageData& message_description;

    std::unique_ptr<CUDAMessageList> message_list;  // CUDAMessageMap message_list;

    unsigned int message_count;
    unsigned int max_list_size;

    Curve &curve;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAMESSAGE_H_
