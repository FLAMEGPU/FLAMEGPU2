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

/**
 * This class is CUDAAgentModel's internal handler for message functionality
 */
class CUDAMessage {
 public:
     /**
      * Constructs a CUDAMessage object
      * Allocates enough memory for each variable within the provided MessageData
      * @param description The message to represent
      */
    explicit CUDAMessage(const MessageData& description);
    /**
     * Destructor, releases CUDA memory
     */
    virtual ~CUDAMessage(void);
    /**
     * Return an immutable reference to the message description represented by the CUDAMessage instance
     */
    const MessageData& getMessageDescription() const;
    /**
     * @return The currently allocated length of the message array (in the number of messages)
     */
    unsigned int getMaximumListSize() const;
    /**
     * @return The current number of messages
     */
    unsigned int getMessageCount() const;
    /**
     * Updates message_count to equal newSize, internally reallocates buffer space if more space is required
     * @param newSize The number of messages that the buffer should be capable of storing
     */
    void resize(unsigned int newSize);
    /**
     * Uses the cuRVE runtime to map the variables used by the agent function to the cuRVE library so that can be accessed by name within a n agent function
     * The read runtime variables are to be used when reading messages
     * @param func The agent function, this is used for the cuRVE hash mapping
     */
    void mapReadRuntimeVariables(const AgentFunctionData& func) const;
    /**
     * Uses the cuRVE runtime to map the variables used by the agent function to the cuRVE library so that can be accessed by name within a n agent function
     * The write runtime variables are to be used when creating messages, as they are output to swap space
     * @param func The agent function, this is used for the cuRVE hash mapping
     * @note swap() or scatter() should be called after the agent function has written messages
     */
    void mapWriteRuntimeVariables(const AgentFunctionData& func) const;
    /**
     * Uses the cuRVE runtime to unmap the variables used by the agent function to the cuRVE
     * library so that they are unavailable to be accessed by name within an agent function.
     * @param func The agent function, this is used for the cuRVE hash mapping
     */
    void unmapRuntimeVariables(const AgentFunctionData& func) const;
    /**
     * Swaps the two internal maps within message_list
     */
    virtual void swap();

 protected:
    /** 
     * Zero all message variable data.
     */
    void zeroAllMessageData();

 private:
     /**
      * Holds the definition of the message type represented by this CUDAMessage
      */
    const MessageData& message_description;
    /**
     * Holds/Manages the cuda memory for each of the message variables
     */
    std::unique_ptr<CUDAMessageList> message_list;  // CUDAMessageMap message_list;
    /**
     * The number of messages currently in the message list
     * This is set to the number of messages to be written prior to message output
     * and may be reduced downwards after (e.g. optional messages)
     */
    unsigned int message_count;
    /**
     * The current number of messages that can be represented by the allocated space
     */
    unsigned int max_list_size;
    /**
     * The number of messages CUB temp has been allocated for
     */
    unsigned int cub_temp_size_max_list_size;
    /**
     * The size of current cub temp allocation
     */
    size_t cub_temp_size;
    /**
     * Pointer to cub memory
     */
    void * d_cub_temp;
    /**
     * Reference to curve instance used internally
     */
    Curve &curve;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAMESSAGE_H_
