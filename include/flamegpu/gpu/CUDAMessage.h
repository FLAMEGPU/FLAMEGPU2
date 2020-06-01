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
#include <utility>
#include <string>

// include sub classes
#include "flamegpu/gpu/CUDAMessageList.h"
#include "flamegpu/runtime/messaging/BruteForce/BruteForceHost.h"

// forward declare classes from other modules

struct AgentFunctionData;
struct MessageData;
class AgentPopulation;
class Curve;
class MsgSpecialisationHandler;
class CUDAAgent;
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
    explicit CUDAMessage(const MsgBruteForce::Data& description, const CUDAAgentModel& cuda_model);
    /**
     * Destructor, releases CUDA memory
     */
    virtual ~CUDAMessage(void);
    /**
     * Return an immutable reference to the message description represented by the CUDAMessage instance
     */
    const MsgBruteForce::Data& getMessageDescription() const;
    /**
     * @return The currently allocated length of the message array (in the number of messages)
     */
    unsigned int getMaximumListSize() const;
    /**
     * @return The current number of messages
     */
    unsigned int getMessageCount() const;
    /**
     * Manually update the message count
     * @note This should be used cautiously
     * @note Required by array message types
     */
    void setMessageCount(const unsigned int &_message_count);
    /**
     * Updates message_count to equal newSize, internally reallocates buffer space if more space is required
     * @param newSize The number of messages that the buffer should be capable of storing
     */
    void resize(unsigned int newSize, const unsigned int &streamId);
    /**
     * Uses the cuRVE runtime to map the variables used by the agent function to the cuRVE library so that can be accessed by name within a n agent function
     * The read runtime variables are to be used when reading messages
     * @param func The agent function, this is used for the cuRVE hash mapping
     */
    void mapReadRuntimeVariables(const AgentFunctionData& func, const CUDAAgent& cuda_agent) const;
    /**
     * Uses the cuRVE runtime to map the variables used by the agent function to the cuRVE library so that can be accessed by name within a n agent function
     * The write runtime variables are to be used when creating messages, as they are output to swap space
     * @param func The agent function, this is used for the cuRVE hash mapping
     * @param writeLen The number of messages to be output, as the length isn't updated till after ouput
     * @note swap() or scatter() should be called after the agent function has written messages
     */
    void mapWriteRuntimeVariables(const AgentFunctionData& func, const CUDAAgent& cuda_agent, const unsigned int &writeLen) const;
    /**
     * Uses the cuRVE runtime to unmap the variables used by the agent function to the cuRVE
     * library so that they are unavailable to be accessed by name within an agent function.
     * @param func The agent function, this is used for the cuRVE hash mapping
     */
    void unmapRuntimeVariables(const AgentFunctionData& func) const;
    void *getReadPtr(const std::string &var_name);
    const CUDAMsgMap &getReadList() { return message_list->getReadList(); }
    const CUDAMsgMap &getWriteList() { return message_list->getWriteList(); }
    /**
     * Swaps the two internal maps within message_list
     * @param isOptional If optional newMsgCount will be reduced based on scan_flag[streamId]
     * @param newMsgCount The number of output messages (including optional messages which were not output)
     * @param streamId Index of stream specific structures used
     */
    virtual void swap(bool isOptional, const unsigned int &newMsgCount, const unsigned int &streamId);
    /**
     * Basic list swap with no additional actions
     */
    virtual void swap();
    bool getTruncateMessageListFlag() const { return truncate_messagelist_flag; }
    void setTruncateMessageListFlag() { truncate_messagelist_flag = true; }
    void clearTruncateMessageListFlag() { truncate_messagelist_flag = false; }
    bool getPBMConstructionRequiredFlag() const  { return pbm_construction_required; }
    void setPBMConstructionRequiredFlag() { pbm_construction_required = true; }
    void clearPBMConstructionRequiredFlag() { pbm_construction_required = false; }
    void buildIndex();
    const void *getMetaDataDevicePtr() const;

 protected:
    /** 
     * Zero all message variable data.
     */
    void zeroAllMessageData();

 private:
     /**
      * Holds the definition of the message type represented by this CUDAMessage
      */
    const MsgBruteForce::Data& message_description;
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
     * When this flag is set to True before message output, 
     * message output truncates the messagelist rather than appending
     * 
     * Set to True at start of each step
     * Set to False after first message output
     * @note Flag is currently updated in correct places, however I don't think it's used by message output
     */
    bool truncate_messagelist_flag;
    /**
     * When this flag is set to True before message input,
     * PBM is constructed
     * This behaviour is likely surplus for messaging types,
     * but the flag can be maintained
     *
     * Set to True each time the message list is updated
     * Set to False before messages are read
     */
    bool pbm_construction_required;
    std::unique_ptr<MsgSpecialisationHandler> specialisation_handler;

    /**
     * A reference to the cuda model which this object belongs to
     */
    const CUDAAgentModel& cuda_model;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAMESSAGE_H_
