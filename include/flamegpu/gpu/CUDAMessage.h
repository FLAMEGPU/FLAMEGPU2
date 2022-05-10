#ifndef INCLUDE_FLAMEGPU_GPU_CUDAMESSAGE_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAMESSAGE_H_

#include <memory>
#include <utility>
#include <string>

// include sub classes
#include "flamegpu/gpu/CUDAMessageList.h"
#include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceHost.h"

// forward declare classes from other modules

namespace flamegpu {

class CUDAScatter;
class CUDASimulation;
struct AgentFunctionData;
struct MessageData;
namespace detail {
namespace curve {
class HostCurve;
class Curve;
}  // namespace curve
}  // namespace detail
class MessageSpecialisationHandler;
class CUDAAgent;
/**
 * This class is CUDASimulation's internal handler for message functionality
 */
class CUDAMessage {
 public:
     /**
      * Constructs a CUDAMessage object
      * Allocates enough memory for each variable within the provided MessageData
      * @param description The message to represent
      * @param cudaSimulation The simulation which owns the CUDAMessage
      */
    explicit CUDAMessage(const MessageBruteForce::Data& description, const CUDASimulation& cudaSimulation);
    /**
     * Destructor, releases CUDA memory
     */
    virtual ~CUDAMessage(void);
    /**
     * Return an immutable reference to the message description represented by the CUDAMessage instance
     */
    const MessageBruteForce::Data& getMessageDescription() const;
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
     * Initialise the CUDAMessagelist
     * This allocates and initialises any CUDA data structures for reading the messagelist, and sets them asthough the messagelist were empty.
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId Index of stream specific structures used
     * @param stream The CUDAStream to use for CUDA operations
     */
    void init(CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Updates message_count to equal newSize, internally reallocates buffer space if more space is required
     * @param newSize The number of messages that the buffer should be capable of storing
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId Index of stream specific structures used
     * @param keepLen Number of existing messages worth of data to retain through the resize
     */
    void resize(unsigned int newSize, CUDAScatter &scatter, cudaStream_t stream, unsigned int streamId, unsigned int keepLen = 0);
    /**
     * Uses the cuRVE runtime to map the variables used by the agent function to the cuRVE library so that can be accessed by name within a n agent function
     * The read runtime variables are to be used when reading messages
     * @param func The agent function, this is used for the cuRVE hash mapping
     * @param cuda_agent Agent which owns the agent function (condition) being mapped, if RTC function this holds the RTC header
     */
    void mapReadRuntimeVariables(const AgentFunctionData& func, const CUDAAgent& cuda_agent) const;
    /**
     * Uses the cuRVE runtime to map the variables used by the agent function to the cuRVE library so that can be accessed by name within a n agent function
     * The write runtime variables are to be used when creating messages, as they are output to swap space
     * @param func The agent function, this is used for the cuRVE hash mapping
     * @param cuda_agent Agent which owns the agent function (condition) being mapped, if RTC function this holds the RTC header
     * @param writeLen The number of messages to be output, as the length isn't updated till after output
     * @param stream The CUDAStream to use for CUDA operations
     * @note swap() or scatter() should be called after the agent function has written messages
     */
    void mapWriteRuntimeVariables(const AgentFunctionData& func, const CUDAAgent& cuda_agent, const unsigned int &writeLen, cudaStream_t stream) const;
    void *getReadPtr(const std::string &var_name);
    const CUDAMessageMap &getReadList() { return message_list->getReadList(); }
    const CUDAMessageMap &getWriteList() { return message_list->getWriteList(); }
    /**
     * Swaps the two internal maps within message_list
     * @param isOptional If optional newMessageCount will be reduced based on scan_flag[streamId]
     * @param newMessageCount The number of output messages (including optional messages which were not output)
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId Index of stream specific structures used
     * @throw exception::InvalidCudaMessage If this is called before the internal buffers have been allocated
     */
    void swap(bool isOptional, unsigned int newMessageCount, CUDAScatter &scatter, cudaStream_t stream, unsigned int streamId);
    /**
     * Basic list swap with no additional actions
     */
    void swap();
    bool getTruncateMessageListFlag() const { return truncate_messagelist_flag; }
    void setTruncateMessageListFlag() { truncate_messagelist_flag = true; }
    void clearTruncateMessageListFlag() { truncate_messagelist_flag = false; }
    bool getPBMConstructionRequiredFlag() const  { return pbm_construction_required; }
    void setPBMConstructionRequiredFlag() { pbm_construction_required = true; }
    void clearPBMConstructionRequiredFlag() { pbm_construction_required = false; }
    /**
     * Builds index, required to read messages (some messaging types won't require an implementation)
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void buildIndex(CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream);
    const void *getMetaDataDevicePtr() const;

 protected:
    /** 
     * Zero all message variable data.
     */
    void zeroAllMessageData(cudaStream_t stream);

 private:
     /**
      * Holds the definition of the message type represented by this CUDAMessage
      */
    const MessageBruteForce::Data& message_description;
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
    std::unique_ptr<MessageSpecialisationHandler> specialisation_handler;

    /**
     * A reference to the cuda model which this object belongs to
     */
    const CUDASimulation& cudaSimulation;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAMESSAGE_H_
