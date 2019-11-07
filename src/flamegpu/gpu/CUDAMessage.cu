/**
* @file CUDAMessage.cpp
* @authors
* @date
* @brief
*
* @see
* @warning
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/CUDAMessageList.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"

#include "flamegpu/model/MessageDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/pop/AgentPopulation.h"

/**
* CUDAMessage class
* @brief allocates the hash table/list for message variables and copy the list to device
*/
CUDAMessage::CUDAMessage(const MessageDescription& description) : message_description(description), max_list_size(0) {
    setInitialMessageList();
}

/**
 * A destructor.
 * @brief Destroys the CUDAMessage object
 */
CUDAMessage::~CUDAMessage(void) {
}

/**
* @brief Returns message description
* @param none
* @return MessageDescription object
*/
const MessageDescription& CUDAMessage::getMessageDescription() const {
    return message_description;
}

/**
* @brief Sets initial message data to zero by allocating memory for message lists
* @param empty
* @return none
*/
void CUDAMessage::setInitialMessageList() {  // used to be const AgentPopulation& population
    // check that the message list has not already been set
    if (message_list)
        throw InvalidMessageData("Error: Initial message list already set");

    /*
    unsigned int size = message_description.getMaximumMessageListCapacity();
    max_list_size = population.getMaximumStateListCapacity();

    // set the maximum population state size
    if (max_list_size > size)
        throw InvalidMessageSize("Error: Invalid Message List size");
    */

    max_list_size = message_description.getMaximumMessageListCapacity();  // maxmimum message list, not the population

    // allocate memory for each message list
    message_list = std::unique_ptr<CUDAMessageList>(new CUDAMessageList(*this));
    // message_list = std::make_unique<CUDAMessageList>(*this);  // you may replace *this with "new CUDAMessageList(*this)" , compile this with -std=c++14. Not possible with CUDA 8 under linux using cmake.

    /**set the message list to zero*/
    zeroAllMessageData();
}

/**
* @brief Returns the maximum list size
* @param none
* @return maximum size list that is equal to the maximum list size
* @note may want to change this to maximum population size
*/
unsigned int CUDAMessage::getMaximumListSize() const {
    return max_list_size;
}

/**
* @brief Sets all message variable data to zero
* @param none
* @return none
*/
void CUDAMessage::zeroAllMessageData() {
    message_list->zeroMessageData();
}

/**
@bug message_name is input or output, run some tests to see which one is correct
*/
void CUDAMessage::mapRuntimeVariables(const AgentFunctionDescription& func) const {
    // check that the message list has been allocated
    if (!message_list)
        throw InvalidMessageData("Error: Initial message has not been allocated");

    const std::string message_name = message_description.getName();

    // loop through the message variables to map each variable name using cuRVE
    for (VariableMapPair mmp : message_description.getVariableMap()) {
        // get a device pointer for the message variable name
        void* d_ptr = message_list->getMessageListVariablePointer(mmp.first);

        // map using curve
        CurveVariableHash var_hash = curveVariableRuntimeHash(mmp.first.c_str());
        CurveVariableHash message_hash = curveVariableRuntimeHash(message_name.c_str());
        CurveVariableHash agent_hash = curveVariableRuntimeHash(func.getParent().getName().c_str());
        CurveVariableHash func_hash = curveVariableRuntimeHash(func.getName().c_str());

        // get the message variable size
        size_t size;
        size = message_description.getMessageVariableSize(mmp.first.c_str());

       // maximum population size
        unsigned int length = this->getMaximumListSize();  // check to see if it is equal to pop

        curveRegisterVariableByHash(var_hash + agent_hash + func_hash + message_hash, d_ptr, size, length);
    }
}

void CUDAMessage::unmapRuntimeVariables(const AgentFunctionDescription& func) const {
    const std::string message_name = message_description.getName();
    // loop through the message variables to map each variable name using cuRVE
    for (VariableMapPair mmp : message_description.getVariableMap()) {
        // get a device pointer for the message variable name
        // void* d_ptr = message_list->getMessageListVariablePointer(mmp.first);

        // unmap using curve
        CurveVariableHash var_hash = curveVariableRuntimeHash(mmp.first.c_str());
        CurveVariableHash message_hash = curveVariableRuntimeHash(message_name.c_str());
        CurveVariableHash agent_hash = curveVariableRuntimeHash(func.getParent().getName().c_str());
        CurveVariableHash func_hash = curveVariableRuntimeHash(func.getName().c_str());

        curveUnregisterVariableByHash(var_hash + agent_hash + func_hash + message_hash);
    }
}

