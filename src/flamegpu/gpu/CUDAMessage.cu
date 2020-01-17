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

#ifdef _MSC_VER
#pragma warning(push, 3)
#include <cub/cub.cuh>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#endif

/**
* CUDAMessage class
* @brief allocates the hash table/list for message variables and copy the list to device
*/
CUDAMessage::CUDAMessage(const MessageData& description)
    : message_description(description)
    , message_count(0)
    , max_list_size(0)
    , cub_temp_size_max_list_size(0)
    , cub_temp_size(0)
    , d_cub_temp(nullptr)
    , curve(Curve::getInstance()) {
    // resize(0); // Think this call is redundant
}

/**
 * A destructor.
 * @brief Destroys the CUDAMessage object
 */
CUDAMessage::~CUDAMessage(void) {
    if (d_cub_temp) {
        gpuErrchk(cudaFree(d_cub_temp));
    }
}

/**
* @brief Returns message description
* @param none
* @return MessageDescription object
*/
const MessageData& CUDAMessage::getMessageDescription() const {
    return message_description;
}

/**
* @brief Sets initial message data to zero by allocating memory for message lists
* @param empty
* @return none
*/

void CUDAMessage::resize(unsigned int newSize, const unsigned int &streamId) {
    // Only grow currently
    max_list_size = max_list_size < 2 ? 2 : max_list_size;
    if (newSize > max_list_size) {
        while (max_list_size < newSize) {
            max_list_size = static_cast<unsigned int>(max_list_size * 1.5);
        }
        // This drops old message data
        message_list = std::unique_ptr<CUDAMessageList>(new CUDAMessageList(*this));
        flamegpu_internal::CUDAScanCompaction::resizeMessages(max_list_size, streamId);

// #ifdef _DEBUG
        /**set the message list to zero*/
        zeroAllMessageData();
// #endif
    }
    message_count = newSize;  // This will be reduced down after function call if optional
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
unsigned int CUDAMessage::getMessageCount() const {
    return message_count;
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
void CUDAMessage::mapReadRuntimeVariables(const AgentFunctionData& func) const {
    // check that the message list has been allocated
    if (!message_list) {
        THROW InvalidMessageData("Error: Initial message list for message '%s' has not been allocated, "
            "in CUDAMessage::mapRuntimeVariables()",
            message_description.name.c_str());
    }

    const std::string message_name = message_description.name;

    const Curve::VariableHash message_hash = curve.variableRuntimeHash(message_name.c_str());
    const Curve::VariableHash agent_hash = curve.variableRuntimeHash(func.parent.lock()->name.c_str());
    const Curve::VariableHash func_hash = curve.variableRuntimeHash(func.name.c_str());
    // loop through the message variables to map each variable name using cuRVE
    for (const auto &mmp : message_description.variables) {
        // get a device pointer for the message variable name
        void* d_ptr = message_list->getReadMessageListVariablePointer(mmp.first);

        // map using curve
        Curve::VariableHash var_hash = curve.variableRuntimeHash(mmp.first.c_str());

        // get the message variable size
        size_t size = mmp.second.type_size;

       // maximum population size
        unsigned int length = this->getMessageCount();  // check to see if it is equal to pop
        curve.registerVariableByHash(var_hash + agent_hash + func_hash + message_hash, d_ptr, size, length);
    }
}
void CUDAMessage::mapWriteRuntimeVariables(const AgentFunctionData& func) const {
    // check that the message list has been allocated
    if (!message_list) {
        THROW InvalidMessageData("Error: Initial message list for message '%s' has not been allocated, "
            "in CUDAMessage::mapRuntimeVariables()",
            message_description.name.c_str());
    }

    const std::string message_name = message_description.name;

    const Curve::VariableHash message_hash = curve.variableRuntimeHash(message_name.c_str());
    const Curve::VariableHash agent_hash = curve.variableRuntimeHash(func.parent.lock()->name.c_str());
    const Curve::VariableHash func_hash = curve.variableRuntimeHash(func.name.c_str());
    // loop through the message variables to map each variable name using cuRVE
    for (const auto &mmp : message_description.variables) {
        // get a device pointer for the message variable name
        void* d_ptr = message_list->getWriteMessageListVariablePointer(mmp.first);

        // map using curve
        Curve::VariableHash var_hash = curve.variableRuntimeHash(mmp.first.c_str());

        // get the message variable size
        size_t size = mmp.second.type_size;

        // maximum population size
        unsigned int length = this->getMessageCount();  // check to see if it is equal to pop
        curve.registerVariableByHash(var_hash + agent_hash + func_hash + message_hash, d_ptr, size, length);
    }
}

void CUDAMessage::unmapRuntimeVariables(const AgentFunctionData& func) const {
    const std::string message_name = message_description.name;

    const Curve::VariableHash message_hash = curve.variableRuntimeHash(message_name.c_str());
    const Curve::VariableHash agent_hash = curve.variableRuntimeHash(func.parent.lock()->name.c_str());
    const Curve::VariableHash func_hash = curve.variableRuntimeHash(func.name.c_str());
    // loop through the message variables to map each variable name using cuRVE
    for (const auto &mmp : message_description.variables) {
        // get a device pointer for the message variable name
        // void* d_ptr = message_list->getMessageListVariablePointer(mmp.first);

        // unmap using curve
        Curve::VariableHash var_hash = curve.variableRuntimeHash(mmp.first.c_str());
        curve.unregisterVariableByHash(var_hash + agent_hash + func_hash + message_hash);
    }
}
void CUDAMessage::swap(bool isOptional, const unsigned int &streamId) {
    if (isOptional && message_description.optional_outputs > 0) {
        if (message_count > cub_temp_size_max_list_size) {
            if (d_cub_temp) {
                gpuErrchk(cudaFree(d_cub_temp));
            }
            cub_temp_size = 0;
            cub::DeviceScan::ExclusiveSum(
                nullptr,
                cub_temp_size,
                flamegpu_internal::CUDAScanCompaction::hd_message_configs[streamId].d_ptrs.scan_flag,
                flamegpu_internal::CUDAScanCompaction::hd_message_configs[streamId].d_ptrs.position,
                max_list_size + 1);
            gpuErrchk(cudaMalloc(&d_cub_temp, cub_temp_size));
            cub_temp_size_max_list_size = max_list_size;
        }
        cub::DeviceScan::ExclusiveSum(
            d_cub_temp,
            cub_temp_size,
            flamegpu_internal::CUDAScanCompaction::hd_message_configs[streamId].d_ptrs.scan_flag,
            flamegpu_internal::CUDAScanCompaction::hd_message_configs[streamId].d_ptrs.position,
            message_count + 1);
        // Scatter
        message_list->scatter(streamId);
        // Update count (must come after scatter, scatter requires old count)
        gpuErrchk(cudaMemcpy(&message_count, flamegpu_internal::CUDAScanCompaction::hd_message_configs[streamId].d_ptrs.position + message_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    } else {
        message_list->swap();
    }
}
