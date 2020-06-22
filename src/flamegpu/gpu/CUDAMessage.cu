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
#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/gpu/CUDAMessageList.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/gpu/CUDAScatter.h"

#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/runtime/messaging.h"

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
CUDAMessage::CUDAMessage(const MsgBruteForce::Data& description, const CUDAAgentModel& cuda_model)
    : message_description(description)
    , message_count(0)
    , max_list_size(0)
    , truncate_messagelist_flag(true)
    , pbm_construction_required(false)
    , specialisation_handler(description.getSpecialisationHander(*this))
    , cuda_model(cuda_model) {
    // resize(0); // Think this call is redundant
}

/**
 * A destructor.
 * @brief Destroys the CUDAMessage object
 */
CUDAMessage::~CUDAMessage(void) {
    // @todo - this should not be done in a destructor, rather an explicit cleanup method.
    specialisation_handler->freeMetaDataDevicePtr();
}

/**
* @brief Returns message description
* @param none
* @return MessageDescription object
*/
const MsgBruteForce::Data& CUDAMessage::getMessageDescription() const {
    return message_description;
}

/**
* @brief Sets initial message data to zero by allocating memory for message lists
* @param empty
* @return none
*/

void CUDAMessage::resize(unsigned int newSize, CUDAScatter &scatter, const unsigned int &streamId) {
    // Only grow currently
    max_list_size = max_list_size < 2 ? 2 : max_list_size;
    if (newSize > max_list_size) {
        while (max_list_size < newSize) {
            max_list_size = static_cast<unsigned int>(max_list_size * 1.5);
        }
        // This drops old message data
        message_list = std::unique_ptr<CUDAMessageList>(new CUDAMessageList(*this, scatter, streamId));
        scatter.Scan().resize(max_list_size, CUDAScanCompaction::MESSAGE_OUTPUT, streamId);

// #ifdef _DEBUG
        /**set the message list to zero*/
        zeroAllMessageData();
// #endif
    }
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
void CUDAMessage::setMessageCount(const unsigned int &_message_count) {
    if (_message_count > max_list_size) {
        THROW OutOfBoundsException("message count exceeds allocated message list size (%u > %u) in CUDAMessage::setMessageCount().", _message_count, max_list_size);
    }
    message_count = _message_count;
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
void CUDAMessage::mapReadRuntimeVariables(const AgentFunctionData& func, const CUDAAgent& cuda_agent) const {
    // check that the message list has been allocated
    if (!message_list) {
        THROW InvalidMessageData("Error: Initial message list for message '%s' has not been allocated, "
            "in CUDAMessage::mapRuntimeVariables()",
            message_description.name.c_str());
    }

    const std::string message_name = message_description.name;

    const Curve::VariableHash message_hash = Curve::getInstance().variableRuntimeHash(message_name.c_str());
    const Curve::VariableHash agent_hash = Curve::getInstance().variableRuntimeHash(func.parent.lock()->name.c_str());
    const Curve::VariableHash func_hash = Curve::getInstance().variableRuntimeHash(func.name.c_str());
    // loop through the message variables to map each variable name using cuRVE
    for (const auto &mmp : message_description.variables) {
        // get a device pointer for the message variable name
        void* d_ptr = message_list->getReadMessageListVariablePointer(mmp.first);

        // map using curve
        Curve::VariableHash var_hash = Curve::getInstance().variableRuntimeHash(mmp.first.c_str());

        // get the message variable size
        size_t size = mmp.second.type_size;

       // maximum population size
        unsigned int length = this->getMessageCount();  // check to see if it is equal to pop
        Curve::getInstance().registerVariableByHash(var_hash + agent_hash + func_hash + message_hash, d_ptr, size, length);

        // Map RTC variables (these must be mapped before each function execution as the runtime pointer may have changed to the swapping)
        if (!func.rtc_func_name.empty()) {
            // get the rtc variable ptr
            const jitify::KernelInstantiation& instance = cuda_agent.getRTCInstantiation(func.rtc_func_name);
            std::stringstream d_var_ptr_name;
            d_var_ptr_name << "curve_rtc_ptr_" << agent_hash + func_hash + message_hash << "_" << mmp.first;
            CUdeviceptr d_var_ptr = instance.get_global_ptr(d_var_ptr_name.str().c_str());
            // copy runtime ptr (d_ptr) to rtc ptr (d_var_ptr)
            gpuErrchkDriverAPI(cuMemcpyHtoD(d_var_ptr, &d_ptr, sizeof(void*)));
        }
    }
}

void *CUDAMessage::getReadPtr(const std::string &var_name) {
    return message_list->getReadMessageListVariablePointer(var_name);
}
void CUDAMessage::mapWriteRuntimeVariables(const AgentFunctionData& func, const CUDAAgent& cuda_agent, const unsigned int &writeLen) const {
    // check that the message list has been allocated
    if (!message_list) {
        THROW InvalidMessageData("Error: Initial message list for message '%s' has not been allocated, "
            "in CUDAMessage::mapRuntimeVariables()",
            message_description.name.c_str());
    }

    const std::string message_name = message_description.name;

    const Curve::VariableHash message_hash = Curve::getInstance().variableRuntimeHash(message_name.c_str());
    const Curve::VariableHash agent_hash = Curve::getInstance().variableRuntimeHash(func.parent.lock()->name.c_str());
    const Curve::VariableHash func_hash = Curve::getInstance().variableRuntimeHash(func.name.c_str());
    // loop through the message variables to map each variable name using cuRVE
    for (const auto &mmp : message_description.variables) {
        // get a device pointer for the message variable name
        void* d_ptr = message_list->getWriteMessageListVariablePointer(mmp.first);

        // map using curve
        Curve::VariableHash var_hash = Curve::getInstance().variableRuntimeHash(mmp.first.c_str());

        // get the message variable size
        size_t size = mmp.second.type_size;

        // maximum population size
        unsigned int length = writeLen;  // check to see if it is equal to pop
        Curve::getInstance().registerVariableByHash(var_hash + agent_hash + func_hash + message_hash, d_ptr, size, length);

        // Map RTC variables (these must be mapped before each function execution as the runtime pointer may have changed to the swapping)
        if (!func.rtc_func_name.empty()) {
            // get the rtc variable ptr
            const jitify::KernelInstantiation& instance = cuda_agent.getRTCInstantiation(func.rtc_func_name);
            std::stringstream d_var_ptr_name;
            d_var_ptr_name << "curve_rtc_ptr_" << agent_hash + func_hash + message_hash << "_" << mmp.first;
            CUdeviceptr d_var_ptr = instance.get_global_ptr(d_var_ptr_name.str().c_str());
            // copy runtime ptr (d_ptr) to rtc ptr (d_var_ptr)
            gpuErrchkDriverAPI(cuMemcpyHtoD(d_var_ptr, &d_ptr, sizeof(void*)));
        }
    }

    // Allocate the metadata if required.
    specialisation_handler->allocateMetaDataDevicePtr();
}

void CUDAMessage::unmapRuntimeVariables(const AgentFunctionData& func) const {
    const std::string message_name = message_description.name;

    const Curve::VariableHash message_hash = Curve::getInstance().variableRuntimeHash(message_name.c_str());
    const Curve::VariableHash agent_hash = Curve::getInstance().variableRuntimeHash(func.parent.lock()->name.c_str());
    const Curve::VariableHash func_hash = Curve::getInstance().variableRuntimeHash(func.name.c_str());
    // loop through the message variables to map each variable name using cuRVE
    for (const auto &mmp : message_description.variables) {
        // get a device pointer for the message variable name
        // void* d_ptr = message_list->getMessageListVariablePointer(mmp.first);

        // unmap using curve
        Curve::VariableHash var_hash = Curve::getInstance().variableRuntimeHash(mmp.first.c_str());
        Curve::getInstance().unregisterVariableByHash(var_hash + agent_hash + func_hash + message_hash);
    }
}
void CUDAMessage::swap(bool isOptional, const unsigned int &newMsgCount, CUDAScatter &scatter, const unsigned int &streamId) {
    if (isOptional && message_description.optional_outputs > 0) {
        CUDAScanCompactionConfig &scanCfg = scatter.Scan().Config(CUDAScanCompaction::Type::MESSAGE_OUTPUT, streamId);
        if (newMsgCount > scanCfg.cub_temp_size_max_list_size) {
            if (scanCfg.hd_cub_temp) {
                gpuErrchk(cudaFree(scanCfg.hd_cub_temp));
            }
            scanCfg.cub_temp_size = 0;
            gpuErrchk(cub::DeviceScan::ExclusiveSum(
                nullptr,
                scanCfg.cub_temp_size,
                scanCfg.d_ptrs.scan_flag,
                scanCfg.d_ptrs.position,
                max_list_size + 1));
            gpuErrchk(cudaMalloc(&scanCfg.hd_cub_temp,
                scanCfg.cub_temp_size));
            scanCfg.cub_temp_size_max_list_size = max_list_size;
        }
        gpuErrchk(cub::DeviceScan::ExclusiveSum(
            scanCfg.hd_cub_temp,
            scanCfg.cub_temp_size,
            scanCfg.d_ptrs.scan_flag,
            scanCfg.d_ptrs.position,
            newMsgCount + 1));
        // Scatter
        // Update count
        message_count = message_list->scatter(newMsgCount, scatter, streamId, !this->truncate_messagelist_flag);
    } else {
        if (this->truncate_messagelist_flag) {
            message_count = newMsgCount;
            message_list->swap();
        } else {
            assert(message_count + newMsgCount <= max_list_size);
            // We're appending so use our scatter kernel
            message_count = message_list->scatterAll(newMsgCount, scatter, streamId);
        }
    }
}
void CUDAMessage::swap() {
    message_list->swap();
}

void CUDAMessage::buildIndex(CUDAScatter &scatter, const unsigned int &streamId) {
    // Build the index if required.
    if (pbm_construction_required) {
        specialisation_handler->buildIndex(scatter, streamId);
        pbm_construction_required = false;
    }
}
const void *CUDAMessage::getMetaDataDevicePtr() const {
    return specialisation_handler->getMetaDataDevicePtr();
}
