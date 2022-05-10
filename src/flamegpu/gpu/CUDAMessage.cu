#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/gpu/CUDAMessageList.h"
#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"
#include "flamegpu/gpu/CUDAScatter.cuh"

#include "flamegpu/runtime/messaging/MessageBruteForce.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/runtime/detail/curve/HostCurve.cuh"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/runtime/messaging.h"

#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable : 4706 4834)
#include <cub/cub.cuh>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#endif

namespace flamegpu {

CUDAMessage::CUDAMessage(const MessageBruteForce::Data& description, const CUDASimulation& cudaSimulation)
    : message_description(description)
    , message_count(0)
    , max_list_size(0)
    , truncate_messagelist_flag(true)
    , pbm_construction_required(false)
    , specialisation_handler(description.getSpecialisationHander(*this))
    , cudaSimulation(cudaSimulation) {
    // resize(0); // Think this call is redundant
}

CUDAMessage::~CUDAMessage(void) {
    // @todo - this should not be done in a destructor, rather an explicit cleanup method.
    specialisation_handler->freeMetaDataDevicePtr();
}

const MessageBruteForce::Data& CUDAMessage::getMessageDescription() const {
    return message_description;
}

void CUDAMessage::resize(unsigned int newSize, CUDAScatter &scatter, cudaStream_t stream, unsigned int streamId, unsigned int keepLen) {
    // Only grow currently
    if (newSize > max_list_size) {
        const unsigned int _keep_len = std::min(max_list_size, keepLen);
        max_list_size = std::max<unsigned int>(max_list_size, 2u);
        while (max_list_size < newSize) {
            max_list_size = static_cast<unsigned int>(max_list_size * 1.5);
        }
        if (message_list) {
            message_list->resize(scatter, stream, streamId, _keep_len);
        } else {
            // If the list has not already been allocated, create a new
            message_list = std::unique_ptr<CUDAMessageList>(new CUDAMessageList(*this, scatter, stream, streamId));
        }
        scatter.Scan().resize(max_list_size, CUDAScanCompaction::MESSAGE_OUTPUT, streamId);
    }
}


unsigned int CUDAMessage::getMaximumListSize() const {
    return max_list_size;
}
unsigned int CUDAMessage::getMessageCount() const {
    return message_count;
}
void CUDAMessage::setMessageCount(const unsigned int &_message_count) {
    if (_message_count > max_list_size) {
        THROW exception::OutOfBoundsException("message count exceeds allocated message list size (%u > %u) in CUDAMessage::setMessageCount().", _message_count, max_list_size);
    }
    message_count = _message_count;
}
void CUDAMessage::init(CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) {
    specialisation_handler->init(scatter, streamId, stream);
}
void CUDAMessage::zeroAllMessageData(cudaStream_t stream) {
    if (!message_list) {
        THROW exception::InvalidMessageData("MessageList '%s' is not yet allocated, in CUDAMessage::swap()\n", message_description.name.c_str());
    }
    message_list->zeroMessageData(stream);
}

void CUDAMessage::mapReadRuntimeVariables(const AgentFunctionData& func, const CUDAAgent& cuda_agent) const {
    // check that the message list has been allocated
    if (!message_list) {
        if (getMessageCount() == 0) {
            return;  // Message list is empty, this should be safe
        }
        THROW exception::InvalidMessageData("Error: Initial message list for message '%s' has not been allocated, "
            "in CUDAMessage::mapRuntimeVariables()",
            message_description.name.c_str());
    }

    // loop through the message variables to map each variable name using cuRVE
    for (const auto &mmp : message_description.variables) {
        // get a device pointer for the message variable name
        void* d_ptr = message_list->getReadMessageListVariablePointer(mmp.first);

        if (func.func) {
            auto& curve = cuda_agent.getCurve(func.name);  // @todo fix the heavy map lookups
            // maximum population size
            unsigned int length = this->getMessageCount();  // check to see if it is equal to pop
            curve.setMessageInputVariable(mmp.first, d_ptr, length);
        } else {
            // Map RTC variables (these must be mapped before each function execution as the runtime pointer may have changed to the swapping)
            // Copy data to rtc header cache
            auto &rtc_header = cuda_agent.getRTCHeader(func.name);
            memcpy(rtc_header.getMessageInVariableCachePtr(mmp.first.c_str()), &d_ptr, sizeof(void*));
        }
    }
}

void *CUDAMessage::getReadPtr(const std::string &var_name) {
    if (!message_list) {
        THROW exception::InvalidMessageData("MessageList '%s' is not yet allocated, in CUDAMessage::swap()\n", message_description.name.c_str());
    }
    return message_list->getReadMessageListVariablePointer(var_name);
}
void CUDAMessage::mapWriteRuntimeVariables(const AgentFunctionData& func, const CUDAAgent& cuda_agent, const unsigned int &writeLen, cudaStream_t stream) const {
    // check that the message list has been allocated
    if (!message_list) {
        THROW exception::InvalidMessageData("Error: Initial message list for message '%s' has not been allocated, "
            "in CUDAMessage::mapRuntimeVariables()",
            message_description.name.c_str());
    }

    // loop through the message variables to map each variable name using cuRVE
    for (const auto &mmp : message_description.variables) {
        // get a device pointer for the message variable name
        void* d_ptr = message_list->getWriteMessageListVariablePointer(mmp.first);

        if (func.func) {
            auto &curve = cuda_agent.getCurve(func.name);  // @todo fix the heavy map lookups
            // maximum population size
            unsigned int length = writeLen;  // check to see if it is equal to pop
            curve.setMessageOutputVariable(mmp.first, d_ptr, length);
        } else {
            // Map RTC variables (these must be mapped before each function execution as the runtime pointer may have changed to the swapping)
            // Copy data to rtc header cache
            auto& rtc_header = cuda_agent.getRTCHeader(func.name);
            memcpy(rtc_header.getMessageOutVariableCachePtr(mmp.first.c_str()), &d_ptr, sizeof(void*));
        }
    }

    // Allocate the metadata if required. (This call should now be redundant)
    specialisation_handler->allocateMetaDataDevicePtr(stream);
}

void CUDAMessage::swap(bool isOptional, unsigned int newMessageCount, CUDAScatter &scatter, cudaStream_t stream, unsigned int streamId) {
    if (!message_list) {
        THROW exception::InvalidMessageData("MessageList '%s' is not yet allocated, in CUDAMessage::swap()\n", message_description.name.c_str());
    }
    if (isOptional && message_description.optional_outputs > 0) {
        auto &scanCfg = scatter.Scan().getConfig(CUDAScanCompaction::MESSAGE_OUTPUT, streamId);
        // Check if we need to resize cub storage
        auto& cub_temp = scatter.CubTemp(streamId);
        size_t tempByte = 0;
        gpuErrchk(cub::DeviceScan::ExclusiveSum(
            nullptr,
            tempByte,
            scanCfg.d_ptrs.scan_flag,
            scanCfg.d_ptrs.position,
            max_list_size + 1,
            stream));
        cub_temp.resize(tempByte);
        gpuErrchk(cub::DeviceScan::ExclusiveSum(
            cub_temp.getPtr(),
            cub_temp.getSize(),
            scanCfg.d_ptrs.scan_flag,
            scanCfg.d_ptrs.position,
            newMessageCount + 1,
            stream));
        // Scatter
        // Update count
        message_count = message_list->scatter(newMessageCount, scatter, stream, streamId, !this->truncate_messagelist_flag);
    } else {
        if (this->truncate_messagelist_flag) {
            message_count = newMessageCount;
            message_list->swap();
        } else {
            assert(message_count + newMessageCount <= max_list_size);
            // We're appending so use our scatter kernel
            message_count = message_list->scatterAll(newMessageCount, scatter, stream, streamId);
        }
    }
}
void CUDAMessage::swap() {
    if (!message_list) {
        THROW exception::InvalidMessageData("MessageList '%s' is not yet allocated, in CUDAMessage::swap()\n", message_description.name.c_str());
    }
    message_list->swap();
}

void CUDAMessage::buildIndex(CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    // Build the index if required.
    if (pbm_construction_required) {
        specialisation_handler->buildIndex(scatter, streamId, stream);
        pbm_construction_required = false;
    }
}
const void *CUDAMessage::getMetaDataDevicePtr() const {
    return specialisation_handler->getMetaDataDevicePtr();
}

}  // namespace flamegpu
