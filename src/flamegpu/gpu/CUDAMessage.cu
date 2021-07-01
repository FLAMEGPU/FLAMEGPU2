#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/gpu/CUDAMessageList.h"
#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"
#include "flamegpu/gpu/CUDAScatter.cuh"

#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/runtime/detail/curve/curve.cuh"
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

CUDAMessage::CUDAMessage(const MsgBruteForce::Data& description, const CUDASimulation& cudaSimulation)
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

const MsgBruteForce::Data& CUDAMessage::getMessageDescription() const {
    return message_description;
}

void CUDAMessage::resize(unsigned int newSize, CUDAScatter &scatter, const unsigned int &streamId) {
    // Only grow currently
    if (newSize > max_list_size) {
        max_list_size = std::max<unsigned int>(max_list_size, 2u);
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
void CUDAMessage::init(CUDAScatter &scatter, const unsigned int &streamId) {
    specialisation_handler->init(scatter, streamId);
}
void CUDAMessage::zeroAllMessageData() {
    if (!message_list) {
        THROW exception::InvalidMessageData("MessageList '%s' is not yet allocated, in CUDAMessage::swap()\n", message_description.name.c_str());
    }
    message_list->zeroMessageData();
}

void CUDAMessage::mapReadRuntimeVariables(const AgentFunctionData& func, const CUDAAgent& cuda_agent, const unsigned int &instance_id) const {
    // check that the message list has been allocated
    if (!message_list) {
        if (getMessageCount() == 0) {
            return;  // Message list is empty, this should be safe
        }
        THROW exception::InvalidMessageData("Error: Initial message list for message '%s' has not been allocated, "
            "in CUDAMessage::mapRuntimeVariables()",
            message_description.name.c_str());
    }

    const std::string message_name = message_description.name;

    const detail::curve::Curve::VariableHash message_hash = detail::curve::Curve::getInstance().variableRuntimeHash(message_name.c_str());
    const detail::curve::Curve::VariableHash agent_hash = detail::curve::Curve::getInstance().variableRuntimeHash(func.parent.lock()->name.c_str());
    const detail::curve::Curve::VariableHash func_hash = detail::curve::Curve::getInstance().variableRuntimeHash(func.name.c_str());
    auto &curve = detail::curve::Curve::getInstance();
    // loop through the message variables to map each variable name using cuRVE
    for (const auto &mmp : message_description.variables) {
        // get a device pointer for the message variable name
        void* d_ptr = message_list->getReadMessageListVariablePointer(mmp.first);

        // map using curve
        detail::curve::Curve::VariableHash var_hash = detail::curve::Curve::getInstance().variableRuntimeHash(mmp.first.c_str());

        // get the message variable size
        size_t size = mmp.second.type_size;

        if (func.func) {
            // maximum population size
            unsigned int length = this->getMessageCount();  // check to see if it is equal to pop
#ifdef _DEBUG
            const detail::curve::Curve::Variable cv = curve.registerVariableByHash(var_hash + agent_hash + func_hash + message_hash + instance_id, d_ptr, size, length);
            if (cv != static_cast<int>((var_hash + agent_hash + func_hash + message_hash + instance_id)%detail::curve::Curve::MAX_VARIABLES)) {
                fprintf(stderr, "detail::curve::Curve Warning: Agent Function '%s' Message In Variable '%s' has a collision and may work improperly.\n", message_name.c_str(), mmp.first.c_str());
            }
#else
            curve.registerVariableByHash(var_hash + agent_hash + func_hash + message_hash + instance_id, d_ptr, size, length);
#endif
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
void CUDAMessage::mapWriteRuntimeVariables(const AgentFunctionData& func, const CUDAAgent& cuda_agent, const unsigned int &writeLen, const unsigned int &instance_id) const {
    // check that the message list has been allocated
    if (!message_list) {
        THROW exception::InvalidMessageData("Error: Initial message list for message '%s' has not been allocated, "
            "in CUDAMessage::mapRuntimeVariables()",
            message_description.name.c_str());
    }

    const std::string message_name = message_description.name;

    const detail::curve::Curve::VariableHash message_hash = detail::curve::Curve::getInstance().variableRuntimeHash(message_name.c_str());
    const detail::curve::Curve::VariableHash agent_hash = detail::curve::Curve::getInstance().variableRuntimeHash(func.parent.lock()->name.c_str());
    const detail::curve::Curve::VariableHash func_hash = detail::curve::Curve::getInstance().variableRuntimeHash(func.name.c_str());
    auto &curve = detail::curve::Curve::getInstance();
    // loop through the message variables to map each variable name using cuRVE
    for (const auto &mmp : message_description.variables) {
        // get a device pointer for the message variable name
        void* d_ptr = message_list->getWriteMessageListVariablePointer(mmp.first);

        // map using curve
        detail::curve::Curve::VariableHash var_hash = detail::curve::Curve::variableRuntimeHash(mmp.first.c_str());

        // get the message variable size
        size_t size = mmp.second.type_size;

        if (func.func) {
            // maximum population size
            unsigned int length = writeLen;  // check to see if it is equal to pop
#ifdef _DEBUG
            const detail::curve::Curve::Variable cv = curve.registerVariableByHash(var_hash + agent_hash + func_hash + message_hash + instance_id, d_ptr, size, length);
            if (cv != static_cast<int>((var_hash + agent_hash + func_hash + message_hash + instance_id)%detail::curve::Curve::MAX_VARIABLES)) {
                fprintf(stderr, "detail::curve::Curve Warning: Agent Function '%s' Message '%s' Out? Variable '%s' has a collision and may work improperly.\n", func.name.c_str(), message_name.c_str(), mmp.first.c_str());
            }
#else
            curve.registerVariableByHash(var_hash + agent_hash + func_hash + message_hash + instance_id, d_ptr, size, length);
#endif
        } else {
            // Map RTC variables (these must be mapped before each function execution as the runtime pointer may have changed to the swapping)
            // Copy data to rtc header cache
            auto& rtc_header = cuda_agent.getRTCHeader(func.name);
            memcpy(rtc_header.getMessageOutVariableCachePtr(mmp.first.c_str()), &d_ptr, sizeof(void*));
        }
    }

    // Allocate the metadata if required. (This call should now be redundant)
    specialisation_handler->allocateMetaDataDevicePtr();
}

void CUDAMessage::unmapRuntimeVariables(const AgentFunctionData& func, const unsigned int &instance_id) const {
    // Skip if RTC
    if (!func.func)
        return;
    if (!message_list) {
      if (getMessageCount() == 0) {
          return;  // Message list is empty, this should be safe
      }
    }
    const std::string message_name = message_description.name;

    const detail::curve::Curve::VariableHash message_hash = detail::curve::Curve::getInstance().variableRuntimeHash(message_name.c_str());
    const detail::curve::Curve::VariableHash agent_hash = detail::curve::Curve::getInstance().variableRuntimeHash(func.parent.lock()->name.c_str());
    const detail::curve::Curve::VariableHash func_hash = detail::curve::Curve::getInstance().variableRuntimeHash(func.name.c_str());
    auto &curve = detail::curve::Curve::getInstance();
    // loop through the message variables to map each variable name using cuRVE
    for (const auto &mmp : message_description.variables) {
        // unmap using curve
        detail::curve::Curve::VariableHash var_hash = detail::curve::Curve::variableRuntimeHash(mmp.first.c_str());
        curve.unregisterVariableByHash(var_hash + agent_hash + func_hash + message_hash + instance_id);
    }
}
void CUDAMessage::swap(bool isOptional, const unsigned int &newMsgCount, CUDAScatter &scatter, const unsigned int &streamId) {
    if (!message_list) {
        THROW exception::InvalidMessageData("MessageList '%s' is not yet allocated, in CUDAMessage::swap()\n", message_description.name.c_str());
    }
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
