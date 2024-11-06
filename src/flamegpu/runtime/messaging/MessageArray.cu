#include "flamegpu/runtime/messaging/MessageArray.h"

#include <utility>
#include <string>
#include <memory>

#include "flamegpu/model/AgentDescription.h"  // Used by Move-Assign
#include "flamegpu/simulation/detail/CUDAMessage.h"
#include "flamegpu/simulation/detail/CUDAScatter.cuh"

#include "flamegpu/runtime/messaging/MessageArray/MessageArrayHost.h"
// #include "flamegpu/runtime/messaging/MessageArray/MessageArrayDevice.cuh"
#include "flamegpu/detail/cuda.cuh"

namespace flamegpu {

/**
 * Constructor
 * Allocates memory on device for message list length
 * @param a Parent CUDAMessage, used to access message settings, data ptrs etc
 */
MessageArray::CUDAModelHandler::CUDAModelHandler(detail::CUDAMessage &a)
    : MessageSpecialisationHandler()
    , d_metadata(nullptr)
    , sim_message(a)
    , d_write_flag(nullptr)
    , d_write_flag_len(0) {
    const Data& d = static_cast<const Data &>(a.getMessageData());
    hd_metadata.length = d.length;
}

void MessageArray::CUDAModelHandler::init(detail::CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) {
    allocateMetaDataDevicePtr(stream);
    // Allocate messages
    this->sim_message.resize(hd_metadata.length, scatter, stream, streamId);
    this->sim_message.setMessageCount(hd_metadata.length);
    // Zero the output arrays
    auto &read_list = this->sim_message.getReadList();
    auto &write_list = this->sim_message.getWriteList();
    for (auto &var : this->sim_message.getMessageData().variables) {
        // Elements is harmless, futureproof for arrays support
        // hd_metadata.length is used, as message array can be longer than message count
        gpuErrchk(cudaMemsetAsync(write_list.at(var.first), 0, var.second.type_size * var.second.elements * hd_metadata.length, stream));
        gpuErrchk(cudaMemsetAsync(read_list.at(var.first), 0, var.second.type_size * var.second.elements * hd_metadata.length, stream));
    }
    gpuErrchk(cudaStreamSynchronize(stream));
}
void MessageArray::CUDAModelHandler::allocateMetaDataDevicePtr(cudaStream_t stream) {
    if (d_metadata == nullptr) {
        gpuErrchk(cudaMalloc(&d_metadata, sizeof(MetaData)));
        gpuErrchk(cudaMemcpyAsync(d_metadata, &hd_metadata, sizeof(MetaData), cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
    }
}

void MessageArray::CUDAModelHandler::freeMetaDataDevicePtr() {
    if (d_metadata != nullptr) {
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_metadata));
    }
    d_metadata = nullptr;

    if (d_write_flag) {
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_write_flag));
    }
    d_write_flag = nullptr;
    d_write_flag_len = 0;
}
void MessageArray::CUDAModelHandler::buildIndex(detail::CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) {
    const unsigned int MESSAGE_COUNT = this->sim_message.getMessageCount();
    // Zero the output arrays
    auto &read_list = this->sim_message.getReadList();
    auto &write_list = this->sim_message.getWriteList();
    for (auto &var : this->sim_message.getMessageData().variables) {
        // Elements is harmless, futureproof for arrays support
        // hd_metadata.length is used, as message array can be longer than message count
        gpuErrchk(cudaMemsetAsync(write_list.at(var.first), 0, var.second.type_size * var.second.elements * hd_metadata.length, stream));
    }

    // Reorder messages
    unsigned int *t_d_write_flag = nullptr;
    if (MESSAGE_COUNT > hd_metadata.length) {
        // Use internal memory for d_write_flag
        if (d_write_flag_len < MESSAGE_COUNT) {
            // Increase length
            if (d_write_flag) {
                gpuErrchk(flamegpu::detail::cuda::cudaFree(d_write_flag));
            }
            d_write_flag_len = static_cast<unsigned int>(MESSAGE_COUNT * 1.1f);
            gpuErrchk(cudaMalloc(&d_write_flag, sizeof(unsigned int) * d_write_flag_len));
        }
        t_d_write_flag = d_write_flag;
    }
    scatter.arrayMessageReorder(streamId, stream, this->sim_message.getMessageData().variables, read_list, write_list, MESSAGE_COUNT, hd_metadata.length, t_d_write_flag);
    this->sim_message.swap();
    // Reset message count back to full array length
    // Array message exposes not output messages as 0
    if (MESSAGE_COUNT != hd_metadata.length)
        this->sim_message.setMessageCount(hd_metadata.length);
    // Detect errors
    // TODO
    gpuErrchk(cudaStreamSynchronize(stream));  // Redundant: Array msg reorder has a sync
}

/// <summary>
/// CDescription
/// </summary>
MessageArray::CDescription::CDescription(std::shared_ptr<Data> data)
    : MessageBruteForce::CDescription(std::move(std::static_pointer_cast<MessageBruteForce::Data>(data))) { }
MessageArray::CDescription::CDescription(std::shared_ptr<const Data> data)
    : CDescription(std::move(std::const_pointer_cast<Data>(data))) { }

bool MessageArray::CDescription::operator==(const CDescription& rhs) const {
    return *this->message == *rhs.message;  // Compare content is functionally the same
}
bool MessageArray::CDescription::operator!=(const CDescription& rhs) const {
    return !(*this == rhs);
}
/**
 * Const accessors
 */
flamegpu::size_type MessageArray::CDescription::getLength() const {
    return std::static_pointer_cast<Data>(message)->length;
}

/// <summary>
/// Description
/// </summary>
MessageArray::Description::Description(std::shared_ptr<Data> data)
    : CDescription(data) { }
/**
 * Accessors
 */
void MessageArray::Description::setLength(const size_type len) {
    if (len == 0) {
        THROW exception::InvalidArgument("Array messaging length must not be zero.\n");
    }
    std::static_pointer_cast<Data>(message)->length = len;
}

/// <summary>
/// Data
/// </summary>
MessageArray::Data::Data(std::shared_ptr<const ModelData> model, const std::string &message_name)
    : MessageBruteForce::Data(model, message_name)
    , length(0) {
    variables.emplace("___INDEX", Variable(1, size_type()));
}
MessageArray::Data::Data(std::shared_ptr<const ModelData> model, const Data &other)
    : MessageBruteForce::Data(model, other)
    , length(other.length) {
    if (length == 0) {
        THROW exception::InvalidMessage("Length must not be zero in array message '%s'\n", other.name.c_str());
    }
}
MessageArray::Data *MessageArray::Data::clone(const std::shared_ptr<const ModelData> &newParent) {
    return new Data(newParent, *this);
}
std::unique_ptr<MessageSpecialisationHandler> MessageArray::Data::getSpecialisationHander(detail::CUDAMessage &owner) const {
    return std::unique_ptr<MessageSpecialisationHandler>(new CUDAModelHandler(owner));
}
std::type_index MessageArray::Data::getType() const { return std::type_index(typeid(MessageArray)); }

}  // namespace flamegpu
