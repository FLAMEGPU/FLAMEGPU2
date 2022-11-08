#include "flamegpu/runtime/messaging/MessageArray2D.h"
#include "flamegpu/model/AgentDescription.h"  // Used by Move-Assign
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/CUDAScatter.cuh"

#include "flamegpu/runtime/messaging/MessageArray2D/MessageArray2DHost.h"
// #include "flamegpu/runtime/messaging/MessageArray2D/MessageArray2DDevice.cuh"
#include "flamegpu/util/detail/cuda.cuh"

namespace flamegpu {

/**
 * Constructor
 * Allocates memory on device for message list length
 * @param a Parent CUDAMessage, used to access message settings, data ptrs etc
 */
MessageArray2D::CUDAModelHandler::CUDAModelHandler(CUDAMessage &a)
    : MessageSpecialisationHandler()
    , d_metadata(nullptr)
    , sim_message(a)
    , d_write_flag(nullptr)
    , d_write_flag_len(0) {
    const Data& d = static_cast<const Data &>(a.getMessageDescription());
    memcpy(&hd_metadata.dimensions, d.dimensions.data(), d.dimensions.size() * sizeof(unsigned int));
    hd_metadata.length = d.dimensions[0] * d.dimensions[1];
}

void MessageArray2D::CUDAModelHandler::init(CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) {
    allocateMetaDataDevicePtr(stream);
    // Allocate messages
    this->sim_message.resize(hd_metadata.length, scatter, stream, streamId);
    this->sim_message.setMessageCount(hd_metadata.length);
    // Zero the output arrays
    auto &read_list = this->sim_message.getReadList();
    auto &write_list = this->sim_message.getWriteList();
    for (auto &var : this->sim_message.getMessageDescription().variables) {
        // Elements is harmless, futureproof for arrays support
        // hd_metadata.length is used, as message array can be longer than message count
        gpuErrchk(cudaMemsetAsync(write_list.at(var.first), 0, var.second.type_size * var.second.elements * hd_metadata.length));
        gpuErrchk(cudaMemsetAsync(read_list.at(var.first), 0, var.second.type_size * var.second.elements * hd_metadata.length));
    }
    gpuErrchk(cudaStreamSynchronize(stream));
}
void MessageArray2D::CUDAModelHandler::allocateMetaDataDevicePtr(cudaStream_t stream) {
    if (d_metadata == nullptr) {
        gpuErrchk(cudaMalloc(&d_metadata, sizeof(MetaData)));
        gpuErrchk(cudaMemcpyAsync(d_metadata, &hd_metadata, sizeof(MetaData), cudaMemcpyHostToDevice));
        gpuErrchk(cudaStreamSynchronize(stream));
    }
}

void MessageArray2D::CUDAModelHandler::freeMetaDataDevicePtr() {
    if (d_metadata != nullptr) {
        gpuErrchk(flamegpu::util::detail::cuda::cudaFree(d_metadata));
    }
    d_metadata = nullptr;

    if (d_write_flag) {
        gpuErrchk(flamegpu::util::detail::cuda::cudaFree(d_write_flag));
    }
    d_write_flag = nullptr;
    d_write_flag_len = 0;
}
void MessageArray2D::CUDAModelHandler::buildIndex(CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) {
    const unsigned int MESSAGE_COUNT = this->sim_message.getMessageCount();
    // Zero the output arrays
    auto &read_list = this->sim_message.getReadList();
    auto &write_list = this->sim_message.getWriteList();
    for (auto &var : this->sim_message.getMessageDescription().variables) {
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
                gpuErrchk(flamegpu::util::detail::cuda::cudaFree(d_write_flag));
            }
            d_write_flag_len = static_cast<unsigned int>(MESSAGE_COUNT * 1.1f);
            gpuErrchk(cudaMalloc(&d_write_flag, sizeof(unsigned int) * d_write_flag_len));
        }
        t_d_write_flag = d_write_flag;
    }
    scatter.arrayMessageReorder(streamId, stream, this->sim_message.getMessageDescription().variables, read_list, write_list, MESSAGE_COUNT, hd_metadata.length, t_d_write_flag);
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
MessageArray2D::CDescription::CDescription(std::shared_ptr<Data> data)
    : MessageBruteForce::Description(std::move(data)) { }
MessageArray2D::CDescription::CDescription(std::shared_ptr<const Data> data)
    : MessageBruteForce::Description(std::move(std::const_pointer_cast<Data>(data))) { }

bool MessageArray2D::CDescription::operator==(const CDescription& rhs) const {
    return *this->message == *rhs.message;  // Compare content is functionally the same
}
bool MessageArray2D::CDescription::operator!=(const CDescription& rhs) const {
    return !(*this == rhs);
}
/**
 * Const accessors
 */
std::array<flamegpu::size_type, 2> MessageArray2D::CDescription::getDimensions() const {
    return std::static_pointer_cast<Data>(message)->dimensions;
}
flamegpu::size_type MessageArray2D::CDescription::getDimX() const {
    return std::static_pointer_cast<Data>(message)->dimensions[0];
}
flamegpu::size_type MessageArray2D::CDescription::getDimY() const {
    return std::static_pointer_cast<Data>(message)->dimensions[1];
}

/// <summary>
/// Description
/// </summary>
MessageArray2D::Description::Description(std::shared_ptr<Data> data)
    : CDescription(data) { }
bool MessageArray2D::Description::operator==(const CDescription& rhs) const {
    return rhs == *this;  // Forward to superclass's equality
}
bool MessageArray2D::Description::operator!=(const CDescription& rhs) const {
    return !(*this == rhs);
}
/**
 * Accessors
 */
void MessageArray2D::Description::setDimensions(const size_type len_x, const size_type len_y) {
    setDimensions({ len_x , len_y });
}
void MessageArray2D::Description::setDimensions(const std::array<size_type, 2>& dims) {
    if (dims[0] == 0 || dims[1] == 0) {
        THROW exception::InvalidArgument("All dimensions must be above zero in array2D message.\n");
    }
    std::static_pointer_cast<Data>(message)->dimensions = dims;
}

/// <summary>
/// Data
/// </summary>
MessageArray2D::Data::Data(std::shared_ptr<const ModelData> model, const std::string &message_name)
    : MessageBruteForce::Data(model, message_name)
    , dimensions({ 0, 0 }) {
    variables.emplace("___INDEX", Variable(1, size_type()));
}
MessageArray2D::Data::Data(std::shared_ptr<const ModelData> model, const Data &other)
    : MessageBruteForce::Data(model, other)
    , dimensions(other.dimensions) {
    if (dimensions[0] == 0 || dimensions[1] == 0) {
        THROW exception::InvalidMessage("All dimensions must be ABOVE zero in array2D message '%s'\n", other.name.c_str());
    }
}
MessageArray2D::Data *MessageArray2D::Data::clone(const std::shared_ptr<const ModelData> &newParent) {
    return new Data(newParent, *this);
}
std::unique_ptr<MessageSpecialisationHandler> MessageArray2D::Data::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MessageSpecialisationHandler>(new CUDAModelHandler(owner));
}
std::type_index MessageArray2D::Data::getType() const { return std::type_index(typeid(MessageArray2D)); }

}  // namespace flamegpu
