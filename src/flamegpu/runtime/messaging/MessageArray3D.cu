#include "flamegpu/runtime/messaging/MessageArray3D.h"
#include "flamegpu/model/AgentDescription.h"  // Used by Move-Assign
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/CUDAScatter.cuh"

#include "flamegpu/runtime/messaging/MessageArray3D/MessageArray3DHost.h"
// #include "flamegpu/runtime/messaging/MessageArray3D/MessageArray3DDevice.cuh"

namespace flamegpu {

/**
 * Constructor
 * Allocates memory on device for message list length
 * @param a Parent CUDAMessage, used to access message settings, data ptrs etc
 */
MessageArray3D::CUDAModelHandler::CUDAModelHandler(CUDAMessage &a)
    : MessageSpecialisationHandler()
    , d_metadata(nullptr)
    , sim_message(a)
    , d_write_flag(nullptr)
    , d_write_flag_len(0) {
    const Data& d = static_cast<const Data &>(a.getMessageDescription());
    memcpy(&hd_metadata.dimensions, d.dimensions.data(), d.dimensions.size() * sizeof(unsigned int));
    hd_metadata.length = d.dimensions[0] * d.dimensions[1] * d.dimensions[2];
}

void MessageArray3D::CUDAModelHandler::init(CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) {
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
        gpuErrchk(cudaMemsetAsync(write_list.at(var.first), 0, var.second.type_size * var.second.elements * hd_metadata.length, stream));
        gpuErrchk(cudaMemsetAsync(read_list.at(var.first), 0, var.second.type_size * var.second.elements * hd_metadata.length, stream));
    }
    gpuErrchk(cudaStreamSynchronize(stream));
}
void MessageArray3D::CUDAModelHandler::allocateMetaDataDevicePtr(cudaStream_t stream) {
    if (d_metadata == nullptr) {
        gpuErrchk(cudaMalloc(&d_metadata, sizeof(MetaData)));
        gpuErrchk(cudaMemcpyAsync(d_metadata, &hd_metadata, sizeof(MetaData), cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
    }
}

void MessageArray3D::CUDAModelHandler::freeMetaDataDevicePtr() {
    if (d_metadata != nullptr) {
        gpuErrchk(cudaFree(d_metadata));
    }
    d_metadata = nullptr;

    if (d_write_flag) {
        gpuErrchk(cudaFree(d_write_flag));
    }
    d_write_flag = nullptr;
    d_write_flag_len = 0;
}
void MessageArray3D::CUDAModelHandler::buildIndex(CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) {
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
                gpuErrchk(cudaFree(d_write_flag));
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


MessageArray3D::Data::Data(const std::shared_ptr<const ModelData> &model, const std::string &message_name)
    : MessageBruteForce::Data(model, message_name)
    , dimensions({0, 0, 0}) {
    description = std::unique_ptr<MessageArray3D::Description>(new MessageArray3D::Description(model, this));
    variables.emplace("___INDEX", Variable(1, size_type()));
}
MessageArray3D::Data::Data(const std::shared_ptr<const ModelData> &model, const Data &other)
    : MessageBruteForce::Data(model, other)
    , dimensions(other.dimensions) {
    description = std::unique_ptr<MessageArray3D::Description>(model ? new MessageArray3D::Description(model, this) : nullptr);
    if (dimensions[0] == 0 || dimensions[1] == 0 || dimensions[2] == 0) {
        THROW exception::InvalidMessage("All dimensions must be above zero in array3D message '%s'\n", other.name.c_str());
    }
}
MessageArray3D::Data *MessageArray3D::Data::clone(const std::shared_ptr<const ModelData> &newParent) {
    return new Data(newParent, *this);
}
std::unique_ptr<MessageSpecialisationHandler> MessageArray3D::Data::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MessageSpecialisationHandler>(new CUDAModelHandler(owner));
}
std::type_index MessageArray3D::Data::getType() const { return std::type_index(typeid(MessageArray3D)); }


MessageArray3D::Description::Description(const std::shared_ptr<const ModelData> &_model, Data *const data)
    : MessageBruteForce::Description(_model, data) { }

void MessageArray3D::Description::setDimensions(const size_type& len_x, const size_type& len_y, const size_type& len_z) {
    setDimensions({ len_x , len_y, len_z});
}
void MessageArray3D::Description::setDimensions(const std::array<size_type, 3> &dims) {
    if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0) {
        THROW exception::InvalidArgument("All dimensions must be above zero in array3D message.\n");
    }
    reinterpret_cast<Data *>(message)->dimensions = dims;
}
std::array<MessageArray3D::size_type, 3> MessageArray3D::Description::getDimensions() const {
    return reinterpret_cast<Data *>(message)->dimensions;
}
MessageArray2D::size_type MessageArray3D::Description::getDimX() const {
    return reinterpret_cast<Data *>(message)->dimensions[0];
}
MessageArray2D::size_type MessageArray3D::Description::getDimY() const {
    return reinterpret_cast<Data *>(message)->dimensions[1];
}
MessageArray2D::size_type MessageArray3D::Description::getDimZ() const {
    return reinterpret_cast<Data *>(message)->dimensions[2];
}

}  // namespace flamegpu
