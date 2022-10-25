#include "flamegpu/runtime/messaging/MessageArray2D.h"
#include "flamegpu/model/AgentDescription.h"  // Used by Move-Assign
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/CUDAScatter.cuh"

#include "flamegpu/runtime/messaging/MessageArray2D/MessageArray2DHost.h"
// #include "flamegpu/runtime/messaging/MessageArray2D/MessageArray2DDevice.cuh"

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
        gpuErrchk(cudaFree(d_metadata));
    }
    d_metadata = nullptr;

    if (d_write_flag) {
        gpuErrchk(cudaFree(d_write_flag));
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


MessageArray2D::Data::Data(const std::shared_ptr<const ModelData> &model, const std::string &message_name)
    : MessageBruteForce::Data(model, message_name)
    , dimensions({ 0, 0 }) {
    description = std::unique_ptr<MessageArray2D::Description>(new MessageArray2D::Description(model, this));
    variables.emplace("___INDEX", Variable(1, size_type()));
}
MessageArray2D::Data::Data(const std::shared_ptr<const ModelData> &model, const Data &other)
    : MessageBruteForce::Data(model, other)
    , dimensions(other.dimensions) {
    description = std::unique_ptr<MessageArray2D::Description>(model ? new MessageArray2D::Description(model, this) : nullptr);
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


MessageArray2D::Description::Description(const std::shared_ptr<const ModelData> &_model, Data *const data)
    : MessageBruteForce::Description(_model, data) { }

void MessageArray2D::Description::setDimensions(const size_type len_x, const size_type len_y) {
    setDimensions({ len_x , len_y });
}
void MessageArray2D::Description::setDimensions(const std::array<size_type, 2> &dims) {
    if (dims[0] == 0 || dims[1] == 0) {
        THROW exception::InvalidArgument("All dimensions must be above zero in array2D message.\n");
    }
    reinterpret_cast<Data *>(message)->dimensions = dims;
}
std::array<MessageArray2D::size_type, 2> MessageArray2D::Description::getDimensions() const {
    return reinterpret_cast<Data *>(message)->dimensions;
}
MessageArray2D::size_type MessageArray2D::Description::getDimX() const {
    return reinterpret_cast<Data *>(message)->dimensions[0];
}
MessageArray2D::size_type MessageArray2D::Description::getDimY() const {
    return reinterpret_cast<Data *>(message)->dimensions[1];
}

}  // namespace flamegpu
