#include "flamegpu/runtime/messaging/Array3D.h"
#include "flamegpu/model/AgentDescription.h"  // Used by Move-Assign
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/CUDAScatter.h"

#include "flamegpu/runtime/messaging/Array3D/Array3DHost.h"
// #include "flamegpu/runtime/messaging/Array3D/Array3DDevice.h"

/**
 * Constructor
 * Allocates memory on device for message list length
 * @param a Parent CUDAMessage, used to access message settings, data ptrs etc
 */
MsgArray3D::CUDAModelHandler::CUDAModelHandler(CUDAMessage &a)
    : MsgSpecialisationHandler()
    , d_metadata(nullptr)
    , sim_message(a)
    , d_write_flag(nullptr)
    , d_write_flag_len(0) {
    const Data& d = static_cast<const Data &>(a.getMessageDescription());
    memcpy(&hd_metadata.dimensions, d.dimensions.data(), d.dimensions.size() * sizeof(unsigned int));
    hd_metadata.length = d.dimensions[0] * d.dimensions[1] * d.dimensions[2];
}

void MsgArray3D::CUDAModelHandler::init(CUDAScatter &scatter, const unsigned int &streamId) {
    allocateMetaDataDevicePtr();
    // Allocate messages
    this->sim_message.resize(hd_metadata.length, scatter, streamId);
    this->sim_message.setMessageCount(hd_metadata.length);
    // Zero the output arrays
    auto &read_list = this->sim_message.getReadList();
    auto &write_list = this->sim_message.getWriteList();
    for (auto &var : this->sim_message.getMessageDescription().variables) {
        // Elements is harmless, futureproof for arrays support
        // hd_metadata.length is used, as message array can be longer than message count
        gpuErrchk(cudaMemset(write_list.at(var.first), 0, var.second.type_size * var.second.elements * hd_metadata.length));
        gpuErrchk(cudaMemset(read_list.at(var.first), 0, var.second.type_size * var.second.elements * hd_metadata.length));
    }
}
void MsgArray3D::CUDAModelHandler::allocateMetaDataDevicePtr() {
    if (d_metadata == nullptr) {
        gpuErrchk(cudaMalloc(&d_metadata, sizeof(MetaData)));
        gpuErrchk(cudaMemcpy(d_metadata, &hd_metadata, sizeof(MetaData), cudaMemcpyHostToDevice));
    }
}

void MsgArray3D::CUDAModelHandler::freeMetaDataDevicePtr() {
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
void MsgArray3D::CUDAModelHandler::buildIndex(CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    const unsigned int MESSAGE_COUNT = this->sim_message.getMessageCount();
    // Zero the output arrays
    auto &read_list = this->sim_message.getReadList();
    auto &write_list = this->sim_message.getWriteList();
    for (auto &var : this->sim_message.getMessageDescription().variables) {
        // Elements is harmless, futureproof for arrays support
        // hd_metadata.length is used, as message array can be longer than message count
        gpuErrchk(cudaMemset(write_list.at(var.first), 0, var.second.type_size * var.second.elements * hd_metadata.length));
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
}


MsgArray3D::Data::Data(const std::shared_ptr<const ModelData> &model, const std::string &message_name)
    : MsgBruteForce::Data(model, message_name)
    , dimensions({0, 0, 0}) {
    description = std::unique_ptr<MsgArray3D::Description>(new MsgArray3D::Description(model, this));
    variables.emplace("___INDEX", Variable(1, size_type()));
}
MsgArray3D::Data::Data(const std::shared_ptr<const ModelData> &model, const Data &other)
    : MsgBruteForce::Data(model, other)
    , dimensions(other.dimensions) {
    description = std::unique_ptr<MsgArray3D::Description>(model ? new MsgArray3D::Description(model, this) : nullptr);
    if (dimensions[0] == 0 || dimensions[1] == 0 || dimensions[2] == 0) {
        THROW InvalidMessage("All dimensions must be above zero in array3D message '%s'\n", other.name.c_str());
    }
}
MsgArray3D::Data *MsgArray3D::Data::clone(const std::shared_ptr<const ModelData> &newParent) {
    return new Data(newParent, *this);
}
std::unique_ptr<MsgSpecialisationHandler> MsgArray3D::Data::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MsgSpecialisationHandler>(new CUDAModelHandler(owner));
}
std::type_index MsgArray3D::Data::getType() const { return std::type_index(typeid(MsgArray3D)); }


MsgArray3D::Description::Description(const std::shared_ptr<const ModelData> &_model, Data *const data)
    : MsgBruteForce::Description(_model, data) { }

void MsgArray3D::Description::setDimensions(const size_type& len_x, const size_type& len_y, const size_type& len_z) {
    setDimensions({ len_x , len_y, len_z});
}
void MsgArray3D::Description::setDimensions(const std::array<size_type, 3> &dims) {
    if (dims[0] == 0 || dims[1] == 0 || dims[2] == 0) {
        THROW InvalidArgument("All dimensions must be above zero in array3D message.\n");
    }
    reinterpret_cast<Data *>(message)->dimensions = dims;
}
std::array<MsgArray3D::size_type, 3> MsgArray3D::Description::getDimensions() const {
    return reinterpret_cast<Data *>(message)->dimensions;
}
MsgArray2D::size_type MsgArray3D::Description::getDimX() const {
    return reinterpret_cast<Data *>(message)->dimensions[0];
}
MsgArray2D::size_type MsgArray3D::Description::getDimY() const {
    return reinterpret_cast<Data *>(message)->dimensions[1];
}
MsgArray2D::size_type MsgArray3D::Description::getDimZ() const {
    return reinterpret_cast<Data *>(message)->dimensions[2];
}
