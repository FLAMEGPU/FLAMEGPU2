#include "flamegpu/runtime/messaging/Array.h"
#include "flamegpu/model/AgentDescription.h"  // Used by Move-Assign
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/CUDAScatter.h"

/**
* Sets the array index to store the message in
*/
__device__ void MsgArray::Out::setIndex(const size_type &id) const {
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Todo: checking if the output message type is single or optional?  (d_message_type)

    // set the variable using curve
    Curve::setVariable<size_type>("___INDEX", combined_hash, id, index);

    // Set scan flag incase the message is optional
    this->scan_flag[index] = 1;
}
__device__ MsgArray::In::Filter::Filter(const size_type &_length, const Curve::NamespaceHash &_combined_hash, const size_type &x, const size_type &_radius)
    : radius(_radius)
    , length(_length)
    , combined_hash(_combined_hash) {
    loc = x;
}
__device__ MsgArray::In::Filter::Message& MsgArray::In::Filter::Message::operator++() {
    relative_cell++;
    // Skip origin cell
    if (relative_cell == 0) {
        relative_cell++;
    }
    // Wrap over boundaries
    index_1d = (this->_parent.loc + relative_cell + this->_parent.length) % this->_parent.length;
    return *this;
}
/**
 * Constructor
 * Allocates memory on device for message list length
 * @param a Parent CUDAMessage, used to access message settings, data ptrs etc
 */
MsgArray::CUDAModelHandler::CUDAModelHandler(CUDAMessage &a)
    : MsgSpecialisationHandler()
    , d_metadata(nullptr)
    , sim_message(a)
    , d_write_flag(nullptr)
    , d_write_flag_len(0) {
    const Data& d = static_cast<const Data &>(a.getMessageDescription());
    hd_metadata.length = d.length;
}

void MsgArray::CUDAModelHandler::allocateMetaDataDevicePtr() {
    if (d_metadata == nullptr) {
        gpuErrchk(cudaMalloc(&d_metadata, sizeof(MetaData)));
        gpuErrchk(cudaMemcpy(d_metadata, &hd_metadata, sizeof(MetaData), cudaMemcpyHostToDevice));
    }
}

void MsgArray::CUDAModelHandler::freeMetaDataDevicePtr() {
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
void MsgArray::CUDAModelHandler::buildIndex(CUDAScatter &scatter, const unsigned int &streamId) {
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
    scatter.arrayMessageReorder(streamId, this->sim_message.getMessageDescription().variables, read_list, write_list, MESSAGE_COUNT, hd_metadata.length, t_d_write_flag);
    this->sim_message.swap();
    // Reset message count back to full array length
    // Array message exposes not output messages as 0
    if (MESSAGE_COUNT != hd_metadata.length)
        this->sim_message.setMessageCount(hd_metadata.length);
    // Detect errors
    // TODO
}


MsgArray::Data::Data(const std::shared_ptr<const ModelData>&model, const std::string &message_name)
    : MsgBruteForce::Data(model, message_name)
    , length(0) {
    description = std::unique_ptr<MsgArray::Description>(new MsgArray::Description(model, this));
    variables.emplace("___INDEX", Variable(1, size_type()));
}
MsgArray::Data::Data(const std::shared_ptr<const ModelData>&model, const Data &other)
    : MsgBruteForce::Data(model, other)
    , length(other.length) {
    description = std::unique_ptr<MsgArray::Description>(model ? new MsgArray::Description(model, this) : nullptr);
    if (length == 0) {
        THROW InvalidMessage("Length must not be zero in array message '%s'\n", other.name.c_str());
    }
}
MsgArray::Data *MsgArray::Data::clone(const std::shared_ptr<const ModelData> &newParent) {
    return new Data(newParent, *this);
}
std::unique_ptr<MsgSpecialisationHandler> MsgArray::Data::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MsgSpecialisationHandler>(new CUDAModelHandler(owner));
}
std::type_index MsgArray::Data::getType() const { return std::type_index(typeid(MsgArray)); }


MsgArray::Description::Description(const std::shared_ptr<const ModelData>&_model, Data *const data)
    : MsgBruteForce::Description(_model, data) { }

void MsgArray::Description::setLength(const size_type &len) {
    if (len == 0) {
        THROW InvalidArgument("Array messaging length must not be zero.\n");
    }
    reinterpret_cast<Data *>(message)->length = len;
}
MsgArray::size_type MsgArray::Description::getLength() const {
    return reinterpret_cast<Data *>(message)->length;
}
