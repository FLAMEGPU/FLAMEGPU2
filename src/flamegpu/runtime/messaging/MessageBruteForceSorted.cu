#include "flamegpu/runtime/messaging/MessageBruteForceSorted/MessageBruteForceSortedHost.h"
#include "flamegpu/runtime/messaging/MessageBruteForceSorted/MessageBruteForceSortedDevice.cuh"

#include "flamegpu/gpu/CUDAScatter.cuh"
#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable : 4706 4834)
#include <cub/cub.cuh>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#endif


namespace flamegpu {


MessageBruteForceSorted::CUDAModelHandler::CUDAModelHandler(CUDAMessage &a)
  : MessageSpecialisationHandler()
  , sim_message(a) {
    NVTX_RANGE("Spatial3D::CUDAModelHandler");
    const Data &d = (const Data &)a.getMessageDescription();
    hd_data.radius = d.radius;
    hd_data.min[0] = d.minX;
    hd_data.min[1] = d.minY;
    hd_data.min[2] = d.minZ;
    hd_data.max[0] = d.maxX;
    hd_data.max[1] = d.maxY;
    hd_data.max[2] = d.maxZ;
    binCount = 1;
    for (unsigned int axis = 0; axis < 3; ++axis) {
        hd_data.environmentWidth[axis] = hd_data.max[axis] - hd_data.min[axis];
        hd_data.gridDim[axis] = static_cast<unsigned int>(ceil(hd_data.environmentWidth[axis] / hd_data.radius));
        binCount *= hd_data.gridDim[axis];
    }
    // Device allocation occurs in allocateMetaDataDevicePtr rather than the constructor.
}

__global__ void computeSpatialHash(
    const MessageBruteForceSorted::MetaData *md,
    unsigned int* bin_index,
    unsigned int* tid,
    unsigned int message_count,
    const float * __restrict__ x,
    const float * __restrict__ y,
    const float * __restrict__ z) {
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Kill excess threads
    if (index >= message_count) return;

    MessageBruteForceSorted::GridPos3D gridPos = getGridPosition3D(md, x[index], y[index], z[index]);
    unsigned int hash = getHash3D(md, gridPos);
    bin_index[index] = hash;
    tid[index] = index;
}

void MessageBruteForceSorted::CUDAModelHandler::init(CUDAScatter &, const unsigned int &) {
    allocateMetaDataDevicePtr();
    // Allocate messages
    hd_data.length = 0;  // This value should already be 0
    gpuErrchk(cudaMemcpy(d_data, &hd_data, sizeof(MetaData), cudaMemcpyHostToDevice));
}

void MessageBruteForceSorted::CUDAModelHandler::allocateMetaDataDevicePtr() {
    if (d_data == nullptr) {
        gpuErrchk(cudaMalloc(&d_data, sizeof(MetaData)));
        gpuErrchk(cudaMemcpy(d_data, &hd_data, sizeof(MetaData), cudaMemcpyHostToDevice));
        resizeCubTemp();
    }
}

void MessageBruteForceSorted::CUDAModelHandler::freeMetaDataDevicePtr() {
    if (d_data != nullptr) {
        d_CUB_temp_storage_bytes = 0;
        gpuErrchk(cudaFree(d_CUB_temp_storage));
        gpuErrchk(cudaFree(d_data));
        d_CUB_temp_storage = nullptr;
        d_data = nullptr;
        if (d_keys) {
            d_keys_vals_storage_bytes = 0;
            gpuErrchk(cudaFree(d_keys));
            gpuErrchk(cudaFree(d_vals));
            gpuErrchk(cudaFree(d_keys_out));
            gpuErrchk(cudaFree(d_vals_out));
            d_keys = nullptr;
            d_vals = nullptr;
            d_keys_out = nullptr;
            d_vals_out = nullptr;
        }
    }
}

void MessageBruteForceSorted::CUDAModelHandler::buildIndex(CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    NVTX_RANGE("MessageBruteForceSorted::CUDAModelHandler::buildIndex");
    
    unsigned int MESSAGE_COUNT = this->sim_message.getMessageCount();
    if (MESSAGE_COUNT != hd_data.length) {
        hd_data.length = MESSAGE_COUNT;
        gpuErrchk(cudaMemcpy(d_data, &hd_data, sizeof(MetaData), cudaMemcpyHostToDevice));
        
        // Resize temp storage
    }
    resizeKeysVals(this->sim_message.getMaximumListSize());  // Resize based on allocated amount rather than message count
    
    { 
        int blockSize;  // The launch configurator returned block size
        gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, computeSpatialHash, 32, 0));  // Randomly 32
                                                                                                         // Round up according to array size
        int gridSize = (MESSAGE_COUNT + blockSize - 1) / blockSize;
        computeSpatialHash <<<gridSize, blockSize, 0, stream >>>(d_data, d_keys, d_vals, MESSAGE_COUNT,
            reinterpret_cast<float*>(this->sim_message.getReadPtr("x")),
            reinterpret_cast<float*>(this->sim_message.getReadPtr("y")),
            reinterpret_cast<float*>(this->sim_message.getReadPtr("z")));
    }
    {  // Reorder messages

        // d_keys currently contains hashes, d_vals contains tid 1..message_count
        // Perform pair sort on hashes to get sorted message order
        gpuErrchk(cub::DeviceRadixSort::SortPairs(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_keys, d_keys_out, d_vals, d_vals_out, MESSAGE_COUNT));
       
        // Reorder actual messages
        // Copy messages from d_messages to d_messages_swap, in hash order
        //scatter.pbm_reorder(streamId, stream, this->sim_message.getMessageDescription().variables, this->sim_message.getReadList(), this->sim_message.getWriteList(), MESSAGE_COUNT, d_keys, d_vals, hd_data.PBM);
        //this->sim_message.swap();  // Stream id is unused here
        //gpuErrchk(cudaStreamSynchronize(stream));  // Not striclty neceesary while pbm_reorder is synchronous.
    }
}

void MessageBruteForceSorted::CUDAModelHandler::resizeCubTemp() {
}

void MessageBruteForceSorted::CUDAModelHandler::resizeKeysVals(const unsigned int &newSize) {
    size_t bytesCheck = newSize * sizeof(unsigned int);
    if (bytesCheck > d_keys_vals_storage_bytes) {
        if (d_keys) {
            gpuErrchk(cudaFree(d_keys));
            gpuErrchk(cudaFree(d_vals));
            gpuErrchk(cudaFree(d_keys_out));
            gpuErrchk(cudaFree(d_vals_out));
        }
        d_keys_vals_storage_bytes = bytesCheck;
        gpuErrchk(cudaMalloc(&d_keys, d_keys_vals_storage_bytes));
        gpuErrchk(cudaMalloc(&d_vals, d_keys_vals_storage_bytes));
        gpuErrchk(cudaMalloc(&d_keys_out, d_keys_vals_storage_bytes));
        gpuErrchk(cudaMalloc(&d_vals_out, d_keys_vals_storage_bytes));
        gpuErrchk(cub::DeviceRadixSort::SortPairs(nullptr, d_CUB_temp_storage_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, newSize));
    }
}

MessageBruteForceSorted::Data::Data(const std::shared_ptr<const ModelData> &model, const std::string &message_name)
    : MessageSpatial2D::Data(model, message_name)
    , minZ(NAN)
    , maxZ(NAN) {
    description = std::unique_ptr<Description>(new Description(model, this));
    description->newVariable<float>("z");
}
MessageBruteForceSorted::Data::Data(const std::shared_ptr<const ModelData> &model, const Data &other)
    : MessageSpatial2D::Data(model, other)
    , minZ(other.minZ)
    , maxZ(other.maxZ) {
    description = std::unique_ptr<Description>(model ? new Description(model, this) : nullptr);
    if (isnan(minZ)) {
        THROW exception::InvalidMessage("Environment minimum z bound has not been set in spatial message '%s'\n", other.name.c_str());
    }
    if (isnan(maxZ)) {
        THROW exception::InvalidMessage("Environment maximum z bound has not been set in spatial message '%s'\n", other.name.c_str());
    }
}
MessageBruteForceSorted::Data *MessageBruteForceSorted::Data::clone(const std::shared_ptr<const ModelData> &newParent) {
    return new Data(newParent, *this);
}
std::unique_ptr<MessageSpecialisationHandler> MessageBruteForceSorted::Data::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MessageSpecialisationHandler>(new CUDAModelHandler(owner));
}
std::type_index MessageBruteForceSorted::Data::getType() const { return std::type_index(typeid(MessageBruteForceSorted)); }

MessageBruteForceSorted::Description::Description(const std::shared_ptr<const ModelData> &_model, Data *const data)
    : MessageBruteForce::Description(_model, data) { }

void MessageBruteForceSorted::Description::setRadius(const float &r) {
    if (r <= 0) {
        THROW exception::InvalidArgument("Spatial messaging radius must be a positive value, %f is not valid.", r);
    }
    reinterpret_cast<Data *>(message)->radius = r;
}
void MessageBruteForceSorted::Description::setMinX(const float &x) {
    if (!isnan(reinterpret_cast<Data *>(message)->maxX) &&
        x >= reinterpret_cast<Data *>(message)->maxX) {
        THROW exception::InvalidArgument("Spatial messaging min x bound must be lower than max bound, %f !< %f", x, reinterpret_cast<Data *>(message)->maxX);
    }
    reinterpret_cast<Data *>(message)->minX = x;
}
void MessageBruteForceSorted::Description::setMinY(const float &y) {
    if (!isnan(reinterpret_cast<Data *>(message)->maxY) &&
        y >= reinterpret_cast<Data *>(message)->maxY) {
        THROW exception::InvalidArgument("Spatial messaging min bound must be lower than max bound, %f !< %f", y, reinterpret_cast<Data *>(message)->maxY);
    }
    reinterpret_cast<Data *>(message)->minY = y;
}
void MessageBruteForceSorted::Description::setMinZ(const float &z) {
    if (!isnan(reinterpret_cast<Data *>(message)->maxZ) &&
        z >= reinterpret_cast<Data *>(message)->maxZ) {
        THROW exception::InvalidArgument("Spatial messaging min z bound must be lower than max bound, %f !< %f", z, reinterpret_cast<Data *>(message)->maxZ);
    }
    reinterpret_cast<Data *>(message)->minZ = z;
}
void MessageBruteForceSorted::Description::setMin(const float &x, const float &y, const float &z) {
    if (!isnan(reinterpret_cast<Data *>(message)->maxX) &&
        x >= reinterpret_cast<Data *>(message)->maxX) {
        THROW exception::InvalidArgument("Spatial messaging min x bound must be lower than max bound, %f !< %f", x, reinterpret_cast<Data *>(message)->maxX);
    }
    if (!isnan(reinterpret_cast<Data *>(message)->maxY) &&
        y >= reinterpret_cast<Data *>(message)->maxY) {
        THROW exception::InvalidArgument("Spatial messaging min y bound must be lower than max bound, %f !< %f", y, reinterpret_cast<Data *>(message)->maxY);
    }
    if (!isnan(reinterpret_cast<Data *>(message)->maxZ) &&
        z >= reinterpret_cast<Data *>(message)->maxZ) {
        THROW exception::InvalidArgument("Spatial messaging min z bound must be lower than max bound, %f !< %f", z, reinterpret_cast<Data *>(message)->maxZ);
    }
    reinterpret_cast<Data *>(message)->minX = x;
    reinterpret_cast<Data *>(message)->minY = y;
    reinterpret_cast<Data *>(message)->minZ = z;
}
void MessageBruteForceSorted::Description::setMaxX(const float &x) {
    if (!isnan(reinterpret_cast<Data *>(message)->minX) &&
        x <= reinterpret_cast<Data *>(message)->minX) {
        THROW exception::InvalidArgument("Spatial messaging max x bound must be greater than min bound, %f !> %f", x, reinterpret_cast<Data *>(message)->minX);
    }
    reinterpret_cast<Data *>(message)->maxX = x;
}
void MessageBruteForceSorted::Description::setMaxY(const float &y) {
    if (!isnan(reinterpret_cast<Data *>(message)->minY) &&
        y <= reinterpret_cast<Data *>(message)->minY) {
        THROW exception::InvalidArgument("Spatial messaging max y bound must be greater than min bound, %f !> %f", y, reinterpret_cast<Data *>(message)->minY);
    }
    reinterpret_cast<Data *>(message)->maxY = y;
}
void MessageBruteForceSorted::Description::setMaxZ(const float &z) {
    if (!isnan(reinterpret_cast<Data *>(message)->minZ) &&
        z <= reinterpret_cast<Data *>(message)->minZ) {
        THROW exception::InvalidArgument("Spatial messaging max z bound must be greater than min bound, %f !> %f", z, reinterpret_cast<Data *>(message)->minZ);
    }
    reinterpret_cast<Data *>(message)->maxZ = z;
}
void MessageBruteForceSorted::Description::setMax(const float &x, const float &y, const float &z) {
    if (!isnan(reinterpret_cast<Data *>(message)->minX) &&
        x <= reinterpret_cast<Data *>(message)->minX) {
        THROW exception::InvalidArgument("Spatial messaging max x bound must be greater than min bound, %f !> %f", x, reinterpret_cast<Data *>(message)->minX);
    }
    if (!isnan(reinterpret_cast<Data *>(message)->minY) &&
        y <= reinterpret_cast<Data *>(message)->minY) {
        THROW exception::InvalidArgument("Spatial messaging max y bound must be greater than min bound, %f !> %f", y, reinterpret_cast<Data *>(message)->minY);
    }
    if (!isnan(reinterpret_cast<Data *>(message)->minZ) &&
        z <= reinterpret_cast<Data *>(message)->minZ) {
        THROW exception::InvalidArgument("Spatial messaging max z bound must be greater than min bound, %f !> %f", z, reinterpret_cast<Data *>(message)->minZ);
    }
    reinterpret_cast<Data *>(message)->maxX = x;
    reinterpret_cast<Data *>(message)->maxY = y;
    reinterpret_cast<Data *>(message)->maxZ = z;
}

float MessageBruteForceSorted::Description::getRadius() const {
    return reinterpret_cast<Data *>(message)->radius;
}
float MessageBruteForceSorted::Description::getMinX() const {
    return reinterpret_cast<Data *>(message)->minX;
}
float MessageBruteForceSorted::Description::getMinY() const {
    return reinterpret_cast<Data *>(message)->minY;
}
float MessageBruteForceSorted::Description::getMinZ() const {
    return reinterpret_cast<Data *>(message)->minZ;
}
float MessageBruteForceSorted::Description::getMaxX() const {
    return reinterpret_cast<Data *>(message)->maxX;
}
float MessageBruteForceSorted::Description::getMaxY() const {
    return reinterpret_cast<Data *>(message)->maxY;
}
float MessageBruteForceSorted::Description::getMaxZ() const {
    return reinterpret_cast<Data *>(message)->maxZ;
}

}  // namespace flamegpu
