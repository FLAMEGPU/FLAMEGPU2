#include "flamegpu/runtime/messaging/MessageSpatial3D/MessageSpatial3DHost.h"
#include "flamegpu/runtime/messaging/MessageSpatial3D/MessageSpatial3DDevice.cuh"
#include "flamegpu/util/detail/cuda.cuh"
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
MessageSpatial3D::CUDAModelHandler::CUDAModelHandler(CUDAMessage &a)
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

__global__ void atomicHistogram3D(
    const MessageSpatial3D::MetaData *md,
    unsigned int* bin_index,
    unsigned int* bin_sub_index,
    unsigned int *pbm_counts,
    unsigned int message_count,
    const float * __restrict__ x,
    const float * __restrict__ y,
    const float * __restrict__ z) {
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Kill excess threads
    if (index >= message_count) return;

    MessageSpatial3D::GridPos3D gridPos = getGridPosition3D(md, x[index], y[index], z[index]);
    unsigned int hash = getHash3D(md, gridPos);
    bin_index[index] = hash;
    unsigned int bin_idx = atomicInc((unsigned int*)&pbm_counts[hash], 0xFFFFFFFF);
    bin_sub_index[index] = bin_idx;
}

void MessageSpatial3D::CUDAModelHandler::init(CUDAScatter &, unsigned int, cudaStream_t stream) {
    allocateMetaDataDevicePtr(stream);
    // Set PBM to 0
    gpuErrchk(cudaMemsetAsync(hd_data.PBM, 0x00000000, (binCount + 1) * sizeof(unsigned int), stream));
    gpuErrchk(cudaStreamSynchronize(stream));  // This could probably be skipped/delayed safely
}

void MessageSpatial3D::CUDAModelHandler::allocateMetaDataDevicePtr(cudaStream_t stream) {
    if (d_data == nullptr) {
        gpuErrchk(cudaMalloc(&d_histogram, (binCount + 1) * sizeof(unsigned int)));
        gpuErrchk(cudaMalloc(&hd_data.PBM, (binCount + 1) * sizeof(unsigned int)));
        gpuErrchk(cudaMalloc(&d_data, sizeof(MetaData)));
        gpuErrchk(cudaMemcpyAsync(d_data, &hd_data, sizeof(MetaData), cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        resizeCubTemp(stream);
    }
}

void MessageSpatial3D::CUDAModelHandler::freeMetaDataDevicePtr() {
    if (d_data != nullptr) {
        d_CUB_temp_storage_bytes = 0;
        gpuErrchk(flamegpu::util::detail::cuda::cudaFree(d_CUB_temp_storage));
        gpuErrchk(flamegpu::util::detail::cuda::cudaFree(d_histogram));
        gpuErrchk(flamegpu::util::detail::cuda::cudaFree(hd_data.PBM));
        gpuErrchk(flamegpu::util::detail::cuda::cudaFree(d_data));
        d_CUB_temp_storage = nullptr;
        d_histogram = nullptr;
        hd_data.PBM = nullptr;
        d_data = nullptr;
        if (d_keys) {
            d_keys_vals_storage_bytes = 0;
            gpuErrchk(flamegpu::util::detail::cuda::cudaFree(d_keys));
            gpuErrchk(flamegpu::util::detail::cuda::cudaFree(d_vals));
            d_keys = nullptr;
            d_vals = nullptr;
        }
    }
}

void MessageSpatial3D::CUDAModelHandler::buildIndex(CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) {
    NVTX_RANGE("MessageSpatial3D::CUDAModelHandler::buildIndex");
    const unsigned int MESSAGE_COUNT = this->sim_message.getMessageCount();
    resizeKeysVals(this->sim_message.getMaximumListSize());  // Resize based on allocated amount rather than message count
    {  // Build atomic histogram
        gpuErrchk(cudaMemsetAsync(d_histogram, 0x00000000, (binCount + 1) * sizeof(unsigned int), stream));
        int blockSize;  // The launch configurator returned block size
        gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, atomicHistogram3D, 32, 0));  // Randomly 32
                                                                                                         // Round up according to array size
        int gridSize = (MESSAGE_COUNT + blockSize - 1) / blockSize;
        atomicHistogram3D <<<gridSize, blockSize, 0, stream >>>(d_data, d_keys, d_vals, d_histogram, MESSAGE_COUNT,
            reinterpret_cast<float*>(this->sim_message.getReadPtr("x")),
            reinterpret_cast<float*>(this->sim_message.getReadPtr("y")),
            reinterpret_cast<float*>(this->sim_message.getReadPtr("z")));
    }
    {  // Scan (sum), to finalise PBM
        gpuErrchk(cub::DeviceScan::ExclusiveSum(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_histogram, hd_data.PBM, binCount + 1, stream));
    }
    {  // Reorder messages
       // Copy messages from d_messages to d_messages_swap, in hash order
        scatter.pbm_reorder(streamId, stream, this->sim_message.getMessageDescription().variables, this->sim_message.getReadList(), this->sim_message.getWriteList(), MESSAGE_COUNT, d_keys, d_vals, hd_data.PBM);
        this->sim_message.swap();  // Stream id is unused here
        gpuErrchk(cudaStreamSynchronize(stream));  // Not striclty neceesary while pbm_reorder is synchronous.
    }
    {  // Fill PBM and Message Texture Buffers
       // gpuErrchk(cudaBindTexture(nullptr, d_texMessages, d_agents, sizeof(glm::vec4) * MESSAGE_COUNT));
       // gpuErrchk(cudaBindTexture(nullptr, d_texPBM, d_PBM, sizeof(unsigned int) * (binCount + 1)));
    }
}

void MessageSpatial3D::CUDAModelHandler::resizeCubTemp(cudaStream_t stream) {
    size_t bytesCheck = 0;
    gpuErrchk(cub::DeviceScan::ExclusiveSum(nullptr, bytesCheck, hd_data.PBM, d_histogram, binCount + 1, stream));
    if (bytesCheck > d_CUB_temp_storage_bytes) {
        if (d_CUB_temp_storage) {
            gpuErrchk(flamegpu::util::detail::cuda::cudaFree(d_CUB_temp_storage));
        }
        d_CUB_temp_storage_bytes = bytesCheck;
        gpuErrchk(cudaMalloc(&d_CUB_temp_storage, d_CUB_temp_storage_bytes));
    }
}

void MessageSpatial3D::CUDAModelHandler::resizeKeysVals(const unsigned int newSize) {
    size_t bytesCheck = newSize * sizeof(unsigned int);
    if (bytesCheck > d_keys_vals_storage_bytes) {
        if (d_keys) {
            gpuErrchk(flamegpu::util::detail::cuda::cudaFree(d_keys));
            gpuErrchk(flamegpu::util::detail::cuda::cudaFree(d_vals));
        }
        d_keys_vals_storage_bytes = bytesCheck;
        gpuErrchk(cudaMalloc(&d_keys, d_keys_vals_storage_bytes));
        gpuErrchk(cudaMalloc(&d_vals, d_keys_vals_storage_bytes));
    }
}
/// <summary>
/// CDescription
/// </summary>
MessageSpatial3D::CDescription::CDescription(std::shared_ptr<Data> data)
    : MessageSpatial2D::Description(std::move(data)) { }
MessageSpatial3D::CDescription::CDescription(std::shared_ptr<const Data> data)
    : MessageSpatial2D::Description(std::move(std::const_pointer_cast<Data>(data))) { }

bool MessageSpatial3D::CDescription::operator==(const CDescription& rhs) const {
    return *this->message == *rhs.message;  // Compare content is functionally the same
}
bool MessageSpatial3D::CDescription::operator!=(const CDescription& rhs) const {
    return !(*this == rhs);
}
#ifndef __CUDACC__
MessageSpatial3D::CDescription::operator MessageBruteForce::CDescription() const {
    return *this;
}
#endif
/**
 * Const accessors
 */
float MessageSpatial3D::CDescription::getMinZ() const {
    return std::static_pointer_cast<Data>(message)->minZ;
}
float MessageSpatial3D::CDescription::getMaxZ() const {
    return std::static_pointer_cast<Data>(message)->maxZ;
}

/// <summary>
/// Description
/// </summary>
MessageSpatial3D::Description::Description(std::shared_ptr<Data> data)
    : CDescription(data) { }
bool MessageSpatial3D::Description::operator==(const CDescription& rhs) const {
    return rhs == *this;  // Forward to superclass's equality
}
bool MessageSpatial3D::Description::operator!=(const CDescription& rhs) const {
    return !(*this == rhs);
}
/**
 * Accessors
 */
void MessageSpatial3D::Description::setMinZ(const float z) {
    if (!isnan(std::static_pointer_cast<Data>(message)->maxZ) &&
        z >= std::static_pointer_cast<Data>(message)->maxZ) {
        THROW exception::InvalidArgument("Spatial messaging min z bound must be lower than max bound, %f !< %f", z, std::static_pointer_cast<Data>(message)->maxZ);
    }
    std::static_pointer_cast<Data>(message)->minZ = z;
}
void MessageSpatial3D::Description::setMin(const float x, const float y, const float z) {
    if (!isnan(std::static_pointer_cast<Data>(message)->maxX) &&
        x >= std::static_pointer_cast<Data>(message)->maxX) {
        THROW exception::InvalidArgument("Spatial messaging min x bound must be lower than max bound, %f !< %f", x, std::static_pointer_cast<Data>(message)->maxX);
    }
    if (!isnan(std::static_pointer_cast<Data>(message)->maxY) &&
        y >= std::static_pointer_cast<Data>(message)->maxY) {
        THROW exception::InvalidArgument("Spatial messaging min y bound must be lower than max bound, %f !< %f", y, std::static_pointer_cast<Data>(message)->maxY);
    }
    if (!isnan(std::static_pointer_cast<Data>(message)->maxZ) &&
        z >= std::static_pointer_cast<Data>(message)->maxZ) {
        THROW exception::InvalidArgument("Spatial messaging min z bound must be lower than max bound, %f !< %f", z, std::static_pointer_cast<Data>(message)->maxZ);
    }
    std::static_pointer_cast<Data>(message)->minX = x;
    std::static_pointer_cast<Data>(message)->minY = y;
    std::static_pointer_cast<Data>(message)->minZ = z;
}
void MessageSpatial3D::Description::setMaxZ(const float z) {
    if (!isnan(std::static_pointer_cast<Data>(message)->minZ) &&
        z <= std::static_pointer_cast<Data>(message)->minZ) {
        THROW exception::InvalidArgument("Spatial messaging max z bound must be greater than min bound, %f !> %f", z, std::static_pointer_cast<Data>(message)->minZ);
    }
    std::static_pointer_cast<Data>(message)->maxZ = z;
}
void MessageSpatial3D::Description::setMax(const float x, const float y, const float z) {
    if (!isnan(std::static_pointer_cast<Data>(message)->minX) &&
        x <= std::static_pointer_cast<Data>(message)->minX) {
        THROW exception::InvalidArgument("Spatial messaging max x bound must be greater than min bound, %f !> %f", x, std::static_pointer_cast<Data>(message)->minX);
    }
    if (!isnan(std::static_pointer_cast<Data>(message)->minY) &&
        y <= std::static_pointer_cast<Data>(message)->minY) {
        THROW exception::InvalidArgument("Spatial messaging max y bound must be greater than min bound, %f !> %f", y, std::static_pointer_cast<Data>(message)->minY);
    }
    if (!isnan(std::static_pointer_cast<Data>(message)->minZ) &&
        z <= std::static_pointer_cast<Data>(message)->minZ) {
        THROW exception::InvalidArgument("Spatial messaging max z bound must be greater than min bound, %f !> %f", z, std::static_pointer_cast<Data>(message)->minZ);
    }
    std::static_pointer_cast<Data>(message)->maxX = x;
    std::static_pointer_cast<Data>(message)->maxY = y;
    std::static_pointer_cast<Data>(message)->maxZ = z;
}

/// <summary>
/// Data
/// </summary>
MessageSpatial3D::Data::Data(std::shared_ptr<const ModelData> model, const std::string &message_name)
    : MessageSpatial2D::Data(model, message_name)
    , minZ(NAN)
    , maxZ(NAN) {
    // MessageSpatial3D has x/y/z variables by default (x/y are inherited)
    variables.emplace("z", Variable(std::array<typename type_decode<float>::type_t, 1>{}));
}
MessageSpatial3D::Data::Data(std::shared_ptr<const ModelData> model, const Data &other)
    : MessageSpatial2D::Data(model, other)
    , minZ(other.minZ)
    , maxZ(other.maxZ) {
    if (isnan(minZ)) {
        THROW exception::InvalidMessage("Environment minimum z bound has not been set in spatial message '%s'\n", other.name.c_str());
    }
    if (isnan(maxZ)) {
        THROW exception::InvalidMessage("Environment maximum z bound has not been set in spatial message '%s'\n", other.name.c_str());
    }
}
MessageSpatial3D::Data *MessageSpatial3D::Data::clone(const std::shared_ptr<const ModelData> &newParent) {
    return new Data(newParent, *this);
}
std::unique_ptr<MessageSpecialisationHandler> MessageSpatial3D::Data::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MessageSpecialisationHandler>(new CUDAModelHandler(owner));
}
std::type_index MessageSpatial3D::Data::getType() const { return std::type_index(typeid(MessageSpatial3D)); }

flamegpu::MessageSortingType MessageSpatial3D::Data::getSortingType() const {
    return flamegpu::MessageSortingType::spatial3D;
}

}  // namespace flamegpu
