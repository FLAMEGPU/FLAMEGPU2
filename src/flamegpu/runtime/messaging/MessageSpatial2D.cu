#include "flamegpu/runtime/messaging/MessageSpatial2D.h"

#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable : 4706 4834)
#include <cub/cub.cuh>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#endif

#include "flamegpu/runtime/messaging.h"
#include "flamegpu/runtime/messaging/MessageSpatial2D/MessageSpatial2DHost.h"
#include "flamegpu/runtime/messaging/MessageSpatial2D/MessageSpatial2DDevice.cuh"
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/CUDAScatter.cuh"
#include "flamegpu/util/nvtx.h"
#include "flamegpu/util/detail/cuda.cuh"

namespace flamegpu {
MessageSpatial2D::CUDAModelHandler::CUDAModelHandler(CUDAMessage &a)
    : MessageSpecialisationHandler()
    , sim_message(a) {
    flamegpu::util::nvtx::Range range{"MessageSpatial2D::CUDAModelHandler::CUDAModelHandler"};
    const Data &d = (const Data &)a.getMessageData();
    hd_data.radius = d.radius;
    hd_data.min[0] = d.minX;
    hd_data.min[1] = d.minY;
    hd_data.max[0] = d.maxX;
    hd_data.max[1] = d.maxY;
    binCount = 1;
    for (unsigned int axis = 0; axis < 2; ++axis) {
        hd_data.environmentWidth[axis] = hd_data.max[axis] - hd_data.min[axis];
        hd_data.gridDim[axis] = static_cast<unsigned int>(ceil(hd_data.environmentWidth[axis] / hd_data.radius));
        binCount *= hd_data.gridDim[axis];
    }
}
MessageSpatial2D::CUDAModelHandler::~CUDAModelHandler() { }
__global__ void atomicHistogram2D(
    const MessageSpatial2D::MetaData *md,
    unsigned int* bin_index,
    unsigned int* bin_sub_index,
    unsigned int *pbm_counts,
    unsigned int message_count,
    const float * __restrict__ x,
    const float * __restrict__ y) {
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Kill excess threads
    if (index >= message_count) return;

    MessageSpatial2D::GridPos2D gridPos = getGridPosition2D(md, x[index], y[index]);
    unsigned int hash = getHash2D(md, gridPos);
    bin_index[index] = hash;
    unsigned int bin_idx = atomicInc((unsigned int*)&pbm_counts[hash], 0xFFFFFFFF);
    bin_sub_index[index] = bin_idx;
}

void MessageSpatial2D::CUDAModelHandler::init(CUDAScatter &, unsigned int, cudaStream_t stream) {
    allocateMetaDataDevicePtr(stream);
    // Set PBM to 0
    gpuErrchk(cudaMemsetAsync(hd_data.PBM, 0x00000000, (binCount + 1) * sizeof(unsigned int), stream));
    gpuErrchk(cudaStreamSynchronize(stream));  // This could probably be skipped/delayed safely
}

void MessageSpatial2D::CUDAModelHandler::allocateMetaDataDevicePtr(cudaStream_t stream) {
    if (d_data == nullptr) {
        gpuErrchk(cudaMalloc(&d_histogram, (binCount + 1) * sizeof(unsigned int)));
        gpuErrchk(cudaMalloc(&hd_data.PBM, (binCount + 1) * sizeof(unsigned int)));
        gpuErrchk(cudaMalloc(&d_data, sizeof(MetaData)));
        gpuErrchk(cudaMemcpyAsync(d_data, &hd_data, sizeof(MetaData), cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        resizeCubTemp(stream);
    }
}

void MessageSpatial2D::CUDAModelHandler::freeMetaDataDevicePtr() {
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

void MessageSpatial2D::CUDAModelHandler::buildIndex(CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) {
    flamegpu::util::nvtx::Range range{"MessageSpatial2D::CUDAModelHandler::buildIndex"};
    const unsigned int MESSAGE_COUNT = this->sim_message.getMessageCount();
    resizeKeysVals(this->sim_message.getMaximumListSize());  // Resize based on allocated amount rather than message count
    {  // Build atomic histogram
        gpuErrchk(cudaMemsetAsync(d_histogram, 0x00000000, (binCount + 1) * sizeof(unsigned int), stream));
        int blockSize;  // The launch configurator returned block size
        gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, atomicHistogram2D, 32, 0));  // Randomly 32
                                                                                                         // Round up according to array size
        int gridSize = (MESSAGE_COUNT + blockSize - 1) / blockSize;
        atomicHistogram2D <<<gridSize, blockSize, 0, stream >>>(d_data, d_keys, d_vals, d_histogram, MESSAGE_COUNT,
            reinterpret_cast<float*>(this->sim_message.getReadPtr("x")),
            reinterpret_cast<float*>(this->sim_message.getReadPtr("y")));
    }
    {  // Scan (sum), to finalise PBM
        gpuErrchk(cub::DeviceScan::ExclusiveSum(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_histogram, hd_data.PBM, binCount + 1, stream));
    }
    {  // Reorder messages
       // Copy messages from d_messages to d_messages_swap, in hash order
        scatter.pbm_reorder(streamId, stream, this->sim_message.getMessageData().variables, this->sim_message.getReadList(), this->sim_message.getWriteList(), MESSAGE_COUNT, d_keys, d_vals, hd_data.PBM);
        this->sim_message.swap();
        gpuErrchk(cudaStreamSynchronize(stream));  // Not striclty neceesary while pbm_reorder is synchronous.
    }
    {  // Fill PBM and Message Texture Buffers
       // gpuErrchk(cudaBindTexture(nullptr, d_texMessages, d_agents, sizeof(glm::vec4) * MESSAGE_COUNT));
       // gpuErrchk(cudaBindTexture(nullptr, d_texPBM, d_PBM, sizeof(unsigned int) * (binCount + 1)));
    }
}

void MessageSpatial2D::CUDAModelHandler::resizeCubTemp(cudaStream_t stream) {
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

void MessageSpatial2D::CUDAModelHandler::resizeKeysVals(const unsigned int newSize) {
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
MessageSpatial2D::CDescription::CDescription(std::shared_ptr<Data> data)
    : MessageBruteForce::CDescription(std::move(std::static_pointer_cast<MessageBruteForce::Data>(data))) { }
MessageSpatial2D::CDescription::CDescription(std::shared_ptr<const Data> data)
    : CDescription(std::move(std::const_pointer_cast<Data>(data))) { }

bool MessageSpatial2D::CDescription::operator==(const CDescription& rhs) const {
    return *this->message == *rhs.message;  // Compare content is functionally the same
}
bool MessageSpatial2D::CDescription::operator!=(const CDescription& rhs) const {
    return !(*this == rhs);
}
/**
 * Const accessors
 */
float MessageSpatial2D::CDescription::getRadius() const {
    return std::static_pointer_cast<Data>(message)->radius;
}
float MessageSpatial2D::CDescription::getMinX() const {
    return std::static_pointer_cast<Data>(message)->minX;
}
float MessageSpatial2D::CDescription::getMinY() const {
    return std::static_pointer_cast<Data>(message)->minY;
}
float MessageSpatial2D::CDescription::getMaxX() const {
    return std::static_pointer_cast<Data>(message)->maxX;
}
float MessageSpatial2D::CDescription::getMaxY() const {
    return std::static_pointer_cast<Data>(message)->maxY;
}

/// <summary>
/// Description
/// </summary>
MessageSpatial2D::Description::Description(std::shared_ptr<Data> data)
    : CDescription(data) { }
/**
 * Accessors
 */
void MessageSpatial2D::CDescription::setRadius(const float r) {
    if (r <= 0) {
        THROW exception::InvalidArgument("Spatial messaging radius must be a positive value, %f is not valid.", r);
    }
    std::static_pointer_cast<Data>(message)->radius = r;
}
void MessageSpatial2D::CDescription::setMinX(const float x) {
    if (!isnan(std::static_pointer_cast<Data>(message)->maxX) &&
        x >= std::static_pointer_cast<Data>(message)->maxX) {
        THROW exception::InvalidArgument("Spatial messaging minimum bound must be lower than max bound, %f !< %f.", x, std::static_pointer_cast<Data>(message)->maxX);
    }
    std::static_pointer_cast<Data>(message)->minX = x;
}
void MessageSpatial2D::CDescription::setMinY(const float y) {
    if (!isnan(std::static_pointer_cast<Data>(message)->maxY) &&
        y >= std::static_pointer_cast<Data>(message)->maxY) {
        THROW exception::InvalidArgument("Spatial messaging minimum bound must be lower than max bound, %f !< %f.", y, std::static_pointer_cast<Data>(message)->maxY);
    }
    std::static_pointer_cast<Data>(message)->minY = y;
}
void MessageSpatial2D::CDescription::setMin(const float x, const float y) {
    if (!isnan(std::static_pointer_cast<Data>(message)->maxX) &&
        x >= std::static_pointer_cast<Data>(message)->maxX) {
        THROW exception::InvalidArgument("Spatial messaging minimum bound must be lower than max bound, %f !< %f.", x, std::static_pointer_cast<Data>(message)->maxX);
    }
    if (!isnan(std::static_pointer_cast<Data>(message)->maxY) &&
        y >= std::static_pointer_cast<Data>(message)->maxY) {
        THROW exception::InvalidArgument("Spatial messaging minimum bound must be lower than max bound, %f !< %f.", y, std::static_pointer_cast<Data>(message)->maxY);
    }
    std::static_pointer_cast<Data>(message)->minX = x;
    std::static_pointer_cast<Data>(message)->minY = y;
}
void MessageSpatial2D::CDescription::setMaxX(const float x) {
    if (!isnan(std::static_pointer_cast<Data>(message)->minX) &&
        x <= std::static_pointer_cast<Data>(message)->minX) {
        THROW exception::InvalidArgument("Spatial messaging max x bound must be greater than min bound, %f !> %f.", x, std::static_pointer_cast<Data>(message)->minX);
    }
    std::static_pointer_cast<Data>(message)->maxX = x;
}
void MessageSpatial2D::CDescription::setMaxY(const float y) {
    if (!isnan(std::static_pointer_cast<Data>(message)->minY) &&
        y <= std::static_pointer_cast<Data>(message)->minY) {
        THROW exception::InvalidArgument("Spatial messaging max y bound must be greater than min bound, %f !> %f.", y, std::static_pointer_cast<Data>(message)->minY);
    }
    std::static_pointer_cast<Data>(message)->maxY = y;
}
void MessageSpatial2D::CDescription::setMax(const float x, const float y) {
    if (!isnan(std::static_pointer_cast<Data>(message)->minX) &&
        x <= std::static_pointer_cast<Data>(message)->minX) {
        THROW exception::InvalidArgument("Spatial messaging max x bound must be greater than min bound, %f !> %f.", x, std::static_pointer_cast<Data>(message)->minX);
    }
    if (!isnan(std::static_pointer_cast<Data>(message)->minY) &&
        y <= std::static_pointer_cast<Data>(message)->minY) {
        THROW exception::InvalidArgument("Spatial messaging max y bound must be greater than min bound, %f !> %f.", y, std::static_pointer_cast<Data>(message)->minY);
    }
    std::static_pointer_cast<Data>(message)->maxX = x;
    std::static_pointer_cast<Data>(message)->maxY = y;
}

/// <summary>
/// Data
/// </summary>
MessageSpatial2D::Data::Data(std::shared_ptr<const ModelData> model, const std::string &message_name)
    : MessageBruteForce::Data(model, message_name)
    , radius(NAN)
    , minX(NAN)
    , minY(NAN)
    , maxX(NAN)
    , maxY(NAN) {
    // MessageSpatial2D has x/y variables by default
    variables.emplace("x", Variable(std::array<typename type_decode<float>::type_t, 1>{}));
    variables.emplace("y", Variable(std::array<typename type_decode<float>::type_t, 1>{}));
}
MessageSpatial2D::Data::Data(std::shared_ptr<const ModelData> model, const Data &other)
    : MessageBruteForce::Data(model, other)
    , radius(other.radius)
    , minX(other.minX)
    , minY(other.minY)
    , maxX(other.maxX)
    , maxY(other.maxY) {
    if (isnan(radius)) {
        THROW exception::InvalidMessage("Radius has not been set in spatial message '%s'.", other.name.c_str());
    }
    if (isnan(minX)) {
        THROW exception::InvalidMessage("Environment minimum x bound has not been set in spatial message '%s.", other.name.c_str());
    }
    if (isnan(minY)) {
        THROW exception::InvalidMessage("Environment minimum y bound has not been set in spatial message '%s'.", other.name.c_str());
    }
    if (isnan(maxX)) {
        THROW exception::InvalidMessage("Environment maximum x bound has not been set in spatial message '%s'.", other.name.c_str());
    }
    if (isnan(maxY)) {
        THROW exception::InvalidMessage("Environment maximum y bound has not been set in spatial message '%s'.", other.name.c_str());
    }
}
MessageSpatial2D::Data *MessageSpatial2D::Data::clone(const std::shared_ptr<const ModelData> &newParent) {
    return new Data(newParent, *this);
}
std::unique_ptr<MessageSpecialisationHandler> MessageSpatial2D::Data::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MessageSpecialisationHandler>(new CUDAModelHandler(owner));
}
std::type_index MessageSpatial2D::Data::getType() const { return std::type_index(typeid(MessageSpatial2D)); }

flamegpu::MessageSortingType MessageSpatial2D::Data::getSortingType() const {
    return flamegpu::MessageSortingType::spatial2D;
}

}  // namespace flamegpu
