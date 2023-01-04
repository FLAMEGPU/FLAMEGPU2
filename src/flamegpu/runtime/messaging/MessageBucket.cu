#include "flamegpu/runtime/messaging/MessageBucket.h"

#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable : 4706 4834)
#endif  // _MSC_VER
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress 1719
#else
#pragma diag_suppress 1719
#endif  // __NVCC_DIAG_PRAGMA_SUPPORT__
#include <cub/cub.cuh>
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_default 1719
#else
#pragma diag_default 1719
#endif  // __NVCC_DIAG_PRAGMA_SUPPORT__
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/simulation/detail/CUDAMessage.h"
#include "flamegpu/simulation/detail/CUDAScatter.cuh"
#include "flamegpu/util/nvtx.h"

#include "flamegpu/runtime/messaging/MessageBucket/MessageBucketHost.h"
// #include "flamegpu/runtime/messaging/MessageBucket/MessageBucketDevice.cuh"
#include "flamegpu/detail/cuda.cuh"

namespace flamegpu {
MessageBucket::CUDAModelHandler::CUDAModelHandler(detail::CUDAMessage &a)
    : MessageSpecialisationHandler()
    , sim_message(a) {
    flamegpu::util::nvtx::Range range{"MessageBucket::CUDAModelHandler::CUDAModelHandler"};
    const Data &d = (const Data &)a.getMessageData();
    hd_data.min = d.lowerBound;
    // Here we convert it so that upperBound is one greater than the final valid index
    hd_data.max = d.upperBound + 1;
    bucketCount = d.upperBound - d.lowerBound  + 1;
}
MessageBucket::CUDAModelHandler::~CUDAModelHandler() { }

__global__ void atomicHistogram1D(
    const MessageBucket::MetaData *md,
    unsigned int* bin_index,
    unsigned int* bin_sub_index,
    unsigned int *pbm_counts,
    unsigned int message_count,
    const IntT * __restrict__ key) {
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Kill excess threads
    if (index >= message_count) return;

    const unsigned int hash = key[index] - md->min;
    bin_index[index] = hash;
    unsigned int bin_idx = atomicInc((unsigned int*)&pbm_counts[hash], 0xFFFFFFFF);
    bin_sub_index[index] = bin_idx;
}

void MessageBucket::CUDAModelHandler::init(detail::CUDAScatter &, unsigned int, cudaStream_t stream) {
    allocateMetaDataDevicePtr(stream);
    // Set PBM to 0
    gpuErrchk(cudaMemsetAsync(hd_data.PBM, 0x00000000, (bucketCount + 1) * sizeof(unsigned int), stream));
    gpuErrchk(cudaStreamSynchronize(stream));
}

void MessageBucket::CUDAModelHandler::allocateMetaDataDevicePtr(cudaStream_t stream) {
    if (d_data == nullptr) {
        gpuErrchk(cudaMalloc(&d_histogram, (bucketCount + 1) * sizeof(unsigned int)));
        gpuErrchk(cudaMalloc(&hd_data.PBM, (bucketCount + 1) * sizeof(unsigned int)));
        gpuErrchk(cudaMalloc(&d_data, sizeof(MetaData)));
        gpuErrchk(cudaMemcpyAsync(d_data, &hd_data, sizeof(MetaData), cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        resizeCubTemp();
    }
}

void MessageBucket::CUDAModelHandler::freeMetaDataDevicePtr() {
    if (d_data != nullptr) {
        d_CUB_temp_storage_bytes = 0;
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_CUB_temp_storage));
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_histogram));
        gpuErrchk(flamegpu::detail::cuda::cudaFree(hd_data.PBM));
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_data));
        d_CUB_temp_storage = nullptr;
        d_histogram = nullptr;
        hd_data.PBM = nullptr;
        d_data = nullptr;
        if (d_keys) {
            d_keys_vals_storage_bytes = 0;
            gpuErrchk(flamegpu::detail::cuda::cudaFree(d_keys));
            gpuErrchk(flamegpu::detail::cuda::cudaFree(d_vals));
            d_keys = nullptr;
            d_vals = nullptr;
        }
    }
}

void MessageBucket::CUDAModelHandler::buildIndex(detail::CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream) {
    flamegpu::util::nvtx::Range range{"MessageBucket::CUDAModelHandler::buildIndex"};
    // Cuda operations all occur within the stream, so only a final sync is required.s
    const unsigned int MESSAGE_COUNT = this->sim_message.getMessageCount();
    resizeKeysVals(this->sim_message.getMaximumListSize());  // Resize based on allocated amount rather than message count
    {  // Build atomic histogram
        gpuErrchk(cudaMemsetAsync(d_histogram, 0x00000000, (bucketCount + 1) * sizeof(unsigned int), stream));
        int blockSize;  // The launch configurator returned block size
        gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, atomicHistogram1D, 32, 0));  // Randomly 32
                                                                                                         // Round up according to array size
        int gridSize = (MESSAGE_COUNT + blockSize - 1) / blockSize;
        atomicHistogram1D <<<gridSize, blockSize, 0, stream >>>(d_data, d_keys, d_vals, d_histogram, MESSAGE_COUNT,
            reinterpret_cast<IntT*>(this->sim_message.getReadPtr("_key")));
    }
    {  // Scan (sum), to finalise PBM
        gpuErrchk(cub::DeviceScan::ExclusiveSum(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_histogram, hd_data.PBM, bucketCount + 1, stream));
    }
    {  // Reorder messages
       // Copy messages from d_messages to d_messages_swap, in hash order
        scatter.pbm_reorder(streamId, stream, this->sim_message.getMessageData().variables, this->sim_message.getReadList(), this->sim_message.getWriteList(), MESSAGE_COUNT, d_keys, d_vals, hd_data.PBM);
        this->sim_message.swap();
        gpuErrchk(cudaStreamSynchronize(stream));  // Not strictly necessary while pbm_reorder is synchronous.
    }
    {  // Fill PBM and Message Texture Buffers
       // gpuErrchk(cudaBindTexture(nullptr, d_texMessages, d_agents, sizeof(glm::vec4) * MESSAGE_COUNT));
       // gpuErrchk(cudaBindTexture(nullptr, d_texPBM, d_PBM, sizeof(unsigned int) * (bucketCount + 1)));
    }
}

void MessageBucket::CUDAModelHandler::resizeCubTemp() {
    size_t bytesCheck = 0;
    gpuErrchk(cub::DeviceScan::ExclusiveSum(nullptr, bytesCheck, hd_data.PBM, d_histogram, bucketCount + 1));
    if (bytesCheck > d_CUB_temp_storage_bytes) {
        if (d_CUB_temp_storage) {
            gpuErrchk(flamegpu::detail::cuda::cudaFree(d_CUB_temp_storage));
        }
        d_CUB_temp_storage_bytes = bytesCheck;
        gpuErrchk(cudaMalloc(&d_CUB_temp_storage, d_CUB_temp_storage_bytes));
    }
}

void MessageBucket::CUDAModelHandler::resizeKeysVals(const unsigned int newSize) {
    size_t bytesCheck = newSize * sizeof(unsigned int);
    if (bytesCheck > d_keys_vals_storage_bytes) {
        if (d_keys) {
            gpuErrchk(flamegpu::detail::cuda::cudaFree(d_keys));
            gpuErrchk(flamegpu::detail::cuda::cudaFree(d_vals));
        }
        d_keys_vals_storage_bytes = bytesCheck;
        gpuErrchk(cudaMalloc(&d_keys, d_keys_vals_storage_bytes));
        gpuErrchk(cudaMalloc(&d_vals, d_keys_vals_storage_bytes));
    }
}

/// <summary>
/// CDescription
/// </summary>
MessageBucket::CDescription::CDescription(std::shared_ptr<Data> data)
    : MessageBruteForce::CDescription(std::move(std::static_pointer_cast<MessageBruteForce::Data>(data))) { }
MessageBucket::CDescription::CDescription(std::shared_ptr<const Data> data)
    : CDescription(std::move(std::const_pointer_cast<Data>(data))) { }

bool MessageBucket::CDescription::operator==(const CDescription& rhs) const {
    return *this->message == *rhs.message;  // Compare content is functionally the same
}
bool MessageBucket::CDescription::operator!=(const CDescription& rhs) const {
    return !(*this == rhs);
}
/**
 * Const accessors
 */
IntT MessageBucket::CDescription::getLowerBound() const {
    return std::static_pointer_cast<Data>(message)->lowerBound;
}
IntT MessageBucket::CDescription::getUpperBound() const {
    return std::static_pointer_cast<Data>(message)->upperBound;
}

/// <summary>
/// Description
/// </summary>
MessageBucket::Description::Description(std::shared_ptr<Data> data)
    : CDescription(data) { }
/**
 * Accessors
 */
void MessageBucket::Description::setLowerBound(const IntT min) {
    if (std::static_pointer_cast<Data>(message)->upperBound != std::numeric_limits<IntT>::max() &&
        min >= std::static_pointer_cast<Data>(message)->upperBound) {
        THROW exception::InvalidArgument("Bucket messaging minimum bound must be lower than upper bound, %lld !< %lld.", min, static_cast<int64_t>(std::static_pointer_cast<Data>(message)->upperBound));
    }
    std::static_pointer_cast<Data>(message)->lowerBound = min;
}
void MessageBucket::Description::setUpperBound(const IntT max) {
    if (max <= std::static_pointer_cast<Data>(message)->lowerBound) {
        THROW exception::InvalidArgument("Bucket messaging upperBound bound must be greater than lower bound, %lld !> %lld.", static_cast<int64_t>(max), static_cast<int64_t>(std::static_pointer_cast<Data>(message)->lowerBound));
    }
    std::static_pointer_cast<Data>(message)->upperBound = max;
}
void MessageBucket::Description::setBounds(const IntT min, const IntT max) {
    if (max <= min) {
        THROW exception::InvalidArgument("Bucket messaging upperBound bound must be greater than lower bound, %lld !> %lld.", static_cast<int64_t>(max), static_cast<int64_t>(min));
    }
    std::static_pointer_cast<Data>(message)->lowerBound = min;
    std::static_pointer_cast<Data>(message)->upperBound = max;
}
/// <summary>
/// Data
/// </summary>
MessageBucket::Data::Data(std::shared_ptr<const ModelData> model, const std::string &message_name)
    : MessageBruteForce::Data(model, message_name)
    , lowerBound(0)
    , upperBound(std::numeric_limits<IntT>::max()) {
    variables.emplace("_key", Variable(1, static_cast<IntT>(0)));
}
MessageBucket::Data::Data(std::shared_ptr<const ModelData> model, const Data &other)
    : MessageBruteForce::Data(model, other)
    , lowerBound(other.lowerBound)
    , upperBound(other.upperBound) {
    if (lowerBound == std::numeric_limits<IntT>::max()) {
        THROW exception::InvalidMessage("Minimum bound has not been set for bucket message '%s.", other.name.c_str());
    }
    if (upperBound == std::numeric_limits<IntT>::max()) {
        THROW exception::InvalidMessage("Maximum bound has not been set for bucket message '%s.", other.name.c_str());
    }
}
MessageBucket::Data *MessageBucket::Data::clone(const std::shared_ptr<const ModelData> &newParent) {
    return new Data(newParent, *this);
}
std::unique_ptr<MessageSpecialisationHandler> MessageBucket::Data::getSpecialisationHander(detail::CUDAMessage &owner) const {
    return std::unique_ptr<MessageSpecialisationHandler>(new CUDAModelHandler(owner));
}
std::type_index MessageBucket::Data::getType() const { return std::type_index(typeid(MessageBucket)); }




}  // namespace flamegpu
