#include "flamegpu/runtime/messaging/MessageBucket.h"

#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable : 4706 4834)
#include <cub/cub.cuh>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#endif

#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/CUDAScatter.cuh"
#include "flamegpu/util/nvtx.h"

#include "flamegpu/runtime/messaging/MessageBucket/MessageBucketHost.h"
// #include "flamegpu/runtime/messaging/MessageBucket/MessageBucketDevice.cuh"

namespace flamegpu {

MessageBucket::CUDAModelHandler::CUDAModelHandler(CUDAMessage &a)
    : MessageSpecialisationHandler()
    , sim_message(a) {
    NVTX_RANGE("MessageBucket::CUDAModelHandler::CUDAModelHandler");
    const Data &d = (const Data &)a.getMessageDescription();
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

void MessageBucket::CUDAModelHandler::init(CUDAScatter &, const unsigned int &) {
    allocateMetaDataDevicePtr();
    // Set PBM to 0
    gpuErrchk(cudaMemset(hd_data.PBM, 0x00000000, (bucketCount + 1) * sizeof(unsigned int)));
}

void MessageBucket::CUDAModelHandler::allocateMetaDataDevicePtr() {
    if (d_data == nullptr) {
        gpuErrchk(cudaMalloc(&d_histogram, (bucketCount + 1) * sizeof(unsigned int)));
        gpuErrchk(cudaMalloc(&hd_data.PBM, (bucketCount + 1) * sizeof(unsigned int)));
        gpuErrchk(cudaMalloc(&d_data, sizeof(MetaData)));
        gpuErrchk(cudaMemcpy(d_data, &hd_data, sizeof(MetaData), cudaMemcpyHostToDevice));
        resizeCubTemp();
    }
}

void MessageBucket::CUDAModelHandler::freeMetaDataDevicePtr() {
    if (d_data != nullptr) {
        d_CUB_temp_storage_bytes = 0;
        gpuErrchk(cudaFree(d_CUB_temp_storage));
        gpuErrchk(cudaFree(d_histogram));
        gpuErrchk(cudaFree(hd_data.PBM));
        gpuErrchk(cudaFree(d_data));
        d_CUB_temp_storage = nullptr;
        d_histogram = nullptr;
        hd_data.PBM = nullptr;
        d_data = nullptr;
        if (d_keys) {
            d_keys_vals_storage_bytes = 0;
            gpuErrchk(cudaFree(d_keys));
            gpuErrchk(cudaFree(d_vals));
            d_keys = nullptr;
            d_vals = nullptr;
        }
    }
}

void MessageBucket::CUDAModelHandler::buildIndex(CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    NVTX_RANGE("MessageBucket::CUDAModelHandler::buildIndex");
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
        scatter.pbm_reorder(streamId, stream, this->sim_message.getMessageDescription().variables, this->sim_message.getReadList(), this->sim_message.getWriteList(), MESSAGE_COUNT, d_keys, d_vals, hd_data.PBM);
        this->sim_message.swap();
        gpuErrchk(cudaStreamSynchronize(stream));  // Not striclty neceesary while pbm_reorder is synchronous.
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
            gpuErrchk(cudaFree(d_CUB_temp_storage));
        }
        d_CUB_temp_storage_bytes = bytesCheck;
        gpuErrchk(cudaMalloc(&d_CUB_temp_storage, d_CUB_temp_storage_bytes));
    }
}

void MessageBucket::CUDAModelHandler::resizeKeysVals(const unsigned int &newSize) {
    size_t bytesCheck = newSize * sizeof(unsigned int);
    if (bytesCheck > d_keys_vals_storage_bytes) {
        if (d_keys) {
            gpuErrchk(cudaFree(d_keys));
            gpuErrchk(cudaFree(d_vals));
        }
        d_keys_vals_storage_bytes = bytesCheck;
        gpuErrchk(cudaMalloc(&d_keys, d_keys_vals_storage_bytes));
        gpuErrchk(cudaMalloc(&d_vals, d_keys_vals_storage_bytes));
    }
}


MessageBucket::Data::Data(const std::shared_ptr<const ModelData> &model, const std::string &message_name)
    : MessageBruteForce::Data(model, message_name)
    , lowerBound(0)
    , upperBound(std::numeric_limits<IntT>::max()) {
    description = std::unique_ptr<MessageBucket::Description>(new MessageBucket::Description(model, this));
    variables.emplace("_key", Variable(1, static_cast<IntT>(0)));
}
MessageBucket::Data::Data(const std::shared_ptr<const ModelData> &model, const Data &other)
    : MessageBruteForce::Data(model, other)
    , lowerBound(other.lowerBound)
    , upperBound(other.upperBound) {
    description = std::unique_ptr<MessageBucket::Description>(model ? new MessageBucket::Description(model, this) : nullptr);
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
std::unique_ptr<MessageSpecialisationHandler> MessageBucket::Data::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MessageSpecialisationHandler>(new CUDAModelHandler(owner));
}
std::type_index MessageBucket::Data::getType() const { return std::type_index(typeid(MessageBucket)); }


MessageBucket::Description::Description(const std::shared_ptr<const ModelData> &_model, Data *const data)
    : MessageBruteForce::Description(_model, data) { }

void MessageBucket::Description::setLowerBound(const IntT &min) {
    if (reinterpret_cast<Data *>(message)->upperBound != std::numeric_limits<IntT>::max() &&
        min >= reinterpret_cast<Data *>(message)->upperBound) {
        THROW exception::InvalidArgument("Bucket messaging minimum bound must be lower than upper bound, %lld !< %lld.", min, static_cast<int64_t>(reinterpret_cast<Data *>(message)->upperBound));
    }
    reinterpret_cast<Data *>(message)->lowerBound = min;
}
void MessageBucket::Description::setUpperBound(const IntT &max) {
    if (max <= reinterpret_cast<Data *>(message)->lowerBound) {
        THROW exception::InvalidArgument("Bucket messaging upperBound bound must be greater than lower bound, %lld !> %lld.", static_cast<int64_t>(max), static_cast<int64_t>(reinterpret_cast<Data *>(message)->lowerBound));
    }
    reinterpret_cast<Data *>(message)->upperBound = max;
}
void MessageBucket::Description::setBounds(const IntT &min, const IntT &max) {
    if (max <= min) {
        THROW exception::InvalidArgument("Bucket messaging upperBound bound must be greater than lower bound, %lld !> %lld.", static_cast<int64_t>(max), static_cast<int64_t>(min));
    }
    reinterpret_cast<Data *>(message)->lowerBound = min;
    reinterpret_cast<Data *>(message)->upperBound = max;
}

IntT MessageBucket::Description::getLowerBound() const {
    return reinterpret_cast<Data *>(message)->lowerBound;
}
IntT MessageBucket::Description::getUpperBound() const {
    return reinterpret_cast<Data *>(message)->upperBound;
}

}  // namespace flamegpu
