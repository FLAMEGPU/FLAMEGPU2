#include "flamegpu/runtime/messaging/Bucket.h"

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
#include "flamegpu/gpu/CUDAScatter.h"
#include "flamegpu/util/nvtx.h"

#include "flamegpu/runtime/messaging/Bucket/BucketHost.h"
// #include "flamegpu/runtime/messaging/Bucket/BucketDevice.h"

MsgBucket::CUDAModelHandler::CUDAModelHandler(CUDAMessage &a)
    : MsgSpecialisationHandler()
    , sim_message(a) {
    NVTX_RANGE("MsgBucket::CUDAModelHandler::CUDAModelHandler");
    const Data &d = (const Data &)a.getMessageDescription();
    hd_data.min = d.lowerBound;
    // Here we convert it so that upperBound is one greater than the final valid index
    hd_data.max = d.upperBound + 1;
    bucketCount = d.upperBound - d.lowerBound  + 1;
}
MsgBucket::CUDAModelHandler::~CUDAModelHandler() { }

__global__ void atomicHistogram1D(
    const MsgBucket::MetaData *md,
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

void MsgBucket::CUDAModelHandler::init(CUDAScatter &, const unsigned int &) {
    allocateMetaDataDevicePtr();
    // Set PBM to 0
    gpuErrchk(cudaMemset(hd_data.PBM, 0x00000000, (bucketCount + 1) * sizeof(unsigned int)));
}

void MsgBucket::CUDAModelHandler::allocateMetaDataDevicePtr() {
    if (d_data == nullptr) {
        gpuErrchk(cudaMalloc(&d_histogram, (bucketCount + 1) * sizeof(unsigned int)));
        gpuErrchk(cudaMalloc(&hd_data.PBM, (bucketCount + 1) * sizeof(unsigned int)));
        gpuErrchk(cudaMalloc(&d_data, sizeof(MetaData)));
        gpuErrchk(cudaMemcpy(d_data, &hd_data, sizeof(MetaData), cudaMemcpyHostToDevice));
        resizeCubTemp();
    }
}

void MsgBucket::CUDAModelHandler::freeMetaDataDevicePtr() {
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

void MsgBucket::CUDAModelHandler::buildIndex(CUDAScatter &scatter, const unsigned int &streamId) {
    const unsigned int MESSAGE_COUNT = this->sim_message.getMessageCount();
    resizeKeysVals(this->sim_message.getMaximumListSize());  // Resize based on allocated amount rather than message count
    {  // Build atomic histogram
        gpuErrchk(cudaMemset(d_histogram, 0x00000000, (bucketCount + 1) * sizeof(unsigned int)));
        int blockSize;  // The launch configurator returned block size
        gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, atomicHistogram1D, 32, 0));  // Randomly 32
                                                                                                         // Round up according to array size
        int gridSize = (MESSAGE_COUNT + blockSize - 1) / blockSize;
        atomicHistogram1D << <gridSize, blockSize >> >(d_data, d_keys, d_vals, d_histogram, MESSAGE_COUNT,
            reinterpret_cast<IntT*>(this->sim_message.getReadPtr("_key")));
        gpuErrchk(cudaDeviceSynchronize());
    }
    {  // Scan (sum), to finalise PBM
        gpuErrchk(cub::DeviceScan::ExclusiveSum(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_histogram, hd_data.PBM, bucketCount + 1));
    }
    {  // Reorder messages
       // Copy messages from d_messages to d_messages_swap, in hash order
        scatter.pbm_reorder(streamId, this->sim_message.getMessageDescription().variables, this->sim_message.getReadList(), this->sim_message.getWriteList(), MESSAGE_COUNT, d_keys, d_vals, hd_data.PBM);
        this->sim_message.swap();
    }
    {  // Fill PBM and Message Texture Buffers
       // gpuErrchk(cudaBindTexture(nullptr, d_texMessages, d_agents, sizeof(glm::vec4) * MESSAGE_COUNT));
       // gpuErrchk(cudaBindTexture(nullptr, d_texPBM, d_PBM, sizeof(unsigned int) * (bucketCount + 1)));
    }
}

void MsgBucket::CUDAModelHandler::resizeCubTemp() {
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

void MsgBucket::CUDAModelHandler::resizeKeysVals(const unsigned int &newSize) {
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


MsgBucket::Data::Data(const std::shared_ptr<const ModelData> &model, const std::string &message_name)
    : MsgBruteForce::Data(model, message_name)
    , lowerBound(0)
    , upperBound(std::numeric_limits<IntT>::max()) {
    description = std::unique_ptr<MsgBucket::Description>(new MsgBucket::Description(model, this));
    variables.emplace("_key", Variable(1, static_cast<IntT>(0)));
}
MsgBucket::Data::Data(const std::shared_ptr<const ModelData> &model, const Data &other)
    : MsgBruteForce::Data(model, other)
    , lowerBound(other.lowerBound)
    , upperBound(other.upperBound) {
    description = std::unique_ptr<MsgBucket::Description>(model ? new MsgBucket::Description(model, this) : nullptr);
    if (lowerBound == std::numeric_limits<IntT>::max()) {
        THROW InvalidMessage("Minimum bound has not been set for bucket message '%s.", other.name.c_str());
    }
    if (upperBound == std::numeric_limits<IntT>::max()) {
        THROW InvalidMessage("Maximum bound has not been set for bucket message '%s.", other.name.c_str());
    }
}
MsgBucket::Data *MsgBucket::Data::clone(const std::shared_ptr<const ModelData> &newParent) {
    return new Data(newParent, *this);
}
std::unique_ptr<MsgSpecialisationHandler> MsgBucket::Data::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MsgSpecialisationHandler>(new CUDAModelHandler(owner));
}
std::type_index MsgBucket::Data::getType() const { return std::type_index(typeid(MsgBucket)); }


MsgBucket::Description::Description(const std::shared_ptr<const ModelData> &_model, Data *const data)
    : MsgBruteForce::Description(_model, data) { }

void MsgBucket::Description::setLowerBound(const IntT &min) {
    if (reinterpret_cast<Data *>(message)->upperBound != std::numeric_limits<IntT>::max() &&
        min >= reinterpret_cast<Data *>(message)->upperBound) {
        THROW InvalidArgument("Bucket messaging minimum bound must be lower than upper bound, %lld !< %lld.", min, static_cast<int64_t>(reinterpret_cast<Data *>(message)->upperBound));
    }
    reinterpret_cast<Data *>(message)->lowerBound = min;
}
void MsgBucket::Description::setUpperBound(const IntT &max) {
    if (max <= reinterpret_cast<Data *>(message)->lowerBound) {
        THROW InvalidArgument("Bucket messaging upperBound bound must be greater than lower bound, %lld !> %lld.", static_cast<int64_t>(max), static_cast<int64_t>(reinterpret_cast<Data *>(message)->lowerBound));
    }
    reinterpret_cast<Data *>(message)->upperBound = max;
}
void MsgBucket::Description::setBounds(const IntT &min, const IntT &max) {
    if (max <= min) {
        THROW InvalidArgument("Bucket messaging upperBound bound must be greater than lower bound, %lld !> %lld.", static_cast<int64_t>(max), static_cast<int64_t>(min));
    }
    reinterpret_cast<Data *>(message)->lowerBound = min;
    reinterpret_cast<Data *>(message)->upperBound = max;
}

IntT MsgBucket::Description::getLowerBound() const {
    return reinterpret_cast<Data *>(message)->lowerBound;
}
IntT MsgBucket::Description::getUpperBound() const {
    return reinterpret_cast<Data *>(message)->upperBound;
}
