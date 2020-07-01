#include "flamegpu/runtime/messaging/Spatial2D.h"

#ifdef _MSC_VER
#pragma warning(push, 3)
#include <cub/cub.cuh>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#endif

#include "flamegpu/runtime/messaging.h"
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/CUDAScatter.h"
#include "flamegpu/util/nvtx.h"


__device__ __forceinline__ MsgSpatial2D::GridPos2D getGridPosition2D(const MsgSpatial2D::MetaData *md, float x, float y) {
    // Clamp each grid coord to 0<=x<dim
    int gridPos[2] = {
        static_cast<int>(floor((x / md->environmentWidth[0])*md->gridDim[0])),
        static_cast<int>(floor((y / md->environmentWidth[1])*md->gridDim[1]))
    };
    MsgSpatial2D::GridPos2D rtn = {
        gridPos[0] < 0 ? 0 : (gridPos[0] >= md->gridDim[0] ? static_cast<int>(md->gridDim[0] - 1) : gridPos[0]),
        gridPos[1] < 0 ? 0 : (gridPos[1] >= md->gridDim[1] ? static_cast<int>(md->gridDim[1] - 1) : gridPos[1])
    };
    return rtn;
}
__device__ __forceinline__ unsigned int getHash2D(const MsgSpatial2D::MetaData *md, const MsgSpatial2D::GridPos2D &xyz) {
    // Bound gridPos to gridDimensions
    unsigned int gridPos[3] = {
        (unsigned int)(xyz.x < 0 ? 0 : (xyz.x >= md->gridDim[0] - 1 ? md->gridDim[0] - 1 : xyz.x)),  // Only x should ever be out of bounds here
        (unsigned int) xyz.y,  // xyz.y < 0 ? 0 : (xyz.y >= md->gridDim[1] - 1 ? md->gridDim[1] - 1 : xyz.y)
    };
    // Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
    return (unsigned int)(
        (gridPos[1] * md->gridDim[0]) +                    // y
        gridPos[0]);                                      // x
}

__device__ void MsgSpatial2D::Out::setLocation(const float &x, const float &y) const {
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // set the variables using curve
    Curve::setVariable<float>("x", combined_hash, x, index);
    Curve::setVariable<float>("y", combined_hash, y, index);

    // Set scan flag incase the message is optional
    this->scan_flag[index] = 1;
}

__device__ MsgSpatial2D::In::Filter::Filter(const MetaData* _metadata, const Curve::NamespaceHash &_combined_hash, const float& x, const float& y)
    : metadata(_metadata)
    , combined_hash(_combined_hash) {
    loc[0] = x;
    loc[1] = y;
    cell = getGridPosition2D(_metadata, x, y);
}
__device__ MsgSpatial2D::In::Filter::Message& MsgSpatial2D::In::Filter::Message::operator++() {
    cell_index++;
    bool move_strip = cell_index >= cell_index_max;
    while (move_strip) {
        nextStrip();
        cell_index = 0;
        cell_index_max = 1;
        if (relative_cell < 2) {
            // Calculate the strips start and end hash
            int absolute_cell_y = _parent.cell.y + relative_cell;
            // Skip the strip if it is completely out of bounds
            if (absolute_cell_y >= 0 && absolute_cell_y < _parent.metadata->gridDim[1]) {
                unsigned int start_hash = getHash2D(_parent.metadata, { _parent.cell.x - 1, absolute_cell_y });
                unsigned int end_hash = getHash2D(_parent.metadata, { _parent.cell.x + 1, absolute_cell_y });
                // Lookup start and end indicies from PBM
                cell_index = _parent.metadata->PBM[start_hash];
                cell_index_max = _parent.metadata->PBM[end_hash + 1];
            } else {
                // Goto next strip
                // Don't update move_strip
                continue;
            }
        }
        move_strip = cell_index >= cell_index_max;
    }
    return *this;
}


MsgSpatial2D::CUDAModelHandler::CUDAModelHandler(CUDAMessage &a)
    : MsgSpecialisationHandler()
    , sim_message(a) {
    NVTX_RANGE("MsgSpatial2D::CUDAModelHandler::CUDAModelHandler");
    const Data &d = (const Data &)a.getMessageDescription();
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
MsgSpatial2D::CUDAModelHandler::~CUDAModelHandler() { }
__global__ void atomicHistogram2D(
    const MsgSpatial2D::MetaData *md,
    unsigned int* bin_index,
    unsigned int* bin_sub_index,
    unsigned int *pbm_counts,
    unsigned int message_count,
    const float * __restrict__ x,
    const float * __restrict__ y) {
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Kill excess threads
    if (index >= message_count) return;

    MsgSpatial2D::GridPos2D gridPos = getGridPosition2D(md, x[index], y[index]);
    unsigned int hash = getHash2D(md, gridPos);
    bin_index[index] = hash;
    unsigned int bin_idx = atomicInc((unsigned int*)&pbm_counts[hash], 0xFFFFFFFF);
    bin_sub_index[index] = bin_idx;
}

void MsgSpatial2D::CUDAModelHandler::allocateMetaDataDevicePtr() {
    if (d_data == nullptr) {
        gpuErrchk(cudaMalloc(&d_histogram, (binCount + 1) * sizeof(unsigned int)));
        gpuErrchk(cudaMalloc(&hd_data.PBM, (binCount + 1) * sizeof(unsigned int)));
        gpuErrchk(cudaMalloc(&d_data, sizeof(MetaData)));
        gpuErrchk(cudaMemcpy(d_data, &hd_data, sizeof(MetaData), cudaMemcpyHostToDevice));
        resizeCubTemp();
    }
}

void MsgSpatial2D::CUDAModelHandler::freeMetaDataDevicePtr() {
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

void MsgSpatial2D::CUDAModelHandler::buildIndex(CUDAScatter &scatter, const unsigned int &streamId) {
    const unsigned int MESSAGE_COUNT = this->sim_message.getMessageCount();
    resizeKeysVals(this->sim_message.getMaximumListSize());  // Resize based on allocated amount rather than message count
    {  // Build atomic histogram
        gpuErrchk(cudaMemset(d_histogram, 0x00000000, (binCount + 1) * sizeof(unsigned int)));
        int blockSize;  // The launch configurator returned block size
        gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, atomicHistogram2D, 32, 0));  // Randomly 32
                                                                                                         // Round up according to array size
        int gridSize = (MESSAGE_COUNT + blockSize - 1) / blockSize;
        atomicHistogram2D << <gridSize, blockSize >> >(d_data, d_keys, d_vals, d_histogram, MESSAGE_COUNT,
            reinterpret_cast<float*>(this->sim_message.getReadPtr("x")),
            reinterpret_cast<float*>(this->sim_message.getReadPtr("y")));
        gpuErrchk(cudaDeviceSynchronize());
    }
    {  // Scan (sum), to finalise PBM
        gpuErrchk(cub::DeviceScan::ExclusiveSum(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_histogram, hd_data.PBM, binCount + 1));
    }
    {  // Reorder messages
       // Copy messages from d_messages to d_messages_swap, in hash order
        scatter.pbm_reorder(streamId, this->sim_message.getMessageDescription().variables, this->sim_message.getReadList(), this->sim_message.getWriteList(), MESSAGE_COUNT, d_keys, d_vals, hd_data.PBM);
        this->sim_message.swap();
    }
    {  // Fill PBM and Message Texture Buffers
       // gpuErrchk(cudaBindTexture(nullptr, d_texMessages, d_agents, sizeof(glm::vec4) * MESSAGE_COUNT));
       // gpuErrchk(cudaBindTexture(nullptr, d_texPBM, d_PBM, sizeof(unsigned int) * (binCount + 1)));
    }
}

void MsgSpatial2D::CUDAModelHandler::resizeCubTemp() {
    size_t bytesCheck = 0;
    gpuErrchk(cub::DeviceScan::ExclusiveSum(nullptr, bytesCheck, hd_data.PBM, d_histogram, binCount + 1));
    if (bytesCheck > d_CUB_temp_storage_bytes) {
        if (d_CUB_temp_storage) {
            gpuErrchk(cudaFree(d_CUB_temp_storage));
        }
        d_CUB_temp_storage_bytes = bytesCheck;
        gpuErrchk(cudaMalloc(&d_CUB_temp_storage, d_CUB_temp_storage_bytes));
    }
}

void MsgSpatial2D::CUDAModelHandler::resizeKeysVals(const unsigned int &newSize) {
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


MsgSpatial2D::Data::Data(const std::shared_ptr<const ModelData> &model, const std::string &message_name)
    : MsgBruteForce::Data(model, message_name)
    , radius(NAN)
    , minX(NAN)
    , minY(NAN)
    , maxX(NAN)
    , maxY(NAN) {
    description = std::unique_ptr<MsgSpatial2D::Description>(new MsgSpatial2D::Description(model, this));
    description->newVariable<float>("x");
    description->newVariable<float>("y");
}
MsgSpatial2D::Data::Data(const std::shared_ptr<const ModelData> &model, const Data &other)
    : MsgBruteForce::Data(model, other)
    , radius(other.radius)
    , minX(other.minX)
    , minY(other.minY)
    , maxX(other.maxX)
    , maxY(other.maxY) {
    description = std::unique_ptr<MsgSpatial2D::Description>(model ? new MsgSpatial2D::Description(model, this) : nullptr);
    if (isnan(radius)) {
        THROW InvalidMessage("Radius has not been set in spatial message '%s'.", other.name.c_str());
    }
    if (isnan(minX)) {
        THROW InvalidMessage("Environment minimum x bound has not been set in spatial message '%s.", other.name.c_str());
    }
    if (isnan(minY)) {
        THROW InvalidMessage("Environment minimum y bound has not been set in spatial message '%s'.", other.name.c_str());
    }
    if (isnan(maxX)) {
        THROW InvalidMessage("Environment maximum x bound has not been set in spatial message '%s'.", other.name.c_str());
    }
    if (isnan(maxY)) {
        THROW InvalidMessage("Environment maximum y bound has not been set in spatial message '%s'.", other.name.c_str());
    }
}
MsgSpatial2D::Data *MsgSpatial2D::Data::clone(const std::shared_ptr<const ModelData> &newParent) {
    return new Data(newParent, *this);
}
std::unique_ptr<MsgSpecialisationHandler> MsgSpatial2D::Data::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MsgSpecialisationHandler>(new CUDAModelHandler(owner));
}
std::type_index MsgSpatial2D::Data::getType() const { return std::type_index(typeid(MsgSpatial2D)); }


MsgSpatial2D::Description::Description(const std::shared_ptr<const ModelData> &_model, Data *const data)
    : MsgBruteForce::Description(_model, data) { }

void MsgSpatial2D::Description::setRadius(const float &r) {
    if (r <= 0) {
        THROW InvalidArgument("Spatial messaging radius must be a positive value, %f is not valid.", r);
    }
    reinterpret_cast<Data *>(message)->radius = r;
}
void MsgSpatial2D::Description::setMinX(const float &x) {
    if (!isnan(reinterpret_cast<Data *>(message)->maxX) &&
        x >= reinterpret_cast<Data *>(message)->maxX) {
        THROW InvalidArgument("Spatial messaging minimum bound must be lower than max bound, %f !< %f.", x, reinterpret_cast<Data *>(message)->maxX);
    }
    reinterpret_cast<Data *>(message)->minX = x;
}
void MsgSpatial2D::Description::setMinY(const float &y) {
    if (!isnan(reinterpret_cast<Data *>(message)->maxY) &&
        y >= reinterpret_cast<Data *>(message)->maxY) {
        THROW InvalidArgument("Spatial messaging minimum bound must be lower than max bound, %f !< %f.", y, reinterpret_cast<Data *>(message)->maxY);
    }
    reinterpret_cast<Data *>(message)->minY = y;
}
void MsgSpatial2D::Description::setMin(const float &x, const float &y) {
    if (!isnan(reinterpret_cast<Data *>(message)->maxX) &&
        x >= reinterpret_cast<Data *>(message)->maxX) {
        THROW InvalidArgument("Spatial messaging minimum bound must be lower than max bound, %f !< %f.", x, reinterpret_cast<Data *>(message)->maxX);
    }
    if (!isnan(reinterpret_cast<Data *>(message)->maxY) &&
        y >= reinterpret_cast<Data *>(message)->maxY) {
        THROW InvalidArgument("Spatial messaging minimum bound must be lower than max bound, %f !< %f.", y, reinterpret_cast<Data *>(message)->maxY);
    }
    reinterpret_cast<Data *>(message)->minX = x;
    reinterpret_cast<Data *>(message)->minY = y;
}
void MsgSpatial2D::Description::setMaxX(const float &x) {
    if (!isnan(reinterpret_cast<Data *>(message)->minX) &&
        x <= reinterpret_cast<Data *>(message)->minX) {
        THROW InvalidArgument("Spatial messaging max x bound must be greater than min bound, %f !> %f.", x, reinterpret_cast<Data *>(message)->minX);
    }
    reinterpret_cast<Data *>(message)->maxX = x;
}
void MsgSpatial2D::Description::setMaxY(const float &y) {
    if (!isnan(reinterpret_cast<Data *>(message)->minY) &&
        y <= reinterpret_cast<Data *>(message)->minY) {
        THROW InvalidArgument("Spatial messaging max y bound must be greater than min bound, %f !> %f.", y, reinterpret_cast<Data *>(message)->minY);
    }
    reinterpret_cast<Data *>(message)->maxY = y;
}
void MsgSpatial2D::Description::setMax(const float &x, const float &y) {
    if (!isnan(reinterpret_cast<Data *>(message)->minX) &&
        x <= reinterpret_cast<Data *>(message)->minX) {
        THROW InvalidArgument("Spatial messaging max x bound must be greater than min bound, %f !> %f.", x, reinterpret_cast<Data *>(message)->minX);
    }
    if (!isnan(reinterpret_cast<Data *>(message)->minY) &&
        y <= reinterpret_cast<Data *>(message)->minY) {
        THROW InvalidArgument("Spatial messaging max y bound must be greater than min bound, %f !> %f.", y, reinterpret_cast<Data *>(message)->minY);
    }
    reinterpret_cast<Data *>(message)->maxX = x;
    reinterpret_cast<Data *>(message)->maxY = y;
}

float MsgSpatial2D::Description::getRadius() const {
    return reinterpret_cast<Data *>(message)->radius;
}
float MsgSpatial2D::Description::getMinX() const {
    return reinterpret_cast<Data *>(message)->minX;
}
float MsgSpatial2D::Description::getMinY() const {
    return reinterpret_cast<Data *>(message)->minY;
}
float MsgSpatial2D::Description::getMaxX() const {
    return reinterpret_cast<Data *>(message)->maxX;
}
float MsgSpatial2D::Description::getMaxY() const {
    return reinterpret_cast<Data *>(message)->maxY;
}
