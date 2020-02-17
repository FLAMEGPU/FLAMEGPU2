#include "flamegpu/runtime/messaging/Spatial3D.h"

#include "flamegpu/gpu/CUDAScatter.h"
#ifdef _MSC_VER
#pragma warning(push, 3)
#include <cub/cub.cuh>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#endif

__device__ __forceinline__ MsgSpatial3D::GridPos3D getGridPosition(const MsgSpatial3D::MetaData *md, float x, float y, float z) {
    // Clamp each grid coord to 0<=x<dim
    int gridPos[3] = {
        static_cast<int>(floor((x / md->environmentWidth[0])*md->gridDim[0])),
        static_cast<int>(floor((y / md->environmentWidth[1])*md->gridDim[1])),
        static_cast<int>(floor((z / md->environmentWidth[2])*md->gridDim[2]))
    };
    MsgSpatial3D::GridPos3D rtn = {
        gridPos[0] < 0 ? 0 : (gridPos[0] >= md->gridDim[0] ? static_cast<int>(md->gridDim[0] - 1) : gridPos[0]),
        gridPos[1] < 0 ? 0 : (gridPos[1] >= md->gridDim[1] ? static_cast<int>(md->gridDim[1] - 1) : gridPos[1]),
        gridPos[2] < 0 ? 0 : (gridPos[2] >= md->gridDim[2] ? static_cast<int>(md->gridDim[2] - 1) : gridPos[2])
    };
    return rtn;
}
__device__ __forceinline__ unsigned int getHash(const MsgSpatial3D::MetaData *md, const MsgSpatial3D::GridPos3D &xyz) {
    // Bound gridPos to gridDimensions
    unsigned int gridPos[3] = {
        xyz.x < 0 ? 0 : (xyz.x >= md->gridDim[0] - 1 ? md->gridDim[0] - 1 : xyz.x),  // Only x should ever be out of bounds here
        xyz.y,  // xyz.y < 0 ? 0 : (xyz.y >= md->gridDim[1] - 1 ? md->gridDim[1] - 1 : xyz.y),
        xyz.z,  // xyz.z < 0 ? 0 : (xyz.z >= md->gridDim[2] - 1 ? md->gridDim[2] - 1 : xyz.z)
    };
    // Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
    return (unsigned int)(
        (gridPos[2] * md->gridDim[0] * md->gridDim[1]) +   // z
        (gridPos[1] * md->gridDim[0]) +                    // y
        gridPos[0]);                                      // x
}

__device__ void MsgSpatial3D::Out::setLocation(const float &x, const float &y, const float &z) const {
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // set the variables using curve
    Curve::setVariable<float>("x", combined_hash, x, index);
    Curve::setVariable<float>("y", combined_hash, y, index);
    Curve::setVariable<float>("z", combined_hash, z, index);

    // Set scan flag incase the message is optional
    flamegpu_internal::CUDAScanCompaction::ds_message_configs[streamId].scan_flag[index] = 1;
}

__device__ MsgSpatial3D::In::Filter::Filter(const MetaData* _metadata, const Curve::NamespaceHash &_combined_hash, const float& x, const float& y, const float& z)
    : metadata(_metadata)
    , combined_hash(_combined_hash) {
    loc[0] = x;
    loc[1] = y;
    loc[2] = z;
    cell = getGridPosition(_metadata, x, y, z);
}
__device__ MsgSpatial3D::In::Filter::Message& MsgSpatial3D::In::Filter::Message::operator++() {
    cell_index++;
    bool move_strip = cell_index >= cell_index_max;
    while (move_strip) {
        nextStrip();
        cell_index = 0;
        cell_index_max = 1;
        if (relative_cell[0] < 2) {
            // Calculate the strips start and end hash
            int absolute_cell[2] = { _parent.cell.y + relative_cell[0], _parent.cell.z + relative_cell[1] };
            // Skip the strip if it is completely out of bounds
            if (absolute_cell[0] >= 0 && absolute_cell[1] >= 0 && absolute_cell[0] < _parent.metadata->gridDim[1] && absolute_cell[1] < _parent.metadata->gridDim[2]) {
                unsigned int start_hash = getHash(_parent.metadata, { _parent.cell.x - 1, absolute_cell[0], absolute_cell[1] });
                unsigned int end_hash = getHash(_parent.metadata, { _parent.cell.x + 1, absolute_cell[0], absolute_cell[1] });
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


__global__ void atomicHistogram3D(
    const MsgSpatial3D::MetaData *md,
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

    MsgSpatial3D::GridPos3D gridPos = getGridPosition(md, x[index], y[index], z[index]);
    unsigned int hash = getHash(md, gridPos);
    bin_index[index] = hash;
    unsigned int bin_idx = atomicInc((unsigned int*)&pbm_counts[hash], 0xFFFFFFFF);
    bin_sub_index[index] = bin_idx;
}

void MsgSpatial3D::CUDAModelHandler::buildIndex() {
    const unsigned int MESSAGE_COUNT = this->sim_message.getMessageCount();
    resizeKeysVals(this->sim_message.getMaximumListSize());  // Resize based on allocated amount rather than message count
    {  // Build atomic histogram
        gpuErrchk(cudaMemset(d_histogram, 0x00000000, (binCount + 1) * sizeof(unsigned int)));
        int blockSize;  // The launch configurator returned block size
        gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, atomicHistogram3D, 32, 0));  // Randomly 32
                                                                                                         // Round up according to array size
        int gridSize = (MESSAGE_COUNT + blockSize - 1) / blockSize;
        atomicHistogram3D << <gridSize, blockSize >> >(d_data, d_keys, d_vals, d_histogram, MESSAGE_COUNT,
            reinterpret_cast<float*>(this->sim_message.getReadPtr("x")),
            reinterpret_cast<float*>(this->sim_message.getReadPtr("y")),
            reinterpret_cast<float*>(this->sim_message.getReadPtr("z")));
        gpuErrchk(cudaDeviceSynchronize());
    }
    {  // Scan (sum), to finalise PBM
        cub::DeviceScan::ExclusiveSum(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_histogram, hd_data.PBM, binCount + 1);
    }
    {  // Reorder messages
       // Copy messages from d_messages to d_messages_swap, in hash order
        auto &cs = CUDAScatter::getInstance(0);  // Choose proper stream_id in future!
        cs.pbm_reorder(this->sim_message.getMessageDescription().variables, this->sim_message.getReadList(), this->sim_message.getWriteList(), MESSAGE_COUNT, d_keys, d_vals, hd_data.PBM);
        this->sim_message.swap(false, 0);  // Stream id is unused here
    }
    {  // Fill PBM and Message Texture Buffers
       // gpuErrchk(cudaBindTexture(nullptr, d_texMessages, d_agents, sizeof(glm::vec4) * MESSAGE_COUNT));
       // gpuErrchk(cudaBindTexture(nullptr, d_texPBM, d_PBM, sizeof(unsigned int) * (binCount + 1)));
    }
}

void MsgSpatial3D::CUDAModelHandler::resizeCubTemp() {
    size_t bytesCheck = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, bytesCheck, hd_data.PBM, d_histogram, binCount + 1);
    if (bytesCheck > d_CUB_temp_storage_bytes) {
        if (d_CUB_temp_storage) {
            gpuErrchk(cudaFree(d_CUB_temp_storage));
        }
        d_CUB_temp_storage_bytes = bytesCheck;
        gpuErrchk(cudaMalloc(&d_CUB_temp_storage, d_CUB_temp_storage_bytes));
    }
}

void MsgSpatial3D::CUDAModelHandler::resizeKeysVals(const unsigned int &newSize) {
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
