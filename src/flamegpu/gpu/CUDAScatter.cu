#include "flamegpu/gpu/CUDAScatter.h"

#include <cuda_runtime.h>
#include <vector>
#include <cassert>

#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"

unsigned int CUDAScatter::simulationInstances = 0;

CUDAScatter::CUDAScatter()
    : d_data(nullptr)
    , data_len(0) {
}
CUDAScatter::~CUDAScatter() {
    free();
}
void CUDAScatter::free() {
    if (d_data) {
        gpuErrchk(cudaFree(d_data));
    }
    d_data = nullptr;
    data_len = 0;
}

void CUDAScatter::resize(const unsigned int &newLen) {
    if (newLen > data_len) {
        if (d_data) {
            gpuErrchk(cudaFree(d_data));
        }
        gpuErrchk(cudaMalloc(&d_data, newLen * sizeof(ScatterData)));
        data_len = newLen;
    }
}

template <typename T>
__global__ void scatter_generic(
    unsigned int threadCount,
    T scan_flag,
    unsigned int *position,
    CUDAScatter::ScatterData *scatter_data,
    const unsigned int scatter_len,
    const unsigned int out_index_offset = 0) {
    // global thread index
    int index = (blockIdx.x*blockDim.x) + threadIdx.x;

    if (index >= threadCount) return;

    // if optional message is to be written
    if (scan_flag[index] == 1) {
        int output_index = position[index];
        for (unsigned int i = 0; i < scatter_len; ++i) {
            memcpy(scatter_data[i].out + ((out_index_offset + output_index) * scatter_data[i].typeLen), scatter_data[i].in + (index * scatter_data[i].typeLen), scatter_data[i].typeLen);
        }
    }
}
__global__ void scatter_all_generic(
    unsigned int threadCount,
    CUDAScatter::ScatterData *scatter_data,
    const unsigned int scatter_len,
    const unsigned int out_index_offset = 0) {
    // global thread index
    int index = (blockIdx.x*blockDim.x) + threadIdx.x;

    if (index >= threadCount) return;
    for (unsigned int i = 0; i < scatter_len; ++i) {
        memcpy(scatter_data[i].out + ((out_index_offset + index) * scatter_data[i].typeLen), scatter_data[i].in + (index * scatter_data[i].typeLen), scatter_data[i].typeLen);
    }
}

unsigned int CUDAScatter::scatter(
    Type messageOrAgent,
    const VariableMap &vars,
    const std::map<std::string, void*> &in,
    const std::map<std::string, void*> &out,
    const unsigned int &itemCount,
    const unsigned int &out_index_offset,
    const bool &invert_scan_flag) {
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size
    // calculate the grid block size for main agent function
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scatter_generic<unsigned int*>, 0, itemCount);
    //! Round up according to CUDAAgent state list size
    gridSize = (itemCount + blockSize - 1) / blockSize;
    // for each variable, scatter from swap to regular
    std::vector<ScatterData> sd;
    for (const auto &v : vars) {
        char *in_p = reinterpret_cast<char*>(in.at(v.first));
        char *out_p = reinterpret_cast<char*>(out.at(v.first));
        sd.push_back({ v.second.type_size, in_p, out_p });
    }
    resize(static_cast<unsigned int>(sd.size()));
    // Important that sd.size() is still used here, incase allocated len (data_len) is bigger
    gpuErrchk(cudaMemcpy(d_data, sd.data(), sizeof(ScatterData) * sd.size(), cudaMemcpyHostToDevice));
    if (invert_scan_flag) {
        if (messageOrAgent == Agent) {
            scatter_generic <<<gridSize, blockSize>>> (
                itemCount,
                InversionIterator(flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].d_ptrs.scan_flag),
                flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].d_ptrs.position,
                d_data, static_cast<unsigned int>(sd.size()),
                out_index_offset);
        } else if (messageOrAgent == Message) {
            scatter_generic <<<gridSize, blockSize>>> (
                itemCount,
                InversionIterator(flamegpu_internal::CUDAScanCompaction::hd_message_configs[streamId].d_ptrs.scan_flag),
                flamegpu_internal::CUDAScanCompaction::hd_message_configs[streamId].d_ptrs.position,
                d_data, static_cast<unsigned int>(sd.size()),
                out_index_offset);
        }
    } else {
        if (messageOrAgent == Agent) {
            scatter_generic <<<gridSize, blockSize>>> (
                itemCount,
                flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].d_ptrs.scan_flag,
                flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].d_ptrs.position,
                d_data, static_cast<unsigned int>(sd.size()),
                out_index_offset);
        } else if (messageOrAgent == Message) {
            scatter_generic <<<gridSize, blockSize>>> (
                itemCount,
                flamegpu_internal::CUDAScanCompaction::hd_message_configs[streamId].d_ptrs.scan_flag,
                flamegpu_internal::CUDAScanCompaction::hd_message_configs[streamId].d_ptrs.position,
                d_data, static_cast<unsigned int>(sd.size()),
                out_index_offset);
        }
    }
    gpuErrchkLaunch();
    // Update count of live agents
    unsigned int rtn = 0;
    if (messageOrAgent == Agent) {
        gpuErrchk(cudaMemcpy(&rtn, flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].d_ptrs.position + itemCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    } else if (messageOrAgent == Message) {
        gpuErrchk(cudaMemcpy(&rtn, flamegpu_internal::CUDAScanCompaction::hd_message_configs[streamId].d_ptrs.position + itemCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }
        return rtn;
}

unsigned int CUDAScatter::scatterAll(
    const VariableMap &vars,
    const std::map<std::string, void*> &in,
    const std::map<std::string, void*> &out,
    const unsigned int &itemCount,
    const unsigned int &out_index_offset) {
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size

                       // calculate the grid block size for main agent function
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scatter_all_generic, 0, itemCount);
    //! Round up according to CUDAAgent state list size
    gridSize = (itemCount + blockSize - 1) / blockSize;
    // for each variable, scatter from swap to regular
    std::vector<ScatterData> sd;
    for (const auto &v : vars) {
        char *in_p = reinterpret_cast<char*>(in.at(v.first));
        char *out_p = reinterpret_cast<char*>(out.at(v.first));
        sd.push_back({ v.second.type_size, in_p, out_p });
    }
    resize(static_cast<unsigned int>(sd.size()));
    // Important that sd.size() is still used here, incase allocated len (data_len) is bigger
    gpuErrchk(cudaMemcpy(d_data, sd.data(), sizeof(ScatterData) * sd.size(), cudaMemcpyHostToDevice));
    scatter_all_generic << <gridSize, blockSize >> > (
        itemCount,
        d_data, static_cast<unsigned int>(sd.size()),
        out_index_offset);
    gpuErrchkLaunch();
    // Update count of live agents
    return itemCount;
}

__global__ void pbm_reorder_generic(
    const unsigned int threadCount,
    const unsigned int * __restrict__ bin_index,
    const unsigned int * __restrict__ bin_sub_index,
    const unsigned int * __restrict__ pbm,
    CUDAScatter::ScatterData *scatter_data,
    const unsigned int scatter_len) {
    // global thread index
    int index = (blockIdx.x*blockDim.x) + threadIdx.x;

    if (index >= threadCount) return;

    const unsigned int sorted_index = pbm[bin_index[index]] + bin_sub_index[index];

    // if optional message is to be written
    for (unsigned int i = 0; i < scatter_len; ++i) {
        memcpy(scatter_data[i].out + (sorted_index * scatter_data[i].typeLen), scatter_data[i].in + (index * scatter_data[i].typeLen), scatter_data[i].typeLen);
    }
}

void CUDAScatter::pbm_reorder(
    const VariableMap &vars,
    const std::map<std::string, void*> &in,
    const std::map<std::string, void*> &out,
    const unsigned int &itemCount,
    const unsigned int *d_bin_index,
    const unsigned int *d_bin_sub_index,
    const unsigned int *d_pbm) {
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size

                       // calculate the grid block size for main agent function
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, pbm_reorder_generic, 0, itemCount);
    //! Round up according to CUDAAgent state list size
    gridSize = (itemCount + blockSize - 1) / blockSize;
    // for each variable, scatter from swap to regular
    std::vector<ScatterData> sd;
    for (const auto &v : vars) {
        char *in_p = reinterpret_cast<char*>(in.at(v.first));
        char *out_p = reinterpret_cast<char*>(out.at(v.first));
        sd.push_back({ v.second.type_size, in_p, out_p });
    }
    resize(static_cast<unsigned int>(sd.size()));
    // Important that sd.size() is still used here, incase allocated len (data_len) is bigger
    gpuErrchk(cudaMemcpy(d_data, sd.data(), sizeof(ScatterData) * sd.size(), cudaMemcpyHostToDevice));
    pbm_reorder_generic <<<gridSize, blockSize>>> (
            itemCount,
            d_bin_index,
            d_bin_sub_index,
            d_pbm,
            d_data, static_cast<unsigned int>(sd.size()));
    gpuErrchkLaunch();
}


void CUDAScatter::increaseSimCounter() {
    simulationInstances++;
}
void CUDAScatter::decreaseSimCounter() {
    simulationInstances--;
    if (simulationInstances == 0) {
        for (unsigned int i = 0; i < flamegpu_internal::CUDAScanCompaction::MAX_STREAMS; ++i) {
            getInstance(i).free();
        }
    }
}
