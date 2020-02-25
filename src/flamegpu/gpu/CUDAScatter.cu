#include "flamegpu/gpu/CUDAScatter.h"

#include <cuda_runtime.h>
#include <vector>
#include <cassert>

#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"
#include "flamegpu/runtime/flamegpu_host_new_agent_api.h"

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
        scatter_generic << <gridSize, blockSize >> > (
            itemCount,
            InversionIterator(flamegpu_internal::CUDAScanCompaction::hd_configs[messageOrAgent][streamId].d_ptrs.scan_flag),
            flamegpu_internal::CUDAScanCompaction::hd_configs[messageOrAgent][streamId].d_ptrs.position,
            d_data, static_cast<unsigned int>(sd.size()),
            out_index_offset);
    } else {
        scatter_generic << <gridSize, blockSize >> > (
            itemCount,
            flamegpu_internal::CUDAScanCompaction::hd_configs[messageOrAgent][streamId].d_ptrs.scan_flag,
            flamegpu_internal::CUDAScanCompaction::hd_configs[messageOrAgent][streamId].d_ptrs.position,
            d_data, static_cast<unsigned int>(sd.size()),
            out_index_offset);
    }
    gpuErrchkLaunch();
    // Update count of live agents
    unsigned int rtn = 0;
    gpuErrchk(cudaMemcpy(&rtn, flamegpu_internal::CUDAScanCompaction::hd_configs[messageOrAgent][streamId].d_ptrs.position + itemCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
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

/**
 * Scatter kernel for host agent creation
 * Input data is stored in AoS, and translated to SoA for device
 * @param threadCount Total number of threads required
 * @param agent_size The total size of an agent's variables in memory, for stepping through input array
 * @param scatter_data Scatter data array location in memory
 * @param scatter_len Length of scatter data array
 * @parma out_index_offset The number of agents already in the output array (so that they are not overwritten)
 */
__global__ void scatter_new_agents(
    unsigned int threadCount,
    const unsigned int agent_size,
    CUDAScatter::ScatterData *scatter_data,
    const unsigned int scatter_len,
    const unsigned int out_index_offset) {
    // global thread index
    int index = (blockIdx.x*blockDim.x) + threadIdx.x;

    if (index >= threadCount) return;

    // Which variable are we outputting
    const unsigned int var_out = index % scatter_len;
    const unsigned int agent_index = index / scatter_len;

    // if optional message is to be written
    char * const in_ptr = scatter_data[var_out].in + (agent_index * agent_size);
    char * const out_ptr = scatter_data[var_out].out + ((out_index_offset + agent_index) * scatter_data[var_out].typeLen);
    memcpy(out_ptr, in_ptr, scatter_data[var_out].typeLen);
}
void CUDAScatter::scatterNewAgents(
    const VariableMap &vars,
    const std::map<std::string, void*> &out,
    void *d_in_buff,
    const VarOffsetStruct &inOffsetData,
    const unsigned int &inCount,
    const unsigned int outIndexOffset) {
    // 1 thread per agent variable
    const unsigned int threadCount = static_cast<unsigned int>(inOffsetData.vars.size()) * inCount;
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size

    // calculate the grid block size for main agent function
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scatter_new_agents, 0, threadCount);
    //! Round up according to CUDAAgent state list size
    gridSize = (threadCount + blockSize - 1) / blockSize;
    // for each variable, scatter from swap to regular
    std::vector<ScatterData> sd;
    for (const auto &v : vars) {
        // In this case, in is the location of first variable, but we step by inOffsetData.totalSize
        char *in_p = reinterpret_cast<char*>(d_in_buff) + inOffsetData.vars.at(v.first).offset;
        char *out_p = reinterpret_cast<char*>(out.at(v.first));
        sd.push_back({ v.second.type_size, in_p, out_p });
    }
    resize(static_cast<unsigned int>(sd.size()));
    // Important that sd.size() is still used here, incase allocated len (data_len) is bigger
    gpuErrchk(cudaMemcpy(d_data, sd.data(), sizeof(ScatterData) * sd.size(), cudaMemcpyHostToDevice));
    scatter_new_agents << <gridSize, blockSize >> > (
        threadCount,
        static_cast<unsigned int>(inOffsetData.totalSize),
        d_data, static_cast<unsigned int>(sd.size()),
        outIndexOffset);
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
