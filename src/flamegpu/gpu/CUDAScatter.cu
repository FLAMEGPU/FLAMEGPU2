#include "flamegpu/gpu/CUDAScatter.h"

#include <cuda_runtime.h>
#include <vector>
#include <cassert>

#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"
#include "flamegpu/gpu/CUDAFatAgentStateList.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"

#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable : 4706 4834)
#include <cub/cub.cuh>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#endif

namespace flamegpu {

// @todo - Make _async variants of functions which launch kernels. This can be called by the non async version and immediately sync.

CUDAScatter::StreamData::StreamData()
    : d_data(nullptr)
    , data_len(0) {
}
CUDAScatter::StreamData::~StreamData() {
    /* @note - Do not clear cuda memory in the destructor of singletons.
     This is because order of static destruction in c++ is undefined
     So the cuda driver is not guaranteed to still exist when the static is destroyed.
     As this is only ever destroyed at exit time, it's not a real memory leak either.
    */
    if (d_data) {
        gpuErrchk(cudaFree(d_data));
    }
    d_data = nullptr;
    data_len = 0;
}
void CUDAScatter::StreamData::purge() {
    d_data = nullptr;
    data_len = 0;
}
void CUDAScatter::StreamData::resize(const unsigned int &newLen) {
    if (newLen > data_len) {
        if (d_data) {
            gpuErrchk(cudaFree(d_data));
        }
        gpuErrchk(cudaMalloc(&d_data, newLen * sizeof(ScatterData)));
        data_len = newLen;
    }
}

void CUDAScatter::purge() {
    for (auto &s : streamResources) {
        s.purge();
    }
    scan.purge();
}


template <typename T>
__global__ void scatter_generic(
    unsigned int threadCount,
    T scan_flag,
    unsigned int *position,
    CUDAScatter::ScatterData *scatter_data,
    const unsigned int scatter_len,
    const unsigned int out_index_offset = 0,
    const unsigned int scatter_all_count = 0) {
    // global thread index
    int index = (blockIdx.x*blockDim.x) + threadIdx.x;

    if (index >= threadCount) return;

    // if optional message is to be written
    if (index < scatter_all_count || scan_flag[index - scatter_all_count] == 1) {
        int output_index = index < scatter_all_count ? index : scatter_all_count + position[index - scatter_all_count];
        for (unsigned int i = 0; i < scatter_len; ++i) {
            memcpy(scatter_data[i].out + ((out_index_offset + output_index) * scatter_data[i].typeLen), scatter_data[i].in + (index * scatter_data[i].typeLen), scatter_data[i].typeLen);
        }
    }
}
__global__ void scatter_position_generic(
    unsigned int threadCount,
    unsigned int *position,
    CUDAScatter::ScatterData *scatter_data,
    const unsigned int scatter_len) {
    // global thread index
    int index = (blockIdx.x*blockDim.x) + threadIdx.x;

    if (index >= threadCount) return;

    // if optional message is to be written
    int input_index = position[index];
    for (unsigned int i = 0; i < scatter_len; ++i) {
        memcpy(scatter_data[i].out + (index * scatter_data[i].typeLen), scatter_data[i].in + (input_index * scatter_data[i].typeLen), scatter_data[i].typeLen);
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
    const unsigned int &streamResourceId,
    const cudaStream_t &stream,
    const Type &messageOrAgent,
    const VariableMap &vars,
    const std::map<std::string, void*> &in,
    const std::map<std::string, void*> &out,
    const unsigned int &itemCount,
    const unsigned int &out_index_offset,
    const bool &invert_scan_flag,
    const unsigned int &scatter_all_count) {
    std::vector<ScatterData> scatterData;
    for (const auto &v : vars) {
        char *in_p = reinterpret_cast<char*>(in.at(v.first));
        char *out_p = reinterpret_cast<char*>(out.at(v.first));
        scatterData.push_back({ v.second.type_size * v.second.elements, in_p, out_p });
    }
    return scatter(streamResourceId, stream, messageOrAgent, scatterData, itemCount, out_index_offset, invert_scan_flag, scatter_all_count);
}
unsigned int CUDAScatter::scatter(
    const unsigned int &streamResourceId,
    const cudaStream_t &stream,
    const Type &messageOrAgent,
    const std::vector<ScatterData> &sd,
    const unsigned int &itemCount,
    const unsigned int &out_index_offset,
    const bool &invert_scan_flag,
    const unsigned int &scatter_all_count) {
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size
    // calculate the grid block size for main agent function
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scatter_generic<unsigned int*>, 0, itemCount));
    //! Round up according to CUDAAgent state list size
    gridSize = (itemCount + blockSize - 1) / blockSize;
    // Make sure we have enough space to store scatterdata
    streamResources[streamResourceId].resize(static_cast<unsigned int>(sd.size()));
    // Important that sd.size() is still used here, incase allocated len (data_len) is bigger
    gpuErrchk(cudaMemcpyAsync(streamResources[streamResourceId].d_data, sd.data(), sizeof(ScatterData) * sd.size(), cudaMemcpyHostToDevice, stream));
    if (invert_scan_flag) {
        scatter_generic <<<gridSize, blockSize, 0, stream>>> (
            itemCount,
            InversionIterator(scan.Config(messageOrAgent, streamResourceId).d_ptrs.scan_flag),
            scan.Config(messageOrAgent, streamResourceId).d_ptrs.position,
            streamResources[streamResourceId].d_data, static_cast<unsigned int>(sd.size()),
            out_index_offset, scatter_all_count);
    } else {
        scatter_generic <<<gridSize, blockSize, 0, stream>>> (
            itemCount,
            scan.Config(messageOrAgent, streamResourceId).d_ptrs.scan_flag,
            scan.Config(messageOrAgent, streamResourceId).d_ptrs.position,
            streamResources[streamResourceId].d_data, static_cast<unsigned int>(sd.size()),
            out_index_offset, scatter_all_count);
    }
    gpuErrchkLaunch();
    // Update count of live agents
    unsigned int rtn = 0;
    gpuErrchk(cudaMemcpyAsync(&rtn, scan.Config(messageOrAgent, streamResourceId).d_ptrs.position + itemCount - scatter_all_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));  // @todo - async + sync variants.
    return rtn + scatter_all_count;
}
void CUDAScatter::scatterPosition(
    const unsigned int &streamResourceId,
    const cudaStream_t &stream,
    const Type &messageOrAgent,
    const std::vector<ScatterData> &sd,
    const unsigned int &itemCount) {
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size
    // calculate the grid block size for main agent function
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scatter_position_generic, 0, itemCount));
    //! Round up according to CUDAAgent state list size
    gridSize = (itemCount + blockSize - 1) / blockSize;
    // Make sure we have enough space to store scatterdata
    streamResources[streamResourceId].resize(static_cast<unsigned int>(sd.size()));
    // Important that sd.size() is still used here, incase allocated len (data_len) is bigger
    gpuErrchk(cudaMemcpyAsync(streamResources[streamResourceId].d_data, sd.data(), sizeof(ScatterData) * sd.size(), cudaMemcpyHostToDevice, stream));
    scatter_position_generic <<<gridSize, blockSize, 0, stream>>> (
        itemCount,
        scan.Config(messageOrAgent, streamResourceId).d_ptrs.position,
        streamResources[streamResourceId].d_data, static_cast<unsigned int>(sd.size()));
    gpuErrchkLaunch();
    gpuErrchk(cudaStreamSynchronize(stream));  // @todo - async + sync variants.
}
unsigned int CUDAScatter::scatterCount(
    const unsigned int &streamResourceId,
    const cudaStream_t &stream,
    const Type &messageOrAgent,
    const unsigned int &itemCount,
    const unsigned int &scatter_all_count) {
    unsigned int rtn = 0;
    gpuErrchk(cudaMemcpy(&rtn, scan.Config(messageOrAgent, streamResourceId).d_ptrs.position + itemCount - scatter_all_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    return rtn;
}

unsigned int CUDAScatter::scatterAll(
    const unsigned int &streamResourceId,
    const cudaStream_t &stream,
    const std::vector<ScatterData> &sd,
    const unsigned int &itemCount,
    const unsigned int &out_index_offset) {
    if (!itemCount)
        return itemCount;  // No work to do
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size

                       // calculate the grid block size for main agent function
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scatter_all_generic, 0, itemCount));
    //! Round up according to CUDAAgent state list size
    gridSize = (itemCount + blockSize - 1) / blockSize;
    streamResources[streamResourceId].resize(static_cast<unsigned int>(sd.size()));
    // Important that sd.size() is still used here, incase allocated len (data_len) is bigger
    gpuErrchk(cudaMemcpyAsync(streamResources[streamResourceId].d_data, sd.data(), sizeof(ScatterData) * sd.size(), cudaMemcpyHostToDevice, stream));
    scatter_all_generic <<<gridSize, blockSize, 0, stream>>> (
        itemCount,
        streamResources[streamResourceId].d_data, static_cast<unsigned int>(sd.size()),
        out_index_offset);
    gpuErrchkLaunch();
    gpuErrchk(cudaStreamSynchronize(stream));  // @todo - async + sync variants.
    // Update count of live agents
    return itemCount;
}
unsigned int CUDAScatter::scatterAll(
    const unsigned int &streamResourceId,
    const cudaStream_t &stream,
    const VariableMap &vars,
    const std::map<std::string, void*> &in,
    const std::map<std::string, void*> &out,
    const unsigned int &itemCount,
    const unsigned int &out_index_offset) {
    std::vector<ScatterData> scatterData;
    for (const auto &v : vars) {
        char *in_p = reinterpret_cast<char*>(in.at(v.first));
        char *out_p = reinterpret_cast<char*>(out.at(v.first));
        scatterData.push_back({ v.second.type_size * v.second.elements, in_p, out_p });
    }
    return scatterAll(streamResourceId, stream, scatterData, itemCount, out_index_offset);
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
    const unsigned int &streamResourceId,
    const cudaStream_t &stream,
    const VariableMap &vars,
    const std::map<std::string, void*> &in,
    const std::map<std::string, void*> &out,
    const unsigned int &itemCount,
    const unsigned int *d_bin_index,
    const unsigned int *d_bin_sub_index,
    const unsigned int *d_pbm) {
    // If itemCount is 0, then there is no work to be done.
    if (itemCount == 0) {
        return;
    }

    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size

                       // calculate the grid block size for main agent function
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, pbm_reorder_generic, 0, itemCount));
    //! Round up according to CUDAAgent state list size
    gridSize = (itemCount + blockSize - 1) / blockSize;
    // for each variable, scatter from swap to regular
    std::vector<ScatterData> sd;
    for (const auto &v : vars) {
        char *in_p = reinterpret_cast<char*>(in.at(v.first));
        char *out_p = reinterpret_cast<char*>(out.at(v.first));
        sd.push_back({ v.second.type_size * v.second.elements, in_p, out_p });
    }
    streamResources[streamResourceId].resize(static_cast<unsigned int>(sd.size()));
    // Important that sd.size() is still used here, incase allocated len (data_len) is bigger
    gpuErrchk(cudaMemcpyAsync(streamResources[streamResourceId].d_data, sd.data(), sizeof(ScatterData) * sd.size(), cudaMemcpyHostToDevice, stream));
    pbm_reorder_generic <<<gridSize, blockSize, 0, stream>>> (
            itemCount,
            d_bin_index,
            d_bin_sub_index,
            d_pbm,
            streamResources[streamResourceId].d_data, static_cast<unsigned int>(sd.size()));
    gpuErrchkLaunch();
    gpuErrchk(cudaStreamSynchronize(stream));  // @todo - async + sync variants.
}

/**
 * Scatter kernel for host agent creation
 * Input data is stored in AoS, and translated to SoA for device
 * @param threadCount Total number of threads required
 * @param agent_size The total size of an agent's variables in memory, for stepping through input array
 * @param scatter_data Scatter data array location in memory
 * @param scatter_len Length of scatter data array
 * @param out_index_offset The number of agents already in the output array (so that they are not overwritten)
 */
__global__ void scatter_new_agents(
    const unsigned int threadCount,
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
    const unsigned int &streamResourceId,
    const cudaStream_t &stream,
    const std::vector<ScatterData> &sd,
    const size_t &totalAgentSize,
    const unsigned int &inCount,
    const unsigned int &outIndexOffset) {
    // 1 thread per agent variable
    const unsigned int threadCount = static_cast<unsigned int>(sd.size()) * inCount;
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size

    // calculate the grid block size for main agent function
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scatter_new_agents, 0, threadCount));
    //! Round up according to CUDAAgent state list size
    gridSize = (threadCount + blockSize - 1) / blockSize;
    streamResources[streamResourceId].resize(static_cast<unsigned int>(sd.size()));
    // Important that sd.size() is still used here, incase allocated len (data_len) is bigger
    gpuErrchk(cudaMemcpyAsync(streamResources[streamResourceId].d_data, sd.data(), sizeof(ScatterData) * sd.size(), cudaMemcpyHostToDevice, stream));
    scatter_new_agents <<<gridSize, blockSize, 0, stream>>> (
        threadCount,
        static_cast<unsigned int>(totalAgentSize),
        streamResources[streamResourceId].d_data, static_cast<unsigned int>(sd.size()),
        outIndexOffset);
    gpuErrchkLaunch();
    gpuErrchk(cudaStreamSynchronize(stream));  // @todo - async + sync variants.
}
/**
* Broadcast kernel for initialising agent variables to default on device
* Input data is stored pointed directly do by scatter_data and translated to SoA for device
* @param threadCount Total number of threads required
* @param scatter_data Scatter data array location in memory
* @param scatter_len Length of scatter data array
* @param out_index_offset The number of agents already in the output array (so that they are not overwritten)
*/
__global__ void broadcastInitKernel(
    const unsigned int threadCount,
    CUDAScatter::ScatterData *scatter_data,
    const unsigned int scatter_len,
    const unsigned int out_index_offset) {
    // global thread index
    int index = (blockIdx.x*blockDim.x) + threadIdx.x;

    if (index >= threadCount) return;

    // Which variable are we outputting
    const unsigned int var_out = index % scatter_len;
    const unsigned int agent_index = index / scatter_len;
    const unsigned int type_len = scatter_data[var_out].typeLen;
    // if optional message is to be written
    char * const in_ptr = scatter_data[var_out].in;
    char * const out_ptr = scatter_data[var_out].out + ((out_index_offset + agent_index) * type_len);
    memcpy(out_ptr, in_ptr, type_len);
}
void CUDAScatter::broadcastInit(
    const unsigned int &streamResourceId,
    const cudaStream_t &stream,
    const std::list<std::shared_ptr<VariableBuffer>> &vars,
    const unsigned int &inCount,
    const unsigned int outIndexOffset) {
    // No variables means no work to do
    if (vars.size() == 0) return;
    // 1 thread per agent variable
    const unsigned int threadCount = static_cast<unsigned int>(vars.size()) * inCount;
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size

    // calculate the grid block size for main agent function
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, broadcastInitKernel, 0, threadCount));
    //! Round up according to CUDAAgent state list size
    gridSize = (threadCount + blockSize - 1) / blockSize;
    // Calculate memory usage (crudely in multiples of ScatterData)
    ptrdiff_t offset = 0;
    for (const auto &v : vars) {
        offset += v->type_size * v->elements;
    }
    streamResources[streamResourceId].resize(static_cast<unsigned int>(offset + vars.size() * sizeof(ScatterData)));
    // Build scatter data structure and init data
    std::vector<ScatterData> sd;
    char *default_data = reinterpret_cast<char*>(malloc(offset));
    offset = 0;
    for (const auto &v : vars) {
        // Scatter data
        char *in_p = reinterpret_cast<char*>(streamResources[streamResourceId].d_data) + offset;
        char *out_p = reinterpret_cast<char*>(v->data_condition);
        sd.push_back({ v->type_size * v->elements, in_p, out_p });
        // Init data
        memcpy(default_data + offset, v->default_value, v->type_size * v->elements);
        // Update offset
        offset += v->type_size * v->elements;
    }
    // Important that sd.size() is used here, as allocated len would exceed 2nd memcpy
    gpuErrchk(cudaMemcpyAsync(streamResources[streamResourceId].d_data, default_data, offset, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(streamResources[streamResourceId].d_data + offset, sd.data(), sizeof(ScatterData) * sd.size(), cudaMemcpyHostToDevice, stream));
    ::free(default_data);
    broadcastInitKernel <<<gridSize, blockSize, 0, stream>>> (
        threadCount,
        streamResources[streamResourceId].d_data + offset, static_cast<unsigned int>(sd.size()),
        outIndexOffset);
    gpuErrchkLaunch();
    gpuErrchk(cudaStreamSynchronize(stream));  // @todo - async + sync variants.
}
void CUDAScatter::broadcastInit(
    const unsigned int &streamResourceId,
    const cudaStream_t &stream,
    const VariableMap &vars,
    void * const d_newBuff,
    const unsigned int &inCount,
    const unsigned int outIndexOffset) {
    // 1 thread per agent variable
    const unsigned int threadCount = static_cast<unsigned int>(vars.size()) * inCount;
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size

    // calculate the grid block size for main agent function
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, broadcastInitKernel, 0, threadCount));
    //! Round up according to CUDAAgent state list size
    gridSize = (threadCount + blockSize - 1) / blockSize;
    // Calculate memory usage (crudely in multiples of ScatterData)
    std::vector<ScatterData> sd;
    ptrdiff_t offset = 0;
    for (const auto &v : vars) {
        offset += v.second.type_size * v.second.elements;
    }
    char *default_data = reinterpret_cast<char*>(malloc(offset));
    streamResources[streamResourceId].resize(static_cast<unsigned int>(offset + vars.size() * sizeof(ScatterData)));
    // Build scatter data structure
    offset = 0;
    char * d_var = static_cast<char*>(d_newBuff);
    for (const auto &v : vars) {
        // In this case, in is the location of first variable, but we step by inOffsetData.totalSize
        char *in_p = reinterpret_cast<char*>(streamResources[streamResourceId].d_data) + offset;
        char *out_p = d_var;
        sd.push_back({ v.second.type_size * v.second.elements, in_p, out_p });
        // Build init data
        memcpy(default_data + offset, v.second.default_value, v.second.type_size * v.second.elements);
        // Prep pointer for next var
        d_var += v.second.type_size * v.second.elements * inCount;
        // Update offset
        offset += v.second.type_size * v.second.elements;
    }
    // Important that sd.size() is still used here, incase allocated len (data_len) is bigger
    gpuErrchk(cudaMemcpyAsync(streamResources[streamResourceId].d_data, default_data, offset, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(streamResources[streamResourceId].d_data + offset, sd.data(), sizeof(ScatterData) * sd.size(), cudaMemcpyHostToDevice, stream));
    ::free(default_data);
    broadcastInitKernel <<<gridSize, blockSize, 0, stream>>> (
        threadCount,
        streamResources[streamResourceId].d_data + offset, static_cast<unsigned int>(sd.size()),
        outIndexOffset);
    gpuErrchkLaunch();
    gpuErrchk(cudaStreamSynchronize(stream));  // @todo - async + sync variants.
}
__global__ void reorder_array_messages(
    const unsigned int threadCount,
    const unsigned int array_length,
    const unsigned int *d_position,
#if !defined(SEATBELTS) || SEATBELTS
    unsigned int *d_write_flag,
#endif
    CUDAScatter::ScatterData *scatter_data,
    const unsigned int scatter_len
) {
    // global thread index
    int index = (blockIdx.x*blockDim.x) + threadIdx.x;

    if (index >= threadCount) return;

    const unsigned int output_index = d_position[index];
    // If out of bounds, put it in 1 out of bounds slot
    if (output_index < array_length) {
        for (unsigned int i = 0; i < scatter_len; ++i) {
            memcpy(scatter_data[i].out + (output_index * scatter_data[i].typeLen), scatter_data[i].in + (index * scatter_data[i].typeLen), scatter_data[i].typeLen);
        }
#if !defined(SEATBELTS) || SEATBELTS
        // Set err check flag
        atomicInc(d_write_flag + output_index, UINT_MAX);
#endif
    }
}
void CUDAScatter::arrayMessageReorder(
    const unsigned int &streamResourceId,
    const cudaStream_t &stream,
    const VariableMap &vars,
    const std::map<std::string, void*> &in,
    const std::map<std::string, void*> &out,
    const unsigned int &itemCount,
    const unsigned int &array_length,
    unsigned int *d_write_flag) {
    // If itemCount is 0, then there is no work to be done.
    if (itemCount == 0) {
        return;
    }

    if (itemCount > array_length) {
        THROW exception::ArrayMessageWriteConflict("Too many messages output for array message structure (%u > %u).\n", itemCount, array_length);
    }
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size
                       // calculate the grid block size for main agent function
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, reorder_array_messages, 0, itemCount);
    //! Round up according to CUDAAgent state list size
    gridSize = (itemCount + blockSize - 1) / blockSize;
    unsigned int *d_position = nullptr;
    // Build AoS -> AoS list
    std::vector<ScatterData> sd;
    for (const auto &v : vars) {
        if (v.first != "___INDEX") {
            char *in_p = reinterpret_cast<char*>(in.at(v.first));
            char *out_p = reinterpret_cast<char*>(out.at(v.first));
            sd.push_back({ v.second.type_size * v.second.elements, in_p, out_p });
        } else {  // Special case, log index var
            d_position = reinterpret_cast<unsigned int*>(in.at(v.first));
            d_write_flag = d_write_flag ? d_write_flag : reinterpret_cast<unsigned int*>(out.at(v.first));
        }
    }
    assert(d_position);  // Not an array message, lacking ___INDEX var
    size_t t_data_len = 0;
    {  // Decide curve memory requirements
        gpuErrchk(cub::DeviceReduce::Max(nullptr, t_data_len, d_write_flag, d_position, array_length, stream));
        if (t_data_len > streamResources[streamResourceId].data_len * sizeof(ScatterData)) {
            // t_data_len is bigger than current allocation
            if (t_data_len > sd.size() * sizeof(ScatterData)) {
                // td_data_len is bigger than sd.size()
                streamResources[streamResourceId].resize(static_cast<unsigned int>((t_data_len / sizeof(ScatterData)) + 1));
            } else {
                // sd.size() is bigger
                streamResources[streamResourceId].resize(static_cast<unsigned int>(sd.size()));
            }
        }
    }
    // Important that sd.size() is still used here, incase allocated len (data_len) is bigger
    gpuErrchk(cudaMemcpyAsync(streamResources[streamResourceId].d_data, sd.data(), sizeof(ScatterData) * sd.size(), cudaMemcpyHostToDevice, stream));
    reorder_array_messages <<<gridSize, blockSize, 0, stream >>> (
        itemCount, array_length,
        d_position,
#if !defined(SEATBELTS) || SEATBELTS
        d_write_flag,
#endif
        streamResources[streamResourceId].d_data, static_cast<unsigned int>(sd.size()));
    gpuErrchkLaunch();
#if !defined(SEATBELTS) || SEATBELTS
    // Check d_write_flag for dupes
    gpuErrchk(cub::DeviceReduce::Max(streamResources[streamResourceId].d_data, t_data_len, d_write_flag, d_position, array_length, stream));
    unsigned int maxBinSize = 0;
    gpuErrchk(cudaMemcpyAsync(&maxBinSize, d_position, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    if (maxBinSize > 1) {
        // Too many messages for single element of array
        // Report bad ones
        unsigned int *hd_write_flag = (unsigned int *)malloc(sizeof(unsigned int) * array_length);
        gpuErrchk(cudaMemcpy(hd_write_flag, d_write_flag, sizeof(unsigned int)* array_length, cudaMemcpyDeviceToHost));
        for (unsigned int i = 0; i < array_length; ++i) {
            if (hd_write_flag[i] > 1)
                fprintf(stderr, "Array messagelist contains %u messages at index %u!\n", hd_write_flag[i], i);
        }
        THROW exception::ArrayMessageWriteConflict("Multiple threads output array messages to the same index, see stderr.\n");
    }
#endif
}

}  // namespace flamegpu
