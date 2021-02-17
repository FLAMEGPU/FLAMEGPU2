#include <cassert>

#include "flamegpu/gpu/CUDAScanCompaction.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/gpu/CUDASimulation.h"

/**
 * CUDAScanCompaction methods
 */
void CUDAScanCompaction::purge() {
    memset(configs, 0, sizeof(configs));
}

void CUDAScanCompaction::resize(const unsigned int& newCount, const Type& type, const unsigned int& streamId) {
    assert(streamId < MAX_STREAMS);
    assert(type < MAX_TYPES);
    configs[type][streamId].resize_scan_flag(newCount);
}

void CUDAScanCompaction::zero(const Type& type, const unsigned int& streamId) {
    assert(streamId < MAX_STREAMS);
    assert(type < MAX_TYPES);
    configs[type][streamId].zero_scan_flag();
}

const CUDAScanCompactionConfig &CUDAScanCompaction::getConfig(const Type& type, const unsigned int& streamId) {
    return configs[type][streamId];
}
CUDAScanCompactionConfig &CUDAScanCompaction::Config(const Type& type, const unsigned int& streamId) {
    return configs[type][streamId];
}
/**
 *
 */
CUDAScanCompactionConfig::~CUDAScanCompactionConfig() {
    free_scan_flag();
}
void CUDAScanCompactionConfig::free_scan_flag() {
    if (d_ptrs.scan_flag) {
        gpuErrchk(cudaFree(d_ptrs.scan_flag));
        d_ptrs.scan_flag = nullptr;
    }
    if (d_ptrs.position) {
        gpuErrchk(cudaFree(d_ptrs.position));
        d_ptrs.position = nullptr;
    }
}

void CUDAScanCompactionConfig::zero_scan_flag() {
    if (d_ptrs.position) {
        gpuErrchk(cudaMemset(d_ptrs.position, 0, scan_flag_len * sizeof(unsigned int)));  // @todo - make this async + streamSync for less ensbemble blocking.
    }
    if (d_ptrs.scan_flag) {
        gpuErrchk(cudaMemset(d_ptrs.scan_flag, 0, scan_flag_len * sizeof(unsigned int)));  // @todo - make this async + streamSync for less ensbemble blocking.
    }
}

void CUDAScanCompactionConfig::resize_scan_flag(const unsigned int& count) {
    if (count + 1 > scan_flag_len) {
        free_scan_flag();
        gpuErrchk(cudaMalloc(&d_ptrs.scan_flag, (count + 1) * sizeof(unsigned int)));  // +1 so we can get the total from the scan
        gpuErrchk(cudaMalloc(&d_ptrs.position, (count + 1) * sizeof(unsigned int)));  // +1 so we can get the total from the scan
        scan_flag_len = count + 1;
    }
}
