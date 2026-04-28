#include "flamegpu/simulation/detail/DeviceStrings.h"

#include <string>

#include "flamegpu/detail/gpu/macros.hpp"
#include "flamegpu/detail/cuda.cuh"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"

namespace flamegpu {
namespace detail {

DeviceStrings::~DeviceStrings() {
    flamegpu::detail::gpuCheck(detail::cuda::cudaFree(device_buffer));
}
void DeviceStrings::registerDeviceString(const std::string &host_string) {
    if (offsets.find(host_string) == offsets.end()) {
        offsets.emplace(host_string, host_buffer.size());
        host_stream << host_string;
        host_stream << '\0';  // Each string requires a null terminating char
        host_buffer = host_stream.str();
    }
}
const char* DeviceStrings::getDeviceString(const std::string &host_string) {
    if (offsets.find(host_string) == offsets.end()) {
        registerDeviceString(host_string);
    }
    const size_t host_buffer_len = host_buffer.size();
    const ptrdiff_t device_string_offset = offsets.at(host_string);
    // Reallocate device buffer if necessary
    if (!device_buffer || device_buffer_len < host_buffer_len) {
        // Double buffer len in size
        device_buffer_len = device_buffer_len == 0 ? 1024 : device_buffer_len * 2;
        flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(Free)(device_buffer));
        flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(Malloc)(&device_buffer, device_buffer_len));
        device_buffer_occupied = 0;
    }
    // Update device buffer if necessary
    if (device_buffer_occupied < host_buffer_len) {
        flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(Memcpy)(device_buffer, host_buffer.c_str(), host_buffer_len, FLAMEGPU_GPU_RUNTIME_SYMBOL(MemcpyHostToDevice)));
        device_buffer_occupied = host_buffer_len;
    }
    // Return
    return device_buffer + device_string_offset;
}

}  // namespace detail
}  // namespace flamegpu
