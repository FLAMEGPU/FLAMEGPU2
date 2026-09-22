#ifndef INCLUDE_FLAMEGPU_DETAIL_GPU_DEVICE_NAME_HPP_
#define INCLUDE_FLAMEGPU_DETAIL_GPU_DEVICE_NAME_HPP_

#include <set>
#include <string>

namespace flamegpu {
namespace detail {
namespace gpu {

/**
 * Get the device name reported by CUDA runtime API
 * @param deviceIndex the index of the device to be queried
 * @return string representing the device name E.g. "NVIDIA GeForce RTX3080", "Radeon RX 7900 XTX"
 */
const std::string getDeviceName(int deviceIndex);

/**
 * Get the device names reported by CUDA runtime API
 * @param set<int> of device id's to be queried
 * @return comma seperated string of device names E.g. "NVIDIA GeForce RTX3080, NVIDIAGe Force RTX3070", "Radeon RX 7900 XTX, Radeon RX 7900 XTX"
 */
const std::string getDeviceNames(std::set<int> devices);

}  // namespace gpu
}  // namespace detail
}  // namespace flamegpu


#endif  // INCLUDE_FLAMEGPU_DETAIL_GPU_DEVICE_NAME_HPP_
