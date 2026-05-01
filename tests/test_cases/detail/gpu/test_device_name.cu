#include <algorithm>
#include <set>
#include <string>
#include <ranges>
#include <vector>
#include "flamegpu/detail/gpu/macros.hpp"
#include "flamegpu/detail/gpu/gpu_api_error_checking.cuh"
#include "flamegpu/detail/gpu/device_name.hpp"

#include "gtest/gtest.h"

namespace flamegpu {

TEST(TestDetailGPUDeviceName, getDeviceName) {
    // Get the number of devices so testing is accurate, assume it must be >= 0 else other tests will be failing
    int deviceCount = 0;
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(GetDeviceCount)(&deviceCount));

    // Expect an exception if a negative device name is used
    EXPECT_THROW(flamegpu::detail::gpu::getDeviceName(-1), flamegpu::exception::InvalidCUDAdevice);

    // Expect an exception if the value is more than the number of devices
    EXPECT_THROW(flamegpu::detail::gpu::getDeviceName(deviceCount + 1), flamegpu::exception::InvalidCUDAdevice);

    // Expect a non-empty string on success (rather than strictly checking the expected value) for the 0th device (which we assume exists)
    std::string name = flamegpu::detail::gpu::getDeviceName(0);
    EXPECT_FALSE(name.empty());
}

TEST(TestDetailGPUDeviceName, getDeviceNames) {
    // The returned value will depend on the number of devices.
    int deviceCount = 0;
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(GetDeviceCount)(&deviceCount));

    std::string names = "";

    // With no devices passed, expect a string that contains atleast deviceCount - 1 commas (in case any device names ever include a comma)
    names = flamegpu::detail::gpu::getDeviceNames({});
    EXPECT_FALSE(names.empty());
    EXPECT_GE(std::count(names.begin(), names.end(), ','), deviceCount - 1);

    // Expect an exception if the set of device indeices contains any negative values
    EXPECT_THROW(flamegpu::detail::gpu::getDeviceNames({-1, 0}), flamegpu::exception::InvalidCUDAdevice);

    // Expect an exception if the set of device indices contains any values >= the number of devices
    EXPECT_THROW(flamegpu::detail::gpu::getDeviceNames({0, deviceCount}), flamegpu::exception::InvalidCUDAdevice);

    // For 1 device, there should be a non empty string containing >= 0 commas
    names = flamegpu::detail::gpu::getDeviceNames({0});
    EXPECT_GE(std::count(names.begin(), names.end(), ','), 0);
    // For more than one device, there should be a non empty string contianing >= devices commas (if there are more than 1 device available)
    if (deviceCount > 1) {
        auto range = std::views::iota(0, deviceCount);
        std::set<int> indices(range.begin(), range.end());
        names = flamegpu::detail::gpu::getDeviceNames(indices);
        EXPECT_GE(std::count(names.begin(), names.end(), ','), deviceCount - 1);
    }
}

}  // namespace flamegpu
