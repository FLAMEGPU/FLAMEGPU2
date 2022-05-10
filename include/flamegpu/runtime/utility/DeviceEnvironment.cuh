#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_DEVICEENVIRONMENT_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_DEVICEENVIRONMENT_CUH_

// #include <cuda_runtime.h>
#include <string>
#include <cassert>

#include "flamegpu/runtime/utility/DeviceMacroProperty.cuh"
#include "flamegpu/util/type_decode.h"
#ifndef __CUDACC_RTC__
#include "flamegpu/runtime/detail/curve/DeviceCurve.cuh"
#endif

namespace flamegpu {

/**
 * Utility for accessing environmental properties
 * These can only be read within agent functions
 * They can be set and updated within host functions
 */
class ReadOnlyDeviceEnvironment {
    /**
     * Constructs the object
     */
    friend class ReadOnlyDeviceAPI;

 public:
    /**
     * Gets an environment property
     * @param name name used for accessing the property, this value should be a string literal e.g. "foobar"
     * @tparam T Type of the environment property being accessed
     * @tparam M Length of property name, this should always be implicit if passing a string literal
     * @throws exception::DeviceError If name is not a valid property within the environment (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If T is not the type of the environment property specified by name (flamegpu must be built with SEATBELTS enabled for device error checking)
     */
    template<typename T, unsigned int M>
    __device__ __forceinline__ T getProperty(const char(&name)[M]) const;
    /**
     * Gets an element of an environment property array
     * @param name name used for accessing the property, this value should be a string literal e.g. "foobar"
     * @param index Index of the element within the environment property array to return
     * @tparam T Type of the environment property being accessed
     * @tparam N Length of the environment property array
     * @tparam M Length of property name, this should always be implicit if passing a string literal
     * @throws exception::DeviceError If name is not a valid property within the environment (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If T is not the type of the environment property specified by name (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If index is out of bounds for the environment property array specified by name (flamegpu must be built with SEATBELTS enabled for device error checking)
     */
    template<typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ T getProperty(const char(&name)[M], const unsigned int&index) const;
    /**
     * Returns a read-only accessor to the named macro property
     * @param name name used for accessing the property, this value should be a string literal e.g. "foobar"
     * @tparam I Length of macro property in the 1st dimension, default 1
     * @tparam J Length of macro property in the 2nd dimension, default 1
     * @tparam K Length of macro property in the 3rd dimension, default 1
     * @tparam W Length of macro property in the 4th dimension, default 1
     * @tparam M Length of variable name, this should always be implicit if passing a string literal
     */
    template<typename T, unsigned int I = 1, unsigned int J = 1, unsigned int K = 1, unsigned int W = 1, unsigned int M>
    __device__ __forceinline__ ReadOnlyDeviceMacroProperty<T, I, J, K, W> getMacroProperty(const char(&name)[M]) const;
};
/**
 * Utility for accessing environmental properties
 * These can only be read within agent functions
 * They can be set and updated within host functions
 * This version also allows limited write access to device macro properties
 */
class DeviceEnvironment : public ReadOnlyDeviceEnvironment {
 public:
    /**
     * Returns a read-only accessor to the named macro property
     * @param name name used for accessing the property, this value should be a string literal e.g. "foobar"
     * @tparam I Length of macro property in the 1st dimension, default 1
     * @tparam J Length of macro property in the 2nd dimension, default 1
     * @tparam K Length of macro property in the 3rd dimension, default 1
     * @tparam W Length of macro property in the 4th dimension, default 1
     * @tparam M Length of variable name, this should always be implicit if passing a string literal
     */
    template<typename T, unsigned int I = 1, unsigned int J = 1, unsigned int K = 1, unsigned int W = 1, unsigned int M>
    __device__ __forceinline__ DeviceMacroProperty<T, I, J, K, W> getMacroProperty(const char(&name)[M]) const;
};

// Mash compilation of these functions from RTC builds as this requires a dynamic implementation of the function in curve_rtc
#ifndef __CUDACC_RTC__
/**
 * Getters
 */
template<typename T, unsigned int M>
__device__ __forceinline__ T ReadOnlyDeviceEnvironment::getProperty(const char(&name)[M]) const {
    return detail::curve::DeviceCurve::getEnvironmentProperty<T>(name);
}
template<typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T ReadOnlyDeviceEnvironment::getProperty(const char(&name)[M], const unsigned int &index) const {
    return detail::curve::DeviceCurve::getEnvironmentArrayProperty<T, N>(name,  index);
}

template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W, unsigned int N>
__device__ __forceinline__ ReadOnlyDeviceMacroProperty<T, I, J, K, W> ReadOnlyDeviceEnvironment::getMacroProperty(const char(&name)[N]) const {
    char * d_ptr = detail::curve::DeviceCurve::getEnvironmentMacroProperty<T, I, J, K, W>(name);
#if !defined(SEATBELTS) || SEATBELTS
    if (!d_ptr) {
        return ReadOnlyDeviceMacroProperty<T, I, J, K, W>(nullptr, nullptr);
    }
    return ReadOnlyDeviceMacroProperty<T, I, J, K, W>(reinterpret_cast<T*>(d_ptr),
        reinterpret_cast<unsigned int*>(d_ptr + (I * J * K * W * sizeof(T))));  // Read-write flag resides in 8 bits at the end of the buffer
#else

    return ReadOnlyDeviceMacroProperty<T, I, J, K, W>(reinterpret_cast<T*>(d_ptr));
#endif
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W, unsigned int N>
__device__ __forceinline__ DeviceMacroProperty<T, I, J, K, W> DeviceEnvironment::getMacroProperty(const char(&name)[N]) const {
    char* d_ptr = detail::curve::DeviceCurve::getEnvironmentMacroProperty<T, I, J, K, W>(name);
#if !defined(SEATBELTS) || SEATBELTS
    if (!d_ptr) {
        return DeviceMacroProperty<T, I, J, K, W>(nullptr, nullptr);
    }
    return DeviceMacroProperty<T, I, J, K, W>(reinterpret_cast<T*>(d_ptr),
        reinterpret_cast<unsigned int*>(d_ptr + (I * J * K * W * sizeof(T))));  // Read-write flag resides in 8 bits at the end of the buffer
#else
    return DeviceMacroProperty<T, I, J, K, W>(reinterpret_cast<T*>(d_ptr));
#endif
}
#endif  // __CUDACC_RTC__

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_DEVICEENVIRONMENT_CUH_
