#ifndef INCLUDE_FLAMEGPU_RUNTIME_ENVIRONMENT_HOSTENVIRONMENT_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_ENVIRONMENT_HOSTENVIRONMENT_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>  // Required for FLAMEGPU_SEATBELTS=OFF builds for some reason.

#include <unordered_map>
#include <array>
#include <string>
#include <utility>
#include <set>
#include <vector>
#include <memory>

#include "flamegpu/simulation/detail/CUDAMacroEnvironment.h"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"
#include "flamegpu/simulation/detail/EnvironmentManager.cuh"
#include "flamegpu/runtime/environment/HostMacroProperty.cuh"

namespace flamegpu {

/**
 * This class provides host function access to Environment Properties
 * It acts as a wrapper to EnvironmentManager, proxying calls, converting variable name and model_name into a combined hash
 * Pairs with EnvironmentManager, AgentEnvironment and EnvironmentDescription
 * This class is only to be constructed by HostAPI
 * @note Not thread-safe
 */
class HostEnvironment {
    /**
     * This class can only be constructed by HostAPI
     */
    friend class HostAPI;

 protected:
    /**
     * Constructor, to be called by HostAPI
     */
    explicit HostEnvironment(unsigned int instance_id, const std::shared_ptr<detail::EnvironmentManager> &env, detail::CUDAMacroEnvironment &_macro_env);
    /**
     * Provides access to EnvironmentManager singleton
     */
    const std::shared_ptr<detail::EnvironmentManager> env_mgr;
    /**
     * Provides access to macro properties for the instance
     */
    detail::CUDAMacroEnvironment& macro_env;
    /**
     * Access to instance id of the CUDASimulation
     * This is used to augment all variable names
     */
    const unsigned int instance_id;

 public:
    /**
     * Gets an environment property
     * @param name name used for accessing the property
     * @tparam T Type of the environment property
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     */
    template<typename T>
    T getProperty(const std::string &name) const;
    /**
     * Gets an environment property array
     * @param name name used for accessing the property
     * @tparam T Type of the elements of the environment property array
     * @tparam N Length of the environment property array
     * @throws exception::InvalidEnvProperty If a property array of the name does not exist
     */
    template<typename T, flamegpu::size_type N>
    std::array<T, N> getProperty(const std::string &name) const;
    /**
     * Gets an element of an environment property array
     * @param name name used for accessing the property
     * @param index Index of the element within the environment property array to return
     * @tparam T Type of the elements of the environment property array
     * @tparam N (Optional) The length of the array variable, available for parity with other APIs, checked if provided
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     * @see get(const std::string &)
     */
    template<typename T, flamegpu::size_type N = 0>
    T getProperty(const std::string &name, flamegpu::size_type index) const;
#ifdef SWIG
    /**
     * Gets an environment property array
     * @param name name used for accessing the property
     * @tparam T Type of the elements of the environment property array
     * @throws exception::InvalidEnvProperty If a property array of the name does not exist
     */
    template<typename T>
    std::vector<T> getPropertyArray(const std::string & name) const;
#endif
    /**
     * Sets an environment property
     * @param name name used for accessing the property
     * @param value to set the property
     * @tparam T Type of the elements of the environment property
     * @return Returns the previous value
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::ReadOnlyEnvProperty If the named property is marked as const
     */
    template<typename T>
    T setProperty(const std::string &name, T value) const;
    /**
     * Sets an environment property array
     * @param name name used for accessing the property array
     * @param value to set the property array
     * @tparam T Type of the elements of the environment property array
     * @tparam N Length of the environmental property array
     * @return Returns the previous value
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::ReadOnlyEnvProperty If the named property is marked as const
     */
    template<typename T, flamegpu::size_type N>
    std::array<T, N> setProperty(const std::string &name, const std::array<T, N> &value) const;
    /**
     * Sets an element of an environment property array
     * @param name name used for accessing the property array
     * @param index element within the environment property array to set
     * @param value to set the element of the property array
     * @tparam T Type of the environmental property array
     * @tparam N (Optional) The length of the array variable, available for parity with other APIs, checked if provided
     * @return Returns the previous value
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     * @see get(const std::string &)
     */
    template<typename T, flamegpu::size_type N = 0>
    T setProperty(const std::string &name, flamegpu::size_type index, T value) const;
#ifdef SWIG
    /**
     * Sets an environment property array
     * @param name name used for accessing the property array
     * @param value to set the property array
     * @tparam T Type of the elements of the environment property array
     * @return Returns the previous value
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::ReadOnlyEnvProperty If the named property is marked as const
     */
    template<typename T>
    std::vector<T> setPropertyArray(const std::string & name, const std::vector<T> & value) const;
#endif
    /**
     * Returns an interface for accessing the named host macro property
     * @param name The name of the environment macro property to return
     */
    template<typename T, unsigned int I = 1, unsigned int J = 1, unsigned int K = 1, unsigned int W = 1>
    HostMacroProperty<T, I, J, K, W> getMacroProperty(const std::string& name) const;
#ifdef SWIG
    /**
     * None-templated dimensions version of getMacroProperty() for SWIG interface
     * @param name The name of the environment macro property to return
     */
    template<typename T>
    HostMacroProperty_swig<T> getMacroProperty_swig(const std::string& name) const;
#endif
};

/**
 * Setters
 */
template<typename T>
T HostEnvironment::setProperty(const std::string &name, const T value) const {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in HostEnvironment::set().");
    }
    return env_mgr->setProperty<T>(name, value);
}
template<typename T, flamegpu::size_type N>
std::array<T, N> HostEnvironment::setProperty(const std::string &name, const std::array<T, N> &value) const {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in HostEnvironment::set().");
    }
    return env_mgr->setProperty<T, N>(name, value);
}
template<typename T, flamegpu::size_type N>
T HostEnvironment::setProperty(const std::string &name, flamegpu::size_type index, const T value) const {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in HostEnvironment::set().");
    }
    return env_mgr->setProperty<T, N>(name, index, value);
}
#ifdef SWIG
template<typename T>
std::vector<T> HostEnvironment::setPropertyArray(const std::string &name, const std::vector<T> &value) const {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in HostEnvironment::setArray().");
    }
    return env_mgr->setPropertyArray<T>(name, value);
}
#endif  // SWIG

/**
 * Getters
 */
template<typename T>
T HostEnvironment::getProperty(const std::string &name) const  {
    return env_mgr->getProperty<T>(name);
}
template<typename T, flamegpu::size_type N>
std::array<T, N> HostEnvironment::getProperty(const std::string &name) const  {
    return env_mgr->getProperty<T, N>(name);
}
template<typename T, flamegpu::size_type N>
T HostEnvironment::getProperty(const std::string &name, const flamegpu::size_type index) const  {
    return env_mgr->getProperty<T, N>(name, index);
}
#ifdef SWIG
template<typename T>
std::vector<T> HostEnvironment::getPropertyArray(const std::string& name) const {
    return env_mgr->getPropertyArray<T>(name);
}
#endif  // SWIG

template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W>
HostMacroProperty<T, I, J, K, W> HostEnvironment::getMacroProperty(const std::string& name) const {
    return macro_env.getProperty<T, I, J, K, W>(name);
}

#ifdef SWIG
template<typename T>
HostMacroProperty_swig<T> HostEnvironment::getMacroProperty_swig(const std::string& name) const {
    return macro_env.getProperty_swig<T>(name);
}
#endif
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_ENVIRONMENT_HOSTENVIRONMENT_CUH_
