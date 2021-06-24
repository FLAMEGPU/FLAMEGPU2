#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTENVIRONMENT_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTENVIRONMENT_CUH_

#include <cuda_runtime.h>

#include <unordered_map>
#include <array>
#include <string>
#include <utility>
#include <set>
#include <vector>

#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/runtime/utility/EnvironmentManager.cuh"

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
    explicit HostEnvironment(const unsigned int &instance_id);
    /**
     * Provides access to EnvironmentManager singleton
     */
    EnvironmentManager &env_mgr;
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
    template<typename T, EnvironmentManager::size_type N>
    std::array<T, N> getProperty(const std::string &name) const;
    /**
     * Gets an element of an environment property array
     * @param name name used for accessing the property
     * @param index Index of the element within the environment property array to return
     * @tparam T Type of the elements of the environment property array
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     * @see get(const std::string &)
     */
    template<typename T>
    T getProperty(const std::string &name, const EnvironmentManager::size_type &index) const;
#ifdef SWIG
    /**
     * Gets an environment property array
     * @param name name used for accessing the property
     * @tparam T Type of the elements of the environment property array
     * @throws exception::InvalidEnvProperty If a property array of the name does not exist
     */
    template<typename T>
    std::vector<T> getPropertyArray(const std::string &name) const;
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
    T setProperty(const std::string &name, const T &value) const;
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
    template<typename T, EnvironmentManager::size_type N>
    std::array<T, N> setProperty(const std::string &name, const std::array<T, N> &value) const;
    /**
     * Sets an element of an environment property array
     * @param name name used for accessing the property array
     * @param index element within the environment property array to set
     * @param value to set the element of the property array
     * @tparam T Type of the environmental property array
     * @return Returns the previous value
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     * @see get(const std::string &)
     */
    template<typename T>
    T setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value) const;
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
    std::vector<T> setPropertyArray(const std::string &name, const std::vector<T> &value) const;
#endif
};

/**
 * Setters
 */
template<typename T>
T HostEnvironment::setProperty(const std::string &name, const T &value) const {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in HostEnvironment::set().");
    }
    return env_mgr.setProperty<T>({ instance_id, name }, value);
}
template<typename T, EnvironmentManager::size_type N>
std::array<T, N> HostEnvironment::setProperty(const std::string &name, const std::array<T, N> &value) const {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in HostEnvironment::set().");
    }
    return env_mgr.setProperty<T, N>({ instance_id, name }, value);
}
template<typename T>
T HostEnvironment::setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value) const {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in HostEnvironment::set().");
    }
    return env_mgr.setProperty<T>({ instance_id, name }, index, value);
}
#ifdef SWIG
template<typename T>
std::vector<T> HostEnvironment::setPropertyArray(const std::string &name, const std::vector<T> &value) const {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in HostEnvironment::setArray().");
    }
    return env_mgr.setPropertyArray<T>({ instance_id, name }, value);
}
#endif  // SWIG

/**
 * Getters
 */
template<typename T>
T HostEnvironment::getProperty(const std::string &name) const  {
    return env_mgr.getProperty<T>({ instance_id, name });
}
template<typename T, EnvironmentManager::size_type N>
std::array<T, N> HostEnvironment::getProperty(const std::string &name) const  {
    return env_mgr.getProperty<T, N>({ instance_id, name });
}
template<typename T>
T HostEnvironment::getProperty(const std::string &name, const EnvironmentManager::size_type &index) const  {
    return env_mgr.getProperty<T>({ instance_id, name }, index);
}
#ifdef SWIG
template<typename T>
std::vector<T> HostEnvironment::getPropertyArray(const std::string& name) const {
    return env_mgr.getPropertyArray<T>({instance_id, name});
}
#endif  // SWIG

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTENVIRONMENT_CUH_
