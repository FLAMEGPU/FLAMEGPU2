#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTENVIRONMENT_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTENVIRONMENT_CUH_

#include <cuda_runtime.h>

#include <unordered_map>
#include <array>
#include <string>
#include <utility>
#include <set>

#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/runtime/utility/EnvironmentManager.cuh"

/**
 * This class provides host function access to Environment Properties
 * It acts as a wrapper to EnvironmentManager, proxying calls, converting variable name and model_name into a combined hash
 * Pairs with EnvironmentManager, AgentEnvironment and EnvironmentDescription
 * This class is only to be constructed by FLAMEGPU_HOST_API
 * @note Not thread-safe
 */
class HostEnvironment {
    /**
     * This class can only be constructed by FLAMEGPU_HOST_API
     */
    friend class FLAMEGPU_HOST_API;

 protected:
    /**
     * Constructor, to be called by FLAMEGPU_HOST_API
     */
    explicit HostEnvironment(const std::string &model_name);
    /**
     * Provides access to EnvironmentManager singleton
     */
    EnvironmentManager &env_mgr;
    /**
     * Access to name of the model
     * This is used to augment all variable names
     */
    const std::string model_name;

 public:
    /**
     * Gets an environment property
     * @param name name used for accessing the property
     * @tparam T Type of the environment property
     * @throws InvalidEnvProperty If a property of the name does not exist
     */
    template<typename T>
    T get(const std::string &name) const;
    /**
     * Gets an environment property array
     * @param name name used for accessing the property
     * @tparam T Type of the elements of the environment property array
     * @throws InvalidEnvProperty If a property array of the name does not exist
     */
    template<typename T, EnvironmentManager::size_type N>
    std::array<T, N> get(const std::string &name) const;
    /**
     * Gets an element of an environment property array
     * @param name name used for accessing the property
     * @tparam T Type of the elements of the environment property array
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     * @see get(const std::string &)
     */
    template<typename T>
    T get(const std::string &name, const EnvironmentManager::size_type &index) const;
    /**
     * Sets an environment property
     * @param name name used for accessing the property
     * @param value to set the property
     * @tparam T Type of the elements of the environment property
     * @return Returns the previous value
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws ReadOnlyEnvProperty If the named property is marked as const
     */
    template<typename T>
    T set(const std::string &name, const T &value) const;
    /**
     * Sets an environment property array
     * @param name name used for accessing the property array
     * @param value to set the property array
     * @tparam T Type of the elements of the environment property array
     * @return Returns the previous value
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws ReadOnlyEnvProperty If the named property is marked as const
     */
    template<typename T, EnvironmentManager::size_type N>
    std::array<T, N> set(const std::string &name, const std::array<T, N> &value) const;
    /**
     * Sets an element of an environment property array
     * @param name name used for accessing the property array
     * @param index element within the environemtn property array to set
     * @param value to set the element of the property array
     * @tparam T Type of the environmental property array
     * @return Returns the previous value
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     * @see get(const std::string &)
     */
    template<typename T>
    T set(const std::string &name, const EnvironmentManager::size_type &index, const T &value) const;
};

/**
 * Setters
 */
template<typename T>
T HostEnvironment::set(const std::string &name, const T &value) const {
    if (!name.empty() && name[0] == '_') {
        THROW ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in HostEnvironment::set().");
    }
    return env_mgr.set<T>({ model_name, name }, value);
}
template<typename T, EnvironmentManager::size_type N>
std::array<T, N> HostEnvironment::set(const std::string &name, const std::array<T, N> &value) const {
    if (!name.empty() && name[0] == '_') {
        THROW ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in HostEnvironment::set().");
    }
    return env_mgr.set<T, N>({ model_name, name }, value);
}
template<typename T>
T HostEnvironment::set(const std::string &name, const EnvironmentManager::size_type &index, const T &value) const {
    if (!name.empty() && name[0] == '_') {
        THROW ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in HostEnvironment::set().");
    }
    return env_mgr.set<T>({ model_name, name }, index, value);
}

/**
 * Getters
 */
template<typename T>
T HostEnvironment::get(const std::string &name) const  {
    return env_mgr.get<T>({ model_name, name });
}
template<typename T, EnvironmentManager::size_type N>
std::array<T, N> HostEnvironment::get(const std::string &name) const  {
    return env_mgr.get<T, N>({ model_name, name });
}
template<typename T>
T HostEnvironment::get(const std::string &name, const EnvironmentManager::size_type &index) const  {
    return env_mgr.get<T>({ model_name, name }, index);
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_HOSTENVIRONMENT_CUH_
