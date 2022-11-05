#ifndef INCLUDE_FLAMEGPU_SIM_RUNPLAN_H_
#define INCLUDE_FLAMEGPU_SIM_RUNPLAN_H_

#include <unordered_map>
#include <string>
#include <typeinfo>
#include <vector>
#include <memory>

#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/util/Any.h"
#include "flamegpu/util/type_decode.h"
#include "flamegpu/gpu/CUDAEnsemble.h"


namespace flamegpu {

class ModelDescription;
class RunPlanVector;
class CUDASimulation;

namespace io {
class JSONLogger;
class XMLLogger;
}  // namespace io

/**
 * Individual run config
 */
class RunPlan {
    friend class RunPlanVector;
    friend class SimRunner;
    friend class CUDASimulation;
    friend class io::JSONLogger;
    friend class io::XMLLogger;

 public:
    /**
     * Constructor this will need to set the random seed to a default value, +1 of previous EnsembleRun would make sense
     * @param environment EnvironmentDescription to limit properties by
     * @todo Where will default steps and random seed come from?
     */
    explicit RunPlan(const ModelDescription &environment);
    RunPlan& operator=(const RunPlan& other);

    /**
     * Set the random seed passed to this run of the simulation
     * @param random_seed Seed for random generation during execution
     */
    void setRandomSimulationSeed(uint64_t random_seed);
    /**
     * Set the number of steps for this instance of the simulation
     * A steps value of 0 requires the ModelDescription to have atleast 1 exit condition
     * @param steps The number of steps to execute, 0 is unlimited but requires an exit condition
     */
    void setSteps(unsigned int steps);
    /**
     * Set the sub directory within the output directory for outputs of this run
     * If left empty, output will not goto subdirectories
     * @param subdir The subdirectory to output logfiles for this run to
     */
    void setOutputSubdirectory(const std::string &subdir);
    /**
     * Set the environment property override for this run of the model
     * @param name Environment property name
     * @param value Environment property value (override)
     * @tparam T Type of the environment property
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T
     */
    template<typename T>
    void setProperty(const std::string &name, T value);
    /**
     * Set the environment property override for this run of the model
     * This version should be used for array properties
     * @param name Environment property name
     * @param value Environment property value (override)
     * @tparam T Type of the environment property
     * @tparam N Length of the array to be returned
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T, or length to N
     */
    template<typename T, flamegpu::size_type N>
    void setProperty(const std::string &name, const std::array<T, N> &value);
    /**
     * Set the environment property override for this run of the model
     * This version should be used for setting individual elements of array properties
     * @param name Environment property name
     * @param index Length of the array to be returned
     * @param value Environment property value (override)
     * @tparam T Type of the environment property
     * @tparam N (Optional) Length of the array property
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T
     * @throws exception::OutOfBoundsException If index is not in range of the length of the property array
     */
    template<typename T, flamegpu::size_type N = 0>
    void setProperty(const std::string &name, flamegpu::size_type index, T value);
#ifdef SWIG
    /**
     * Set the environment property override for this run of the model
     * This version should be used for array properties
     * @param name Environment property name
     * @param value Environment property value (override)
     * @tparam T Type of the environment property
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvProperty If value.size() != length
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T, or length to N
     */
    template<typename T>
    void setPropertyArray(const std::string &name, const std::vector<T> &value);
#endif
    /**
     * Returns the random seed used for this simulation run
     */
    uint64_t getRandomSimulationSeed() const;
    /**
     * Returns the number of steps to be executed for this simulation run
     * 0 means unlimited, and is only available if the model description has an exit condition
     */
    unsigned int getSteps() const;
    /**
     * Returns the currently configured output subdirectory directory
     * Empty string means output for this run will not be placed into a subdirectory
     */
    std::string getOutputSubdirectory() const;

    /**
     * Gets the currently configured environment property value
     * @param name name used for accessing the property
     * @tparam T Type of the value to be returned
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T
     */
    template<typename T>
    T getProperty(const std::string &name) const;
    /**
     * Gets the currently configured environment property array value
     * @param name name used for accessing the property
     * @tparam T Type of the value to be returned
     * @tparam N Length of the array to be returned
     * @throws exception::InvalidEnvProperty If a property array of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T
     */
    template<typename T, flamegpu::size_type N>
    std::array<T, N> getProperty(const std::string &name) const;
    /**
     * Gets an element of the currently configured environment property array
     * @param name name used for accessing the property
     * @param index element from the environment property array to return
     * @tparam T Type of the value to be returned
     * @tparam N (Optional) Length of the array property
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T
     * @throws exception::OutOfBoundsException If index is not in range of the length of the property array
     */
    template<typename T, flamegpu::size_type N = 0>
    T getProperty(const std::string &name, flamegpu::size_type index) const;
#ifdef SWIG
    /**
     * Gets the currently configured environment property array value
     * @param name Environment property name
     * @tparam T Type of the environment property
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T
     */
    template<typename T>
    std::vector<T> getPropertyArray(const std::string &name);
#endif

    /**
     * Operator methods for combining vectors
     */
    RunPlanVector operator+(const RunPlan& rhs) const;
    RunPlanVector operator+(const RunPlanVector& rhs) const;
    RunPlanVector operator*(unsigned int rhs) const;

 private:
    explicit RunPlan(const std::shared_ptr<const std::unordered_map<std::string, EnvironmentDescription::PropData>> &environment, bool allow_0);
    uint64_t random_seed;
    unsigned int steps;
    std::string output_subdirectory;
    std::unordered_map<std::string, util::Any> property_overrides;
    /**
     * Reference to model environment data, for validation
     */
    // This needs to be shared_ptr, reference goes out of scope, otherwise have a copy of the map per RunPlan
    std::shared_ptr<const std::unordered_map<std::string, EnvironmentDescription::PropData>> environment;
    /**
     * This flag denotes whether 0 steps are permitted
     */
    bool allow_0_steps;
};

template<typename T>
void RunPlan::setProperty(const std::string &name, T value) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(typename type_decode<T>::type_t)).name());
    }
    if (it->second.data.elements != type_decode<T>::len_t) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' is an array with %u elements, array method should be used, "
            "in RunPlan::setProperty()\n",
            name.c_str(), it->second.data.elements);
    }
    // Store property
    property_overrides.erase(name);
    property_overrides.emplace(name, util::Any(&value, sizeof(T), typeid(typename type_decode<T>::type_t), type_decode<T>::len_t));
}
template<typename T, flamegpu::size_type N>
void RunPlan::setProperty(const std::string &name, const std::array<T, N> &value) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(typename type_decode<T>::type_t)).name());
    }
    if (it->second.data.elements != N * type_decode<T>::len_t) {
        THROW exception::InvalidEnvPropertyType("Environment property array '%s' length mismatch %u != %u "
            "in RunPlan::setProperty()\n",
            name.c_str(), it->second.data.elements, N * type_decode<T>::len_t);
    }
    // Store property
    property_overrides.erase(name);
    property_overrides.emplace(name, util::Any(value.data(), sizeof(T) * N, typeid(typename type_decode<T>::type_t), type_decode<T>::len_t * N));
}
template<typename T, flamegpu::size_type N>
void RunPlan::setProperty(const std::string &name, const flamegpu::size_type index, T value) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(typename type_decode<T>::type_t)).name());
    }
    if (N && N != it->second.data.elements) {
        THROW exception::OutOfBoundsException("Environment property '%s' length mismatch '%u' != '%u', "
            "in RunPlan::setProperty()\n",
            name.c_str(), N, it->second.data.elements);
    }
    const unsigned int t_index = type_decode<T>::len_t * index + type_decode<T>::len_t;
    if (it->second.data.elements < t_index || t_index < index) {
        throw exception::OutOfBoundsException("Environment property array index out of bounds "
            "in RunPlan::setProperty()\n");
    }
    // Check whether array already exists in property overrides
    auto it2 = property_overrides.find(name);
    if (it2 == property_overrides.end()) {
        // Clone default property first
        it2 = property_overrides.emplace(name, it->second.data).first;
    }
    // Store property
    memcpy(static_cast<T*>(it2->second.ptr) + index, &value, sizeof(T));
}
#ifdef SWIG
template<typename T>
void RunPlan::setPropertyArray(const std::string &name, const std::vector<T> &value) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::setPropertyArray()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::setPropertyArray()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(typename type_decode<T>::type_t)).name());
    }
    if (type_decode<T>::len_t * value.size() != it->second.data.elements) {
        THROW exception::InvalidEnvPropertyType("Environment property array length does not match the value provided, %u != %llu,"
            "in RunPlan::setPropertyArray()\n",
            name.c_str(), type_decode<T>::len_t * value.size(), it->second.data.elements);
    }
    // Store property
    property_overrides.erase(name);
    property_overrides.emplace(name, util::Any(value.data(), sizeof(T) * value.size(), typeid(typename type_decode<T>::type_t), type_decode<T>::len_t * value.size()));
}
#endif

template<typename T>
T RunPlan::getProperty(const std::string &name) const {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(typename type_decode<T>::type_t)).name());
    }
    if (it->second.data.elements != type_decode<T>::len_t) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' is an array with %u elements, array method should be used, "
            "in RunPlan::getProperty()\n",
            name.c_str(), it->second.data.elements);
    }
    // Check whether property already exists in property overrides
    const auto it2 = property_overrides.find(name);
    if (it2 != property_overrides.end()) {
        // The property has been overridden, return the value from the override.
        return *static_cast<T *>(it2->second.ptr);
    } else {
        // The property has not been overridden, so return the value from the environment
        return *static_cast<T *>(it->second.data.ptr);
    }
}
template<typename T, flamegpu::size_type N>
std::array<T, N> RunPlan::getProperty(const std::string &name) const {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(typename type_decode<T>::type_t)).name());
    }
    if (it->second.data.elements != N * type_decode<T>::len_t) {
        THROW exception::InvalidEnvPropertyType("Environment property array '%s' length mismatch %u != %u "
            "in RunPlan::getProperty()\n",
            name.c_str(), it->second.data.elements, N);
    }
    // Check whether array already exists in property overrides
    const auto it2 = property_overrides.find(name);
    std::array<T, N> rtn;
    if (it2 != property_overrides.end()) {
        // The property has been overridden, return the override
        memcpy(rtn.data(), it2->second.ptr, it2->second.length);
    } else {
        // The property has not been overridden, return the environment property
        memcpy(rtn.data(), it->second.data.ptr, it->second.data.length);
    }
    return rtn;
}
template<typename T, flamegpu::size_type N>
T RunPlan::getProperty(const std::string &name, const flamegpu::size_type index) const {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(typename type_decode<T>::type_t)).name());
    }
    if (N && N != it->second.data.elements) {
        THROW exception::OutOfBoundsException("Environment property '%s' length mismatch '%u' != '%u', "
            "in RunPlan::getProperty()\n",
            name.c_str(), N, it->second.data.elements);
    }
    const unsigned int t_index = type_decode<T>::len_t * index + type_decode<T>::len_t;
    if (it->second.data.elements < t_index || t_index < index) {
        throw exception::OutOfBoundsException("Environment property array index out of bounds "
            "in RunPlan::getProperty()\n");
    }
    // Check whether property already exists in property overrides
    const auto it2 = property_overrides.find(name);
    if (it2 != property_overrides.end()) {
        // The property has been overridden, return the override
        return static_cast<T *>(it2->second.ptr)[index];
    } else {
        // The property has not been overridden, return the environment property
        return static_cast<T *>(it->second.data.ptr)[index];
    }
}
#ifdef SWIG
/**
 * Gets the currently configured environment property array value
 * @param name Environment property name
 * @tparam T Type of the environment property
 * @throws exception::InvalidEnvProperty If a property of the name does not exist
 */
template<typename T>
std::vector<T> RunPlan::getPropertyArray(const std::string &name) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(typename type_decode<T>::type_t)).name());
    }
    if (it->second.data.elements % type_decode<T>::len_t != 0) {
        THROW exception::InvalidEnvPropertyType("Environmental property array '%s' length (%u) is not a multiple of vector length (%u), "
            "in RunPlan::getPropertyArray().",
             name.c_str(), type_decode<T>::len_t, it->second.data.elements, type_decode<T>::len_t);
    }
    // Check whether array already exists in property overrides
    const auto it2 = property_overrides.find(name);
    std::vector<T> rtn(it->second.data.elements / type_decode<T>::len_t);
    if (it2 != property_overrides.end()) {
        // The property has been overridden, return the override
        memcpy(rtn.data(), it2->second.ptr, it2->second.length);
    } else {
        // The property has not been overridden, return the environment property
        memcpy(rtn.data(), it->second.data.ptr, it->second.data.length);
    }
    return rtn;
}
#endif

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIM_RUNPLAN_H_
