#ifndef INCLUDE_FLAMEGPU_SIM_RUNPLAN_H_
#define INCLUDE_FLAMEGPU_SIM_RUNPLAN_H_

#include <unordered_map>
#include <string>
#include <typeinfo>
#include <vector>
#include <memory>

#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/util/Any.h"
#include "flamegpu/gpu/CUDAEnsemble.h"


class ModelDescription;
class RunPlanVec;
/**
 * Individual run config
 */
class RunPlan {
    friend class RunPlanVec;
    friend class SimRunner;
    friend class jsonLogger;
    friend class xmlLogger;

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
    void setRandomSimulationSeed(const unsigned int &random_seed);
    /**
     * Set the number of steps for this instance of the simulation
     * A steps value of 0 requires the ModelDescription to have atleast 1 exit condition
     * @param steps The number of steps to execute, 0 is unlimited but requires an exit condition
     */
    void setSteps(const unsigned int &steps);
    /**
     * Set the sub directory within the output directory for outputs of this run
     * If left empty, output will not goto subdirectories
     * @param steps The number of steps to execute, 0 is unlimited but requires an exit condition
     */
    void setOutputSubdirectory(const std::string &subdir);
    /**
     * Set the environment property override for this run of the model
     * @param name Environment property name
     * @param value Environment property value (override)
     * @tparam T Type of the environment property
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws InvalidEnvPropertyType If a property with the name has a type different to T
     */
    template<typename T>
    void setProperty(const std::string &name, const T&value);
    /**
     * Set the environment property override for this run of the model
     * This version should be used for array properties
     * @param name Environment property name
     * @param value Environment property value (override)
     * @tparam T Type of the environment property
     * @tparam N Length of the array to be returned
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws InvalidEnvPropertyType If a property with the name has a type different to T, or length to N
     */
    template<typename T, EnvironmentManager::size_type N>
    void setProperty(const std::string &name, const std::array<T, N> &value);
    /**
     * Set the environment property override for this run of the model
     * This version should be used for setting individual elements of array properties
     * @param name Environment property name
     * @param index Length of the array to be returned
     * @param value Environment property value (override)
     * @tparam T Type of the environment property
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws InvalidEnvPropertyType If a property with the name has a type different to T
     * @throws std::out_of_range If index is not in range of the length of the property array
     */
    template<typename T>
    void setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value);
#ifdef SWIG
    /**
     * Set the environment property override for this run of the model
     * This version should be used for array properties
     * @param name Environment property name
     * @param length Length of the environmental property array to be created
     * @param value Environment property value (override)
     * @tparam T Type of the environment property
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws InvalidEnvProperty If value.size() != length
     * @throws InvalidEnvPropertyType If a property with the name has a type different to T, or length to N
     */
    template<typename T>
    void setPropertyArray(const std::string &name, const EnvironmentManager::size_type &length, const std::vector<T> &value);
#endif
    /**
     * Returns the random seed used for this simulation run
     */
    unsigned int getRandomSimulationSeed() const;
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
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws InvalidEnvPropertyType If a property with the name has a type different to T
     */
    template<typename T>
    T getProperty(const std::string &name) const;
    /**
     * Gets the currently configured environment property array value
     * @param name name used for accessing the property
     * @tparam T Type of the value to be returned
     * @tparam N Length of the array to be returned
     * @throws InvalidEnvProperty If a property array of the name does not exist
     * @throws InvalidEnvPropertyType If a property with the name has a type different to T
     */
    template<typename T, EnvironmentManager::size_type N>
    std::array<T, N> getProperty(const std::string &name) const;
    /**
     * Gets an element of the currently configured environment property array
     * @param name name used for accessing the property
     * @param index element from the environment property array to return
     * @tparam T Type of the value to be returned
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws InvalidEnvPropertyType If a property with the name has a type different to T
     * @throws std::out_of_range If index is not in range of the length of the property array
     */
    template<typename T>
    T getProperty(const std::string &name, const EnvironmentManager::size_type &index) const;
#ifdef SWIG
    /**
     * Gets the currently configured environment property array value
     * @param name Environment property name
     * @tparam T Type of the environment property
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws InvalidEnvPropertyType If a property with the name has a type different to T
     */
    template<typename T>
    std::vector<T> getPropertyArray(const std::string &name);
#endif

    /**
     * Operator methods for combining vectors
     */
    RunPlanVec operator+(const RunPlan& rhs) const;
    RunPlanVec operator+(const RunPlanVec& rhs) const;
    RunPlanVec operator*(const unsigned int& rhs) const;

 private:
    explicit RunPlan(const std::shared_ptr<const std::unordered_map<std::string, EnvironmentDescription::PropData>> &environment, const bool &allow_0);
    unsigned int random_seed;
    unsigned int steps;
    std::string output_subdirectory;
    std::unordered_map<std::string, Any> property_overrides;
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
void RunPlan::setProperty(const std::string &name, const T&value) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(T)).name());
    }
    if (it->second.data.elements != 1) {
        THROW InvalidEnvPropertyType("Environment property '%s' is an array with %u elements, array method should be used, "
            "in RunPlan::setProperty()\n",
            name.c_str(), it->second.data.elements);
    }
    // Store property
    property_overrides.emplace(name, Any(&value, sizeof(T), typeid(T), 1));
}
template<typename T, EnvironmentManager::size_type N>
void RunPlan::setProperty(const std::string &name, const std::array<T, N> &value) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(T)).name());
    }
    if (it->second.data.elements != N) {
        THROW InvalidEnvPropertyType("Environment property array '%s' length mismatch %u != %u "
            "in RunPlan::setProperty()\n",
            name.c_str(), it->second.data.elements, N);
    }
    // Store property
    property_overrides.emplace(name, Any(value.data(), sizeof(T) * N, typeid(T), N));
}
template<typename T>
void RunPlan::setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(T)).name());
    }
    if (it->second.data.elements >= index) {
        throw std::out_of_range("Environment property array index out of bounds "
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
void RunPlan::setPropertyArray(const std::string &name, const EnvironmentManager::size_type &N, const std::vector<T> &value) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::setProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(T)).name());
    }
    if (it->second.data.elements != N) {
        THROW InvalidEnvProperty("Environment property array '%s' length mismatch %u != %u, "
            "in RunPlan::setProperty()\n",
            name.c_str(), it->second.data.elements, N);
    }
    if (value.size() != N) {
        THROW InvalidEnvProperty("Environment property array length does not match the value provided, %u != %llu,"
            "in RunPlan::setProperty()\n",
            name.c_str(), value.size(), N);
    }
    // Store property
    property_overrides.emplace(name, Any(value.data(), sizeof(T) * N, typeid(T), N));
}
#endif

template<typename T>
T RunPlan::getProperty(const std::string &name) const {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(T)).name());
    }
    if (it->second.data.elements != 1) {
        THROW InvalidEnvPropertyType("Environment property '%s' is an array with %u elements, array method should be used, "
            "in RunPlan::getProperty()\n",
            name.c_str(), it->second.data.elements);
    }
    // Check whether property already exists in property overrides
    const auto it2 = property_overrides.find(name);
    if (it2 == property_overrides.end())
      return *static_cast<T *>(it2->second.ptr);
    return *static_cast<T *>(it->second.data.ptr);
}
template<typename T, EnvironmentManager::size_type N>
std::array<T, N> RunPlan::getProperty(const std::string &name) const {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(T)).name());
    }
    if (it->second.data.elements != N) {
        THROW InvalidEnvPropertyType("Environment property array '%s' length mismatch %u != %u "
            "in RunPlan::getProperty()\n",
            name.c_str(), it->second.data.elements, N);
    }
    // Check whether array already exists in property overrides
    const auto it2 = property_overrides.find(name);
    std::array<T, N> rtn;
    if (it2 == property_overrides.end()) {
        memcpy(rtn.data(), it2->second.ptr, it2->second.length);
    } else {
        memcpy(rtn.data(), it->second.data.ptr, it->second.data.length);
    }
    return rtn;
}
template<typename T>
T RunPlan::getProperty(const std::string &name, const EnvironmentManager::size_type &index) const {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(T)).name());
    }
    if (it->second.data.elements >= index) {
        throw std::out_of_range("Environment property array index out of bounds "
            "in RunPlan::getProperty()\n");
    }
    // Check whether property already exists in property overrides
    const auto it2 = property_overrides.find(name);
    if (it2 == property_overrides.end())
      return static_cast<T *>(it2->second.ptr)[index];
    return static_cast<T *>(it->second.data.ptr)[index];
}
#ifdef SWIG
/**
 * Gets the currently configured environment property array value
 * @param name Environment property name
 * @tparam T Type of the environment property
 * @throws InvalidEnvProperty If a property of the name does not exist
 */
template<typename T>
std::vector<T> RunPlan::getPropertyArray(const std::string &name) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlan::getProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(T)).name());
    }
    // Check whether array already exists in property overrides
    const auto it2 = property_overrides.find(name);
    std::vector<T> rtn(it->second.data.elements);
    if (it2 == property_overrides.end()) {
        memcpy(rtn.data(), it2->second.ptr, it2->second.length);
    } else {
        memcpy(rtn.data(), it->second.data.ptr, it->second.data.length);
    }
    return rtn;
}
#endif

#endif  // INCLUDE_FLAMEGPU_SIM_RUNPLAN_H_
