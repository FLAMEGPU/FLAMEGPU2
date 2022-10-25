#ifndef INCLUDE_FLAMEGPU_SIM_RUNPLANVECTOR_H_
#define INCLUDE_FLAMEGPU_SIM_RUNPLANVECTOR_H_

#include <random>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <limits>

#include "flamegpu/sim/RunPlan.h"
#include "flamegpu/util/detail/StaticAssert.h"
#include "flamegpu/util/type_decode.h"


namespace flamegpu {

class ModelDescription;
class EnvironmentDescription;

/**
 * Vector of RunPlan
 * Contains additional methods for generating collections of RunPlans and combining RunPlanVectors
 */
class RunPlanVector : private std::vector<RunPlan>  {
    friend class RunPlan;
    friend class SimRunner;
    friend unsigned int CUDAEnsemble::simulate(const RunPlanVector &plans);

 public:
    /**
     * Constructor, requires the model description to validate environment properties match up
     * @todo Unsure if this will require additional info, e.g. steps?
     */
    explicit RunPlanVector(const ModelDescription &model, unsigned int initial_length);
    /**
     * Set the random simulation seed of each RunPlan currently within this vector
     * @param initial_seed The random seed applied to the first item
     * @param step The value added to the previous seed to calculate the next seed
     * @note A step of 0, will give the exact same seed to all RunPlans
     */
    void setRandomSimulationSeed(const uint64_t &initial_seed, unsigned int step = 0);
    /**
     * Set the steps of each RunPlan currently within this vector
     * @param steps The number of steps to be executed
     * @note If 0 is provided, the model must have an exit condition
     */
    void setSteps(unsigned int steps);
    /**
     * Set the the sub directory within the output directory for outputs of runplans in this vector
     * @param subdir The name of the subdirectory
     * @note Defaults to empty string, where no subdirectory is used
     */
    void setOutputSubdirectory(const std::string &subdir);
    /**
     * Set named environment property to a specific value
     * @param name The name of the environment property to set
     * @param value The value of the environment property to set
     * @tparam T The type of the environment property, this must match the ModelDescription
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T, or length > 1
     */
    template<typename T>
    void setProperty(const std::string &name, const T &value);
    /**
     * Set named environment property array to a specific value
     * This version should be used for array properties
     * @param name Environment property name
     * @param value Environment property value (override)
     * @tparam T Type of the environment property
     * @tparam N Length of the array to be returned
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T, or length to N
     */
    template<typename T, EnvironmentManager::size_type N>
    void setProperty(const std::string &name, const std::array<T, N> &value);
    /**
     * Array property element equivalent of setProperty()
     * @param name The name of the environment property array to affect
     * @param index The index of the environment property array to set
     * @param value The value of the environment property array to set
     * @tparam T The type of the environment property, this must match the ModelDescription
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T
     * @throws exception::OutOfBoundsException If index is not in range of the length of the property array
     * @see setProperty(const std::string &name, const T &value)
     */
    template<typename T>
    void setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value);
#ifdef SWIG
    /**
     * Set named environment property array to a specific value
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
     * Sweep named environment property over an inclusive uniform distribution
     * value = min * (1.0 - a) + max * a, where a = index/(size()-1)
     * Integer types will be rounded to the nearest integer
     * @param name The name of the environment property to set
     * @param min The value to set the first environment property
     * @param max The value to set the last environment property
     * @tparam T The type of the environment property, this must match the ModelDescription
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T, or length > 1
     * @throws exception::OutOfBoundsException If this vector has a length less than 2
     */
    template<typename T>
    void setPropertyUniformDistribution(const std::string &name, const T &min, const T &max);
    /**
     * Array property element equivalent of setPropertyUniformDistribution()
     * Sweep element of named environment property array over an inclusive uniform distribution
     * value = min * (1.0 - a) + max * a, where a = index/(size()-1)
     * Integer types will be rounded to the nearest integer
     * @param name The name of the environment property to set
     * @param index The index of the element within the environment property array to set
     * @param min The value to set the first environment property array element
     * @param max The value to set the last environment property array element
     * @tparam T The type of the environment property, this must match the ModelDescription
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T
     * @throws exception::OutOfBoundsException If index is greater than or equal to the length of the environment property array
     * @throws exception::OutOfBoundsException If this vector has a length less than 2
     * @see setPropertyUniformDistribution(const std::string &name, const T &min, const T &max)
     */
    template<typename T>
    void setPropertyUniformDistribution(const std::string &name, const EnvironmentManager::size_type &index, const T &min, const T &max);
    /**
     * Seed the internal random generator used for random property distributions
     * This will only affect subsequent calls to setPropertyRandom()
     * @param seed The random seed to be used
     */
    void setRandomPropertySeed(const uint64_t &seed);
    /**
     * Get the seed used for the internal random generator used for random property distributions
     * This will only valid for calls to setPropertyRandom() since the last call toSetRandomPropertySeed
     * @return the seed used for random properties since the last call to setPropertyRandom
     */
    uint64_t getRandomPropertySeed();
    /**
     * Sweep named environment property over a uniform random distribution
     * Integer types have a range [min, max]
     * Floating point types have a range [min, max)
     * @param name The name of the environment property to set
     * @param min The value of the range to set the first environment property
     * @param max The value of the range to set the last environment property
     * @tparam T The type of the environment property, this must match the ModelDescription
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T, or length > 1
     * @throws exception::OutOfBoundsException If this vector has a length less than 2
     */
    template<typename T>
    void setPropertyUniformRandom(const std::string &name, const T &min, const T &max);
    /**
     * Array property element equivalent of setPropertyUniformRandom()
     * Sweep named environment property over a uniform random distribution
     * Integer types have a range [min, max]
     * Floating point types have a range [min, max)
     * @param name The name of the environment property to set
     * @param index The index of the array element to set
     * @param min The value of the range to set the first environment property
     * @param max The value of the range to set the last environment property
     * @tparam T The type of the environment property, this must match the ModelDescription
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T
     * @throws exception::OutOfBoundsException If index is greater than or equal to the length of the environment property array
     * @throws exception::OutOfBoundsException If this vector has a length less than 2
     * @see setPropertyUniformRandom(const std::string &name, const T &min, const T &max)
     */
    template<typename T>
    void setPropertyUniformRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &min, const T &max);
    /**
     * Sweep named environment property over a normal random distribution
     * Only floating point types are supported
     * @param name The name of the environment property to set
     * @param mean Mean of the distribution (its expected value). Which coincides with the location of its peak.
     * @param stddev Standard deviation: The square root of variance, representing the dispersion of values from the distribution mean.
     * @tparam T The type of the environment property, this must match the ModelDescription
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T, or length > 1
     * @throws exception::OutOfBoundsException If this vector has a length less than 2
     */
    template<typename T>
    void setPropertyNormalRandom(const std::string &name, const T &mean, const T &stddev);
    /**
     * Array property element equivalent of setPropertyNormalRandom()
     * Sweep named environment property over a normal random distribution
     * Only floating point types are supported
     * @param name The name of the environment property to set
     * @param index The index of the array element to set
     * @param mean Mean of the distribution (its expected value). Which coincides with the location of its peak.
     * @param stddev Standard deviation: The square root of variance, representing the dispersion of values from the distribution mean.
     * @tparam T The type of the environment property, this must match the ModelDescription
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T
     * @throws exception::OutOfBoundsException If index is greater than or equal to the length of the environment property array
     * @throws exception::OutOfBoundsException If this vector has a length less than 2
     * @see setPropertyNormalRandom(const std::string &name, const T &mean, const T &stddev)
     */
    template<typename T>
    void setPropertyNormalRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &mean, const T &stddev);
    /**
     * Sweep named environment property over a log normal random distribution
     * Only floating point types are supported
     * @param name The name of the environment property to set
     * @param mean Mean of the underlying normal distribution formed by the logarithm transformations of the possible values in this distribution.
     * @param stddev Standard deviation of the underlying normal distribution formed by the logarithm transformations of the possible values in this distribution.
     * @tparam T The type of the environment property, this must match the ModelDescription
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T, or length > 1
     * @throws exception::OutOfBoundsException If this vector has a length less than 2
     */
    template<typename T>
    void setPropertyLogNormalRandom(const std::string &name, const T &mean, const T &stddev);
    /**
     * Array property element equivalent of setPropertyLogNormalRandom()
     * Sweep named environment property over a log normal random distribution
     * Only floating point types are supported
     * @param name The name of the environment property to set
     * @param index The index of the array element to set
     * @param mean Mean of the underlying normal distribution formed by the logarithm transformations of the possible values in this distribution.
     * @param stddev Standard deviation of the underlying normal distribution formed by the logarithm transformations of the possible values in this distribution.
     * @tparam T The type of the environment property, this must match the ModelDescription
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T
     * @throws exception::OutOfBoundsException If index is greater than or equal to the length of the environment property array
     * @throws exception::OutOfBoundsException If this vector has a length less than 2
     * @see setPropertyNormalRandom(const std::string &name, const T &mean, const T &stddev)
     */
    template<typename T>
    void setPropertyLogNormalRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &mean, const T &stddev);
    /**
     * Use a random distribution to generate parameters for the named environment property
     * @param name The name of the environment property to set
     * @param distribution The random distribution to use for generating random property values
     * @tparam T The type of the environment property, this must match the ModelDescription
     * @tparam rand_dist An object satisfying the requirements of RandomNumberDistribution e.g. std::uniform_real_distribution
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T, or length to N
     * @throws exception::OutOfBoundsException If this vector has a length less than 2
     */
    template<typename T, typename rand_dist>
    void setPropertyRandom(const std::string &name, rand_dist &distribution);
    /**
     * Array property element equivalent of setPropertyRandom()
     * Use a random distribution to generate parameters for the specified element of the named environment property array
     * @param name The name of the environment property to set
     * @param index The index of the element within the environment property array to set
     * @param distribution The random distribution to use for generating random property values
     * @tparam T The type of the environment property array, this must match the ModelDescription
     * @tparam rand_dist An object satisfying the requirements of RandomNumberDistribution e.g. std::uniform_real_distribution
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::InvalidEnvPropertyType If a property with the name has a type different to T, or length to N
     * @throws exception::OutOfBoundsException If index is greater than or equal to the length of the environment property array
     * @throws exception::OutOfBoundsException If this vector has a length less than 2
     */
    template<typename T, typename rand_dist>
    void setPropertyRandom(const std::string &name, const EnvironmentManager::size_type &index, rand_dist &distribution);

    /**
     * Expose inherited std::vector methods/classes
     */     
#ifndef SWIG
    using std::vector<RunPlan>::begin;
    using std::vector<RunPlan>::end;
    using std::vector<RunPlan>::size;
    using std::vector<RunPlan>::operator[];
    using std::vector<RunPlan>::insert;
#else
    // Can't get SWIG %import to use std::vector<RunPlan> so manually implement the required items
    size_t size() const { return std::vector<RunPlan>::size(); }
    RunPlan& operator[] (const size_t _Pos) { return std::vector<RunPlan>::operator[](_Pos); }
#endif

    /**
     * Operator methods for combining vectors
     */
    RunPlanVector operator+(const RunPlan& rhs) const;
    RunPlanVector operator+(const RunPlanVector& rhs) const;
    RunPlanVector& operator+=(const RunPlan& rhs);
    RunPlanVector& operator+=(const RunPlanVector& rhs);
    RunPlanVector& operator*=(unsigned int rhs);
    RunPlanVector operator*(unsigned int rhs) const;

 private:
    RunPlanVector(const std::shared_ptr<const std::unordered_map<std::string, EnvironmentDescription::PropData>> &environment, bool allow_0_steps);
    /**
     * Seed used for the current `rand` instance, which is only valid for elements generated since the last call to setRandomPropertySeed
     */
    uint64_t randomPropertySeed;
    /**
     * Random distribution used by setPropertyRandom()
     * Initially set to a random seed with std::random_device (which is a platform specific source of random integers)
     */
    std::mt19937_64 rand;
    std::shared_ptr<const std::unordered_map<std::string, EnvironmentDescription::PropData>> environment;
    const bool allow_0_steps;
};

template<typename T>
void RunPlanVector::setProperty(const std::string &name, const T &value) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlanVector::setProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlanVector::setProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(typename type_decode<T>::type_t)).name());
    }
    if (it->second.data.elements != type_decode<T>::len_t) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' is an array with %u elements, array method should be used, "
            "in RunPlanVector::setProperty()\n",
            name.c_str(), it->second.data.elements);
    }
    for (auto &i : *this) {
        i.setProperty<T>(name, value);
    }
}
template<typename T, EnvironmentManager::size_type N>
void RunPlanVector::setProperty(const std::string &name, const std::array<T, N> &value) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlanVector::setProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlanVector::setProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(typename type_decode<T>::type_t)).name());
    }
    if (it->second.data.elements != N * type_decode<T>::len_t) {
        THROW exception::InvalidEnvPropertyType("Environment property array '%s' length mismatch %u != %u "
            "in RunPlanVector::setProperty()\n",
            name.c_str(), it->second.data.elements, N * type_decode<T>::len_t);
    }
    for (auto &i : *this) {
        i.setProperty<T, N>(name, value);
    }
}
template<typename T>
void RunPlanVector::setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlanVector::setProperty()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlanVector::setProperty()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(typename type_decode<T>::type_t)).name());
    }
    const unsigned int t_index = type_decode<T>::len_t * index + type_decode<T>::len_t;
    if (t_index > it->second.data.elements || t_index < index) {
        throw exception::OutOfBoundsException("Environment property array index out of bounds "
            "in RunPlanVector::setProperty()\n");
    }
    for (auto &i : *this) {
        i.setProperty<T>(name, index, value);
    }
}
#ifdef SWIG
template<typename T>
void RunPlanVector::setPropertyArray(const std::string &name, const std::vector<T> &value) {
    // Validation
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlanVector::setPropertyArray()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlanVector::setPropertyArray()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(typename type_decode<T>::type_t)).name());
    }
    if (value.size() * type_decode<T>::len_t != it->second.data.elements) {
        THROW exception::InvalidEnvProperty("Environment property array length does not match the value provided, %u != %llu,"
            "in RunPlanVector::setPropertyArray()\n",
            name.c_str(), value.size() * type_decode<T>::len_t, it->second.data.elements);
    }
    for (auto &i : *this) {
        i.setPropertyArray<T>(name, value);
    }
}
#endif

template<typename T>
void RunPlanVector::setPropertyUniformDistribution(const std::string &name, const T &min, const T &max) {
    // Validation
    if (this->size() < 2) {
        THROW exception::OutOfBoundsException("Unable to apply a property distribution a vector with less than 2 elements, "
            "in RunPlanVector::setPropertyUniformDistribution()\n");
    }
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlanVector::setPropertyUniformDistribution()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(T))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlanVector::setPropertyUniformDistribution()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(T)).name());
    }
    if (it->second.data.elements != 1) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' is an array with %u elements, array method should be used, "
            "in RunPlanVector::setPropertyUniformDistribution()\n",
            name.c_str(), it->second.data.elements);
    }
    unsigned int ct = 0;
    for (auto &i : *this) {
        const double a = static_cast<double>(ct++) / (this->size() - 1);
        double lerp = min * (1.0 - a) + max * a;
        if (std::numeric_limits<T>::is_integer)
            lerp = round(lerp);
        const T lerp_t = static_cast<T>(lerp);
        i.setProperty<T>(name, lerp_t);
    }
}
template<typename T>
void RunPlanVector::setPropertyUniformDistribution(const std::string &name, const EnvironmentManager::size_type &index, const T &min, const T &max) {
    // Validation
    if (this->size() < 2) {
        THROW exception::OutOfBoundsException("Unable to apply a property distribution a vector with less than 2 elements, "
            "in RunPlanVector::setPropertyUniformDistribution()\n");
    }
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlanVector::setPropertyUniformDistribution()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(T))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlanVector::setPropertyUniformDistribution()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(T)).name());
    }
    const unsigned int t_index = type_decode<T>::len_t * index + type_decode<T>::len_t;
    if (t_index > it->second.data.elements || t_index < index) {
        throw exception::OutOfBoundsException("Environment property array index out of bounds "
            "in RunPlanVector::setPropertyUniformDistribution()\n");
    }
    unsigned int ct = 0;
    for (auto &i : *this) {
        const double a = static_cast<double>(ct++) / (this->size() - 1);
        double lerp = min * (1.0 - a) + max * a;
        if (std::numeric_limits<T>::is_integer)
            lerp = round(lerp);
        const T lerp_t = static_cast<T>(lerp);
        i.setProperty<T>(name, index, lerp_t);
    }
}

template<typename T, typename rand_dist>
void RunPlanVector::setPropertyRandom(const std::string &name, rand_dist &distribution) {
    // Validation
    if (this->size() < 2) {
        THROW exception::OutOfBoundsException("Unable to apply a property distribution a vector with less than 2 elements, "
            "in RunPlanVector::setPropertyRandom()\n");
    }
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlanVector::setPropertyRandom()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(T))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlanVector::setPropertyRandom()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(T)).name());
    }
    if (it->second.data.elements != 1) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' is an array with %u elements, array method should be used, "
            "in RunPlanVector::setPropertyRandom()\n",
            name.c_str(), it->second.data.elements);
    }
    for (auto &i : *this) {
        i.setProperty<T>(name, static_cast<T>(distribution(this->rand)));
    }
}
template<typename T, typename rand_dist>
void RunPlanVector::setPropertyRandom(const std::string &name, const EnvironmentManager::size_type &index, rand_dist &distribution) {
    // Validation
    if (this->size() < 2) {
        THROW exception::OutOfBoundsException("Unable to apply a property distribution a vector with less than 2 elements, "
            "in RunPlanVector::setPropertyRandom()\n");
    }
    const auto it = environment->find(name);
    if (it == environment->end()) {
        THROW exception::InvalidEnvProperty("Environment description does not contain property '%s', "
            "in RunPlanVector::setPropertyRandom()\n",
            name.c_str());
    }
    if (it->second.data.type != std::type_index(typeid(T))) {
        THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
            "in RunPlanVector::setPropertyRandom()\n",
            name.c_str(), it->second.data.type.name(), std::type_index(typeid(T)).name());
    }
    const unsigned int t_index = type_decode<T>::len_t * index + type_decode<T>::len_t;
    if (t_index > it->second.data.elements || t_index < index) {
        throw exception::OutOfBoundsException("Environment property array index out of bounds "
            "in RunPlanVector::setPropertyRandom()\n");
    }
    for (auto &i : *this) {
        i.setProperty<T>(name, index, static_cast<T>(distribution(this->rand)));
    }
}
/**
 * Convenience random implementations
 */
template<typename T>
void RunPlanVector::setPropertyUniformRandom(const std::string &name, const T &min, const T &max) {
    static_assert(util::detail::StaticAssert::_Is_IntType<T>::value, "Invalid template argument for RunPlanVector::setPropertyUniformRandom(const std::string &name, const T &min, const T&max)");
    std::uniform_int_distribution<T> dist(min, max);
    setPropertyRandom<T>(name, dist);
}
template<typename T>
void RunPlanVector::setPropertyUniformRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &min, const T &max) {
    static_assert(util::detail::StaticAssert::_Is_IntType<T>::value, "Invalid template argument for RunPlanVector::setPropertyUniformRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &min, const T&max)");
    std::uniform_int_distribution<T> dist(min, max);
    setPropertyRandom<T>(name, index, dist);
}
template<typename T>
void RunPlanVector::setPropertyNormalRandom(const std::string &name, const T &mean, const T &stddev) {
    static_assert(util::detail::StaticAssert::_Is_RealType<T>::value, "Invalid template argument for RunPlanVector::setPropertyNormalRandom(const std::string &name, const T &mean, const T &stddev)");
    std::normal_distribution<T> dist(mean, stddev);
    setPropertyRandom<T>(name, dist);
}
template<typename T>
void RunPlanVector::setPropertyNormalRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &mean, const T &stddev) {
    static_assert(util::detail::StaticAssert::_Is_RealType<T>::value,
        "Invalid template argument for RunPlanVector::setPropertyNormalRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &mean, const T &stddev)");
    std::normal_distribution<T> dist(mean, stddev);
    setPropertyRandom<T>(name, index, dist);
}
template<typename T>
void RunPlanVector::setPropertyLogNormalRandom(const std::string &name, const T &mean, const T &stddev) {
    static_assert(util::detail::StaticAssert::_Is_RealType<T>::value,
    "Invalid template argument for RunPlanVector::setPropertyLogNormalRandom(const std::string &name, const T &mean, const T &stddev)");
    std::lognormal_distribution<T> dist(mean, stddev);
    setPropertyRandom<T>(name, dist);
}
template<typename T>
void RunPlanVector::setPropertyLogNormalRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &mean, const T &stddev) {
    static_assert(util::detail::StaticAssert::_Is_RealType<T>::value,
    "Invalid template argument for RunPlanVector::setPropertyLogNormalRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &mean, const T &stddev)");
    std::lognormal_distribution<T> dist(mean, stddev);
    setPropertyRandom<T>(name, index, dist);
}
/**
 * Special cases
 * std::random doesn't support char, emulate behaviour
 * char != signed char (or unsigned char)
 */
template<>
inline void RunPlanVector::setPropertyUniformRandom(const std::string &name, const float &min, const float &max) {
    std::uniform_real_distribution<float> dist(min, max);
    setPropertyRandom<float>(name, dist);
}
template<>
inline void RunPlanVector::setPropertyUniformRandom(const std::string &name, const EnvironmentManager::size_type &index, const float &min, const float &max) {
    std::uniform_real_distribution<float> dist(min, max);
    setPropertyRandom<float>(name, index, dist);
}
template<>
inline void RunPlanVector::setPropertyUniformRandom(const std::string &name, const double &min, const double &max) {
    std::uniform_real_distribution<double> dist(min, max);
    setPropertyRandom<double>(name, dist);
}
template<>
inline void RunPlanVector::setPropertyUniformRandom(const std::string &name, const EnvironmentManager::size_type &index, const double &min, const double &max) {
    std::uniform_real_distribution<double> dist(min, max);
    setPropertyRandom<double>(name, index, dist);
}
template<>
inline void RunPlanVector::setPropertyUniformRandom(const std::string &name, const char min, const char max) {
    std::uniform_int_distribution<int16_t> dist(min, max);
    setPropertyRandom<char>(name, dist);
}
template<>
inline void RunPlanVector::setPropertyUniformRandom(const std::string &name, const EnvironmentManager::size_type &index, const char min, const char max) {
    std::uniform_int_distribution<int16_t> dist(min, max);
    setPropertyRandom<char>(name, index, dist);
}
template<>
inline void RunPlanVector::setPropertyUniformRandom(const std::string &name, const unsigned char min, const unsigned char max) {
    std::uniform_int_distribution<uint16_t> dist(min, max);
    setPropertyRandom<unsigned char>(name, dist);
}
template<>
inline void RunPlanVector::setPropertyUniformRandom(const std::string &name, const EnvironmentManager::size_type &index, const unsigned char min, const unsigned char max) {
    std::uniform_int_distribution<uint16_t> dist(min, max);
    setPropertyRandom<unsigned char>(name, index, dist);
}
template<>
inline void RunPlanVector::setPropertyUniformRandom(const std::string &name, const signed char &min, const signed char &max) {
    std::uniform_int_distribution<int16_t> dist(min, max);
    setPropertyRandom<signed char>(name, dist);
}
template<>
inline void RunPlanVector::setPropertyUniformRandom(const std::string &name, const EnvironmentManager::size_type &index, const signed char &min, const signed char &max) {
    std::uniform_int_distribution<int16_t> dist(min, max);
    setPropertyRandom<signed char>(name, index, dist);
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIM_RUNPLANVECTOR_H_
