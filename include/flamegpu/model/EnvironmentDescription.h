#ifndef INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDESCRIPTION_H_

#include <unordered_map>
#include <string>
#include <typeinfo>
#include <typeindex>
#include <array>
#include <vector>

#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/runtime/utility/HostEnvironment.cuh"
#include "flamegpu/util/Any.h"
#include "flamegpu/gpu/CUDAEnsemble.h"

/**
 * @brief Description class for environment properties
 * 
 * Allows environment properties to be prepared and attached to a ModelDescription.
 * Properties can be any arithmetic or enum type.
 * Properties marked as const within the EnvironmentDescription cannot be changed during the simulation
 */
class EnvironmentDescription {
    /**
     * EnvironmentManager needs access to our internal members
     * @see EnvironmentManager::init(const std::string &, const EnvironmentDescription &)
     */
    friend class EnvironmentManager;
    /**
     * This directly accesses properties map, to build mappings
     * Not required if this class is changed into description/data format like others
     */
    friend class SubEnvironmentDescription;
    /**
     * Constructor has access to privately add reserved items
     * Might be a cleaner way to do this
     */
    friend class CUDASimulation;

    friend class SimRunner;
    friend void CUDAEnsemble::simulate(const RunPlanVec &plans);

 public:
    /**
     * Holds all of the properties required to add a value to EnvironmentManager
     */
    struct PropData {
        /**
         * @param _is_const Is the property constant
         * @param _data The data to initially fill the property with
         */
        PropData(const bool &_is_const, const Any &_data)
            : isConst(_is_const)
            , data(_data) { }
        bool isConst;
        const Any data;
        bool operator==(const PropData &rhs) const {
            if (this->isConst != rhs.isConst
               || this->data.elements != rhs.data.elements
               || this->data.length != rhs.data.length
               || this->data.type != rhs.data.type)
                return false;
            for (size_t i = 0; i < this->data.length; ++i) {
                if (static_cast<const char *>(this->data.ptr)[i] != static_cast<const char *>(rhs.data.ptr)[i])
                    return false;
            }
            return true;
        }
    };
    /**
     * Default destruction
     */
    EnvironmentDescription();

    bool operator==(const EnvironmentDescription& rhs) const;
    bool operator!=(const EnvironmentDescription& rhs) const;
    /**
     * Adds a new environment property
     * @param name name used for accessing the property
     * @param value stored value of the property
     * @param isConst If set to true, it is not possible to change the value during the simulation
     * @tparam T Type of the environmental property to be created
     * @throws DuplicateEnvProperty If a property of the same name already exists
     */
    template<typename T>
    void newProperty(const std::string &name, const T &value, const bool &isConst = false);
    /**
     * Adds a new environment property array
     * @param name Name used for accessing the property
     * @param value Stored value of the property
     * @param isConst If set to true, it is not possible to change the value during the simulation
     * @tparam T Type of the environmental property array to be created
     * @tparam N Length of the environmental property array to be created
     * @throws DuplicateEnvProperty If a property of the same name already exists
     */
    template<typename T, EnvironmentManager::size_type N>
    void newProperty(const std::string &name, const std::array<T, N> &value, const bool &isConst = false);
#ifdef SWIG
    /**
     * Adds a new environment property array
     * @param name Name used for accessing the property
     * @param length Length of the environmental property array to be created
     * @param value Stored value of the property
     * @param isConst If set to true, it is not possible to change the value during the simulation
     * @tparam T Type of the environmental property array to be created
     * @throws DuplicateEnvProperty If a property of the same name already exists
     */
    template<typename T>
    void newPropertyArray(const std::string &name, const EnvironmentManager::size_type &length, const std::vector<T> &value, const bool &isConst = false);
#endif
    /**
     * Gets an environment property
     * @param name name used for accessing the property
     * @tparam T Type of the value to be returned
     * @throws InvalidEnvProperty If a property of the name does not exist
     */
    template<typename T>
    T getProperty(const std::string &name) const;
    /**
     * Gets an environment property array
     * @param name name used for accessing the property
     * @tparam T Type of the value to be returned
     * @tparam N Length of the array to be returned
     * @throws InvalidEnvProperty If a property array of the name does not exist
     */
    template<typename T, EnvironmentManager::size_type N>
    std::array<T, N> getProperty(const std::string &name) const;
    /**
     * Gets an element of an environment property array
     * @param name name used for accessing the property
     * @param index element from the environment property array to return
     * @tparam T Type of the value to be returned
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     */
    template<typename T>
    T getProperty(const std::string &name, const EnvironmentManager::size_type &index) const;
#ifdef SWIG
    /**
     * Gets an environment property array
     * @param name name used for accessing the property
     * @tparam T Type of the value to be returned
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     */
    template<typename T>
    std::vector<T> getPropertyArray(const std::string &name) const;
#endif
    /**
     * Returns whether an environment property is marked as const
     * @param name name used for accessing the property
     * @throws InvalidEnvProperty If a property of the name does not exist
     */
    bool getConst(const std::string &name);
    /**
     * Sets an environment property
     * @param name name used for accessing the property
     * @param value value to set the property
     * @tparam T Type of the value to be returned
     * @return Returns the previous value
     * @throws InvalidEnvProperty If a property of the name does not exist
     */
    template<typename T>
    T setProperty(const std::string &name, const T &value);
    /**
     * Sets an environment property array
     * @param name name used for accessing the property
     * @param value value to set the property
     * @tparam T Type of the value to be returned
     * @tparam N Length of the array to be returned
     * @return Returns the previous value
     * @throws InvalidEnvProperty If a property of the name does not exist
     */
    template<typename T, EnvironmentManager::size_type N>
    std::array<T, N> setProperty(const std::string &name, const std::array<T, N> &value);
    /**
     * Sets an element of an environment property array
     * @param name name used for accessing the property
     * @param index element from the environment property array to set
     * @param value value to set the property
     * @tparam T Type of the value to be returned
     * @return Returns the previous value of the environment property array element which has been set
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     * @see set(const std::string &, const T &value)
     */
    template<typename T>
    T setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value);
#ifdef SWIG
    /**
     * Sets an environment property array
     * @param name name used for accessing the property
     * @param value value to set the property (vector must be of the correct length)
     * @tparam T Type of the value to be returned
     * @return Returns the previous value
     * @throws InvalidEnvProperty If a property of the name does not exist
     */
    template<typename T>
    std::vector<T> setPropertyArray(const std::string &name, const std::vector<T> &value);
#endif

    const std::unordered_map<std::string, PropData> getPropertiesMap() const;

 private:
    /**
     * Internal common add method, actually performs the heavy lifting of changing properties
     * @param name Name used for accessing the property
     * @param ptr Pointer to data to initially fill property with
     * @param len Length of data pointed to by ptr
     * @param isConst If set to true, it is not possible to change the value during the simulation
     * @param elements How many elements does the property have (1 if it's not an array)
     * @param type value returned by typeid()
     */
    void newProperty(const std::string &name, const char *ptr, const size_t &len, const bool &isConst, const EnvironmentManager::size_type &elements, const std::type_index &type);
    /**
     * Main storage of all properties
     */
    std::unordered_map<std::string, PropData> properties{};
};


/**
 * Constructors
 */
template<typename T>
void EnvironmentDescription::newProperty(const std::string &name, const T &value, const bool &isConst) {
    if (!name.empty() && name[0] == '_') {
        THROW ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::add().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    if (properties.find(name) != properties.end()) {
        THROW DuplicateEnvProperty("Environmental property with name '%s' already exists, "
            "in EnvironmentDescription::add().",
            name.c_str());
    }
    newProperty(name, reinterpret_cast<const char*>(&value), sizeof(T), isConst, 1, typeid(T));
}
template<typename T, EnvironmentManager::size_type N>
void EnvironmentDescription::newProperty(const std::string &name, const std::array<T, N> &value, const bool &isConst) {
    if (!name.empty() && name[0] == '_') {
        THROW ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::add().");
    }
    static_assert(N > 0, "Environment property arrays must have a length greater than 0.");
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    if (properties.find(name) != properties.end()) {
        THROW DuplicateEnvProperty("Environmental property with name '%s' already exists, "
            "in EnvironmentDescription::add().",
            name.c_str());
    }
    newProperty(name, reinterpret_cast<const char*>(value.data()), N * sizeof(T), isConst, N, typeid(T));
}
#ifdef SWIG
template<typename T>
void EnvironmentDescription::newPropertyArray(const std::string &name, const EnvironmentManager::size_type &N, const std::vector<T> &value, const bool& isConst) {
    if (!name.empty() && name[0] == '_') {
        THROW ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::addArray().");
    }
    if (value.size() != N) {
        THROW InvalidEnvProperty("Environment property array length does not match the value provided, %u != %llu,"
            "in EnvironmentDescription::addArray().", N, value.size());
    }
    if (N == 0) {
        THROW InvalidEnvProperty("Environment property arrays must have a length greater than 0."
            "in EnvironmentDescription::addArray().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    if (properties.find(name) != properties.end()) {
        THROW DuplicateEnvProperty("Environmental property with name '%s' already exists, "
            "in EnvironmentDescription::addArray().",
            name.c_str());
    }
    newProperty(name, reinterpret_cast<const char*>(value.data()), N * sizeof(T), isConst, N, typeid(T));
}
#endif
/**
 * Getters
 */
template<typename T>
T EnvironmentDescription::getProperty(const std::string &name) const {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(T))) {
            THROW InvalidEnvPropertyType("Environmental property ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::get().",
                name.c_str(), i->second.data.type.name(), typeid(T).name());
        }
        return *reinterpret_cast<T*>(i->second.data.ptr);
    }
    THROW InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::get().",
        name.c_str());
}
template<typename T, EnvironmentManager::size_type N>
std::array<T, N> EnvironmentDescription::getProperty(const std::string &name) const {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(T))) {
            THROW InvalidEnvPropertyType("Environmental property array ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::get().",
                name.c_str(), i->second.data.type.name(), typeid(T).name());
        }
        if (i->second.data.elements != N) {
            THROW OutOfBoundsException("Length of named environmental property array (%u) does not match template argument N (%u), "
                "in EnvironmentDescription::get().",
                i->second.data.elements, N);
        }
        // Copy old data to return
        std::array<T, N> rtn;
        memcpy(rtn.data(), reinterpret_cast<T*>(i->second.data.ptr), N * sizeof(T));
        return rtn;
    }
    THROW InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::get().",
        name.c_str());
}
template<typename T>
T EnvironmentDescription::getProperty(const std::string &name, const EnvironmentManager::size_type &index) const {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(T))) {
            THROW InvalidEnvPropertyType("Environmental property array ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::get().",
                name.c_str(), i->second.data.type.name(), typeid(T).name());
        }
        if (i->second.data.elements <= index) {
            THROW OutOfBoundsException("Index (%u) exceeds named environmental property array's length (%u), "
                "in EnvironmentDescription::get().",
                index, i->second.data.elements);
        }
        // Copy old data to return
        return *(reinterpret_cast<T*>(i->second.data.ptr) + index);
    }
    THROW InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::get().",
        name.c_str());
}
#ifdef SWIG
template<typename T>
std::vector<T> EnvironmentDescription::getPropertyArray(const std::string& name) const {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(T))) {
            THROW InvalidEnvPropertyType("Environmental property array ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::getArray().",
                name.c_str(), i->second.data.type.name(), typeid(T).name());
        }
        // Copy old data to return
        std::vector<T> rtn(i->second.data.elements);
        memcpy(rtn.data(), reinterpret_cast<T*>(i->second.data.ptr), i->second.data.elements * sizeof(T));
        return rtn;
    }
    THROW InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::getArray().",
        name.c_str());
}
#endif

/**
 * Setters
 */
template<typename T>
T EnvironmentDescription::setProperty(const std::string &name, const T &value) {
    if (!name.empty() && name[0] == '_') {
        THROW ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::set().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(T))) {
            THROW InvalidEnvPropertyType("Environmental property ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::set().",
                name.c_str(), i->second.data.type.name(), typeid(T).name());
        }
        // Copy old data to return
        T rtn = *reinterpret_cast<T*>(i->second.data.ptr);
        // Store data
        memcpy(i->second.data.ptr, &value, sizeof(T));
        return rtn;
    }
    THROW InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::set().",
        name.c_str());
}
template<typename T, EnvironmentManager::size_type N>
std::array<T, N> EnvironmentDescription::setProperty(const std::string &name, const std::array<T, N> &value) {
    if (!name.empty() && name[0] == '_') {
        THROW ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::set().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(T))) {
            THROW InvalidEnvPropertyType("Environmental property array ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::set().",
                name.c_str(), i->second.data.type.name(), typeid(T).name());
        }
        if (i->second.data.elements != N) {
            THROW OutOfBoundsException("Length of named environmental property array (%u) does not match template argument N (%u), "
                "in EnvironmentDescription::set().",
                i->second.data.elements, N);
        }
        // Copy old data to return
        std::array<T, N> rtn;
        memcpy(rtn.data(), reinterpret_cast<T*>(i->second.data.ptr), N * sizeof(T));
        // Store data
        memcpy(reinterpret_cast<T*>(i->second.data.ptr), value.data(), N * sizeof(T));
        return rtn;
    }
    THROW InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::set().",
        name.c_str());
}
template<typename T>
T EnvironmentDescription::setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value) {
    if (!name.empty() && name[0] == '_') {
        THROW ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::set().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(T))) {
            THROW InvalidEnvPropertyType("Environmental property array ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::set().",
                name.c_str(), i->second.data.type.name(), typeid(T).name());
        }
        if (i->second.data.elements <= index) {
            THROW OutOfBoundsException("Index (%u) exceeds named environmental property array's length (%u), "
                "in EnvironmentDescription::set().",
                index, i->second.data.elements);
        }
        // Copy old data to return
        T rtn = *(reinterpret_cast<T*>(i->second.data.ptr) + index);
        // Store data
        memcpy(reinterpret_cast<T*>(i->second.data.ptr) + index, &value, sizeof(T));
        return rtn;
    }
    THROW InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::set().",
        name.c_str());
}
#ifdef SWIG
template<typename T>
std::vector<T> EnvironmentDescription::setPropertyArray(const std::string& name, const std::vector<T>& value) {
    if (!name.empty() && name[0] == '_') {
        THROW ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::set().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(T))) {
            THROW InvalidEnvPropertyType("Environmental property array ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::set().",
                name.c_str(), i->second.data.type.name(), typeid(T).name());
        }
        if (i->second.data.elements != value.size()) {
            THROW OutOfBoundsException("Length of named environmental property array (%u) does not match length of provided vector (%llu), "
                "in EnvironmentDescription::set().",
                i->second.data.elements, value.size());
        }
        // Copy old data to return
        std::vector<T> rtn(i->second.data.elements);
        memcpy(rtn.data(), reinterpret_cast<T*>(i->second.data.ptr), i->second.data.elements * sizeof(T));
        // Store data
        memcpy(reinterpret_cast<T*>(i->second.data.ptr), value.data(), i->second.data.elements * sizeof(T));
        return rtn;
    }
    THROW InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::set().",
        name.c_str());
}
#endif
#endif  // INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDESCRIPTION_H_
