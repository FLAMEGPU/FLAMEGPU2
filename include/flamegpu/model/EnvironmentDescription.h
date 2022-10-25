#ifndef INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDESCRIPTION_H_

#include <unordered_map>
#include <string>
#include <typeinfo>
#include <typeindex>
#include <array>
#include <vector>
#include <utility>

#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/runtime/utility/HostEnvironment.cuh"
#include "flamegpu/util/Any.h"
#include "flamegpu/util/type_decode.h"
#include "flamegpu/gpu/CUDAEnsemble.h"

namespace flamegpu {

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
    friend unsigned int CUDAEnsemble::simulate(const RunPlanVector &plans);

 public:
    /**
     * Holds all of the properties required to add a value to EnvironmentManager
     */
    struct PropData {
        /**
         * @param _is_const Is the property constant
         * @param _data The data to initially fill the property with
         */
        PropData(bool _is_const, const util::Any &_data)
            : isConst(_is_const)
            , data(_data) { }
        bool isConst;
        const util::Any data;
        bool operator==(const PropData &rhs) const {
            if (this == &rhs)
                return true;
            if (this->isConst != rhs.isConst
               || this->data.elements != rhs.data.elements
               || this->data.length != rhs.data.length
               || this->data.type != rhs.data.type)
                return false;
            if (this->data.ptr == rhs.data.ptr)
                return true;
            for (size_t i = 0; i < this->data.length; ++i) {
                if (static_cast<const char *>(this->data.ptr)[i] != static_cast<const char *>(rhs.data.ptr)[i])
                    return false;
            }
            return true;
        }
        bool operator!=(const PropData& rhs) const {
            return !operator==(rhs);
        }
    };
    /**
     * Holds all of the properties required to add a value to EnvironmentManager
     */
    struct MacroPropData {
        /**
         * @param _type The type index of the base type (e.g. typeid(float))
         * @param _type_size The size of the base type (e.g. sizeof(float))
         * @param _elements Number of elements in each dimension
         */
        MacroPropData(const std::type_index &_type, const size_t _type_size, const std::array<unsigned int, 4> &_elements)
            : type(_type)
            , type_size(_type_size)
            , elements(_elements) { }
        std::type_index type;
        size_t type_size;
        std::array<unsigned int, 4> elements;
        bool operator==(const MacroPropData& rhs) const {
            if (this == &rhs)
                return true;
            if (this->type != rhs.type
                || this->type_size != rhs.type_size
                || this->elements[0] != rhs.elements[0]
                || this->elements[1] != rhs.elements[1]
                || this->elements[2] != rhs.elements[2]
                || this->elements[3] != rhs.elements[3])
                return false;
            for (size_t i = 0; i < this->elements.size(); ++i) {
                if (this->elements[i] != rhs.elements[i])
                    return false;
            }
            return true;
        }
        bool operator!=(const MacroPropData& rhs) const {
            return !operator==(rhs);
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
     * @throws exception::DuplicateEnvProperty If a property of the same name already exists
     */
    template<typename T>
    void newProperty(const std::string &name, T value, bool isConst = false);
    /**
     * Adds a new environment property array
     * @param name Name used for accessing the property
     * @param value Stored value of the property
     * @param isConst If set to true, it is not possible to change the value during the simulation
     * @tparam T Type of the environmental property array to be created
     * @tparam N Length of the environmental property array to be created
     * @throws exception::DuplicateEnvProperty If a property of the same name already exists
     */
    template<typename T, EnvironmentManager::size_type N>
    void newProperty(const std::string &name, const std::array<T, N> &value, bool isConst = false);
#ifdef SWIG
    /**
     * Adds a new environment property array
     * @param name Name used for accessing the property
     * @param value Stored value of the property
     * @param isConst If set to true, it is not possible to change the value during the simulation
     * @tparam T Type of the environmental property array to be created
     * @throws exception::DuplicateEnvProperty If a property of the same name already exists
     */
    template<typename T>
    void newPropertyArray(const std::string &name, const std::vector<T> &value, const bool isConst = false);
#endif
    /**
     * Define a new environment macro property
     *
     * Environment macro properties are designed for large environment properties, too large of fast constant memory.
     * This means they must instead be stored in slower global memory, however that allows them to be modified during agent functions via a limited set of atomic operations.
     *
     * @param name Name of the macro property
     * @tparam T Type of the macro property
     * @tparam I Length of the first dimension of the macro property, default 1
     * @tparam J Length of the second dimension of the macro property, default 1
     * @tparam K Length of the third dimension of the macro property, default 1
     * @tparam W Length of the fourth dimension of the macro property, default 1
     */
    template<typename T, EnvironmentManager::size_type I = 1, EnvironmentManager::size_type J = 1, EnvironmentManager::size_type K = 1, EnvironmentManager::size_type W = 1>
    void newMacroProperty(const std::string& name);
#ifdef SWIG
    /**
     * Define a new environment macro property, swig specific version
     *
     * Environment macro properties are designed for large environment properties, too large of fast constant memory.
     * This means they must instead be stored in slower global memory, however that allows them to be modified during agent functions via a limited set of atomic operations.
     *
     * @param name Name of the macro property
     * @param I Length of the first dimension of the macro property, default 1
     * @param J Length of the second dimension of the macro property, default 1
     * @param K Length of the third dimension of the macro property, default 1
     * @param W Length of the fourth dimension of the macro property, default 1
     * @tparam T Type of the macro property
     */
    template<typename T>
    void newMacroProperty_swig(const std::string& name, EnvironmentManager::size_type I = 1, EnvironmentManager::size_type J = 1, EnvironmentManager::size_type K = 1, EnvironmentManager::size_type W = 1);
#endif
    /**
     * Gets an environment property
     * @param name name used for accessing the property
     * @tparam T Type of the value to be returned
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     */
    template<typename T>
    T getProperty(const std::string &name) const;
    /**
     * Gets an environment property array
     * @param name name used for accessing the property
     * @tparam T Type of the value to be returned
     * @tparam N Length of the array to be returned
     * @throws exception::InvalidEnvProperty If a property array of the name does not exist
     */
    template<typename T, EnvironmentManager::size_type N>
    std::array<T, N> getProperty(const std::string &name) const;
    /**
     * Gets an element of an environment property array
     * @param name name used for accessing the property
     * @param index element from the environment property array to return
     * @tparam T Type of the value to be returned
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     */
    template<typename T>
    T getProperty(const std::string &name, EnvironmentManager::size_type index) const;
#ifdef SWIG
    /**
     * Gets an environment property array
     * @param name name used for accessing the property
     * @tparam T Type of the value to be returned
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     */
    template<typename T>
    std::vector<T> getPropertyArray(const std::string &name) const;
#endif
    /**
     * Returns whether an environment property is marked as const
     * @param name name used for accessing the property
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     */
    bool getConst(const std::string &name);
    /**
     * Sets an environment property
     * @param name name used for accessing the property
     * @param value value to set the property
     * @tparam T Type of the value to be returned
     * @return Returns the previous value
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     */
    template<typename T>
    T setProperty(const std::string &name, T value);
    /**
     * Sets an environment property array
     * @param name name used for accessing the property
     * @param value value to set the property
     * @tparam T Type of the value to be returned
     * @tparam N Length of the array to be returned
     * @return Returns the previous value
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
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
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     * @see set(const std::string &, const T &value)
     */
    template<typename T>
    T setProperty(const std::string &name, EnvironmentManager::size_type index, T value);
#ifdef SWIG
    /**
     * Sets an environment property array
     * @param name name used for accessing the property
     * @param value value to set the property (vector must be of the correct length)
     * @tparam T Type of the value to be returned
     * @return Returns the previous value
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     */
    template<typename T>
    std::vector<T> setPropertyArray(const std::string &name, const std::vector<T> &value);
#endif

    const std::unordered_map<std::string, PropData> getPropertiesMap() const;
    const std::unordered_map<std::string, MacroPropData> getMacroPropertiesMap() const;

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
    void newProperty(const std::string &name, const char *ptr, size_t len, bool isConst, EnvironmentManager::size_type elements, const std::type_index &type);
    /**
     * Main storage of all properties
     */
    std::unordered_map<std::string, PropData> properties{};
    /**
     * Main storage of all macroproperties
     */
    std::unordered_map<std::string, MacroPropData> macro_properties{};
};


/**
 * Constructors
 */
template<typename T>
void EnvironmentDescription::newProperty(const std::string &name, T value, bool isConst) {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::newProperty().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<typename type_decode<T>::type_t>::value || std::is_enum<typename type_decode<T>::type_t>::value,
        "Only arithmetic types can be used as environmental properties");
    if (properties.find(name) != properties.end()) {
        THROW exception::DuplicateEnvProperty("Environmental property with name '%s' already exists, "
            "in EnvironmentDescription::newProperty().",
            name.c_str());
    }
    newProperty(name, reinterpret_cast<const char*>(&value), sizeof(T), isConst, type_decode<T>::len_t, typeid(typename type_decode<T>::type_t));
}
template<typename T, EnvironmentManager::size_type N>
void EnvironmentDescription::newProperty(const std::string &name, const std::array<T, N> &value, bool isConst) {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::newProperty().");
    }
    static_assert(type_decode<T>::len_t * N > 0, "Environment property arrays must have a length greater than 0.");
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<typename type_decode<T>::type_t>::value || std::is_enum<typename type_decode<T>::type_t>::value,
        "Only arithmetic types can be used as environmental properties");
    if (properties.find(name) != properties.end()) {
        THROW exception::DuplicateEnvProperty("Environmental property with name '%s' already exists, "
            "in EnvironmentDescription::newProperty().",
            name.c_str());
    }
    newProperty(name, reinterpret_cast<const char*>(value.data()), N * sizeof(T), isConst, type_decode<T>::len_t * N, typeid(typename type_decode<T>::type_t));
}
#ifdef SWIG
template<typename T>
void EnvironmentDescription::newPropertyArray(const std::string &name, const std::vector<T> &value, bool isConst) {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::newPropertyArray().");
    }
    if (value.size() == 0) {
        THROW exception::InvalidEnvProperty("Environment property arrays must have a length greater than 0."
            "in EnvironmentDescription::newPropertyArray().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<typename type_decode<T>::type_t>::value || std::is_enum<typename type_decode<T>::type_t>::value,
        "Only arithmetic types can be used as environmental properties");
    if (properties.find(name) != properties.end()) {
        THROW exception::DuplicateEnvProperty("Environmental property with name '%s' already exists, "
            "in EnvironmentDescription::newPropertyArray().",
            name.c_str());
    }
    newProperty(name, reinterpret_cast<const char*>(value.data()), value.size() * sizeof(T), isConst, type_decode<T>::len_t * value.size(), typeid(typename type_decode<T>::type_t));
}
#endif
/**
 * Getters
 */
template<typename T>
T EnvironmentDescription::getProperty(const std::string &name) const {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<typename type_decode<T>::type_t>::value || std::is_enum<typename type_decode<T>::type_t>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
            THROW exception::InvalidEnvPropertyType("Environmental property ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::getProperty().",
                name.c_str(), i->second.data.type.name(), typeid(typename type_decode<T>::type_t).name());
        }
        if (i->second.data.elements != type_decode<T>::len_t) {
            THROW exception::InvalidEnvPropertyType("Length of named environmental property (%u) does not match vector length (%u), "
                "in EnvironmentDescription::getProperty().",
                i->second.data.elements, type_decode<T>::len_t);
        }
        return *reinterpret_cast<T*>(i->second.data.ptr);
    }
    THROW exception::InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::getProperty().",
        name.c_str());
}
template<typename T, EnvironmentManager::size_type N>
std::array<T, N> EnvironmentDescription::getProperty(const std::string &name) const {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<typename type_decode<T>::type_t>::value || std::is_enum<typename type_decode<T>::type_t>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
            THROW exception::InvalidEnvPropertyType("Environmental property array ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::getProperty().",
                name.c_str(), i->second.data.type.name(), typeid(typename type_decode<T>::type_t).name());
        }
        if (i->second.data.elements != type_decode<T>::len_t * N) {
            THROW exception::InvalidEnvPropertyType("Length of named environmental property array (%u) does not match requested length (%u), "
                "in EnvironmentDescription::getProperty().",
                i->second.data.elements, type_decode<T>::len_t * N);
        }
        // Copy old data to return
        std::array<T, N> rtn;
        memcpy(rtn.data(), reinterpret_cast<T*>(i->second.data.ptr), N * sizeof(T));
        return rtn;
    }
    THROW exception::InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::getProperty().",
        name.c_str());
}
template<typename T>
T EnvironmentDescription::getProperty(const std::string &name, EnvironmentManager::size_type index) const {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<typename type_decode<T>::type_t>::value || std::is_enum<typename type_decode<T>::type_t>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
            THROW exception::InvalidEnvPropertyType("Environmental property array ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::getProperty().",
                name.c_str(), i->second.data.type.name(), typeid(typename type_decode<T>::type_t).name());
        }
        if (i->second.data.elements % type_decode<T>::len_t != 0) {
            THROW exception::InvalidEnvPropertyType("Environmental property array ('%s') length (%u) does not divide by vector length (%u), "
                "in EnvironmentDescription::getPropertyArray().",
                name.c_str(), i->second.data.elements, type_decode<T>::len_t);
        }
        const unsigned int t_index = type_decode<T>::len_t * index + type_decode<T>::len_t;
        if (i->second.data.elements < t_index || t_index < index) {
            THROW exception::OutOfBoundsException("Index (%u) exceeds named environmental property array's length (%u), "
                "in EnvironmentDescription::getProperty().",
                index, i->second.data.elements / type_decode<T>::len_t);
        }
        // Copy old data to return
        return *(reinterpret_cast<T*>(i->second.data.ptr) + index);
    }
    THROW exception::InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::getProperty().",
        name.c_str());
}
#ifdef SWIG
template<typename T>
std::vector<T> EnvironmentDescription::getPropertyArray(const std::string& name) const {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<typename type_decode<T>::type_t>::value || std::is_enum<typename type_decode<T>::type_t>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
            THROW exception::InvalidEnvPropertyType("Environmental property array ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::getPropertyArray().",
                name.c_str(), i->second.data.type.name(), typeid(typename type_decode<T>::type_t).name());
        }
        if (i->second.data.elements % type_decode<T>::len_t != 0) {
            THROW exception::InvalidEnvPropertyType("Environmental property array ('%s') length (%u) does not divide by vector length (%d), "
                "in EnvironmentDescription::getPropertyArray().",
                name.c_str(), i->second.data.elements, type_decode<T>::len_t);
        }
        // Copy old data to return
        std::vector<T> rtn(i->second.data.elements / type_decode<T>::len_t);
        memcpy(rtn.data(), reinterpret_cast<T*>(i->second.data.ptr), i->second.data.elements * sizeof(typename type_decode<T>::type_t));
        return rtn;
    }
    THROW exception::InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::getPropertyArray().",
        name.c_str());
}
#endif

/**
 * Setters
 */
template<typename T>
T EnvironmentDescription::setProperty(const std::string &name, T value) {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::setProperty().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<typename type_decode<T>::type_t>::value || std::is_enum<typename type_decode<T>::type_t>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
            THROW exception::InvalidEnvPropertyType("Environmental property ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::setProperty().",
                name.c_str(), i->second.data.type.name(), typeid(typename type_decode<T>::type_t).name());
        }
        if (i->second.data.elements != type_decode<T>::len_t) {
            THROW exception::InvalidEnvPropertyType("Length of named environmental property (%u) does not match vector length (%u), "
                "in EnvironmentDescription::setProperty().",
                i->second.data.elements, type_decode<T>::len_t);
        }
        // Copy old data to return
        T rtn = *reinterpret_cast<T*>(i->second.data.ptr);
        // Store data
        memcpy(i->second.data.ptr, &value, sizeof(T));
        return rtn;
    }
    THROW exception::InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::setProperty().",
        name.c_str());
}
template<typename T, EnvironmentManager::size_type N>
std::array<T, N> EnvironmentDescription::setProperty(const std::string &name, const std::array<T, N> &value) {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::setProperty().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<typename type_decode<T>::type_t>::value || std::is_enum<typename type_decode<T>::type_t>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
            THROW exception::InvalidEnvPropertyType("Environmental property array ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::setProperty().",
                name.c_str(), i->second.data.type.name(), typeid(typename type_decode<T>::type_t).name());
        }
        if (i->second.data.elements != N * type_decode<T>::len_t) {
            THROW exception::InvalidEnvPropertyType("Length of named environmental property array (%u) does not match requested length (%u), "
                "in EnvironmentDescription::setProperty().",
                i->second.data.elements, N * type_decode<T>::len_t);
        }
        // Copy old data to return
        std::array<T, N> rtn;
        memcpy(rtn.data(), reinterpret_cast<T*>(i->second.data.ptr), N * sizeof(T));
        // Store data
        memcpy(reinterpret_cast<T*>(i->second.data.ptr), value.data(), N * sizeof(T));
        return rtn;
    }
    THROW exception::InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::setProperty().",
        name.c_str());
}
template<typename T>
T EnvironmentDescription::setProperty(const std::string &name, EnvironmentManager::size_type index, T value) {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::setProperty().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<typename type_decode<T>::type_t>::value || std::is_enum<typename type_decode<T>::type_t>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
            THROW exception::InvalidEnvPropertyType("Environmental property array ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::setProperty().",
                name.c_str(), i->second.data.type.name(), typeid(typename type_decode<T>::type_t).name());
        }
        if (i->second.data.elements % type_decode<T>::len_t != 0) {
            THROW exception::InvalidEnvPropertyType("Environmental property array ('%s') length (%u) does not divide by vector length (%u), "
                "in EnvironmentDescription::setProperty().",
                name.c_str(), i->second.data.elements, type_decode<T>::len_t);
        }
        const unsigned int t_index = type_decode<T>::len_t * index + type_decode<T>::len_t;
        if (i->second.data.elements < t_index || t_index < index) {
            THROW exception::OutOfBoundsException("Index (%u) exceeds named environmental property array's length (%u), "
                "in EnvironmentDescription::setProperty().",
                index, i->second.data.elements / type_decode<T>::len_t);
        }
        // Copy old data to return
        T rtn = *(reinterpret_cast<T*>(i->second.data.ptr) +  index);
        // Store data
        memcpy(reinterpret_cast<T*>(i->second.data.ptr) + index, &value, sizeof(T));
        return rtn;
    }
    THROW exception::InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::setProperty().",
        name.c_str());
}
#ifdef SWIG
template<typename T>
std::vector<T> EnvironmentDescription::setPropertyArray(const std::string& name, const std::vector<T>& value) {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::set().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<typename type_decode<T>::type_t>::value || std::is_enum<typename type_decode<T>::type_t>::value,
        "Only arithmetic types can be used as environmental properties");
    auto &&i = properties.find(name);
    if (i != properties.end()) {
        if (i->second.data.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
            THROW exception::InvalidEnvPropertyType("Environmental property array ('%s') type (%s) does not match template argument T (%s), "
                "in EnvironmentDescription::setPropertyArray().",
                name.c_str(), i->second.data.type.name(), typeid(typename type_decode<T>::type_t).name());
        }
        if (i->second.data.elements % type_decode<T>::len_t != 0) {
            THROW exception::InvalidEnvPropertyType("Environmental property array ('%s') length (%u) does not divide by vector length (%u), "
                "in EnvironmentDescription::setPropertyArray().",
                name.c_str(), i->second.data.elements, type_decode<T>::len_t);
        }
        if (i->second.data.elements != value.size() * type_decode<T>::len_t) {
            THROW exception::OutOfBoundsException("Length of named environmental property array (%u) does not match length of provided vector (%llu), "
                "in EnvironmentDescription::setPropertyArray().",
                i->second.data.elements / type_decode<T>::len_t, value.size());
        }
        // Copy old data to return
        std::vector<T> rtn(i->second.data.elements / type_decode<T>::len_t);
        memcpy(rtn.data(), reinterpret_cast<T*>(i->second.data.ptr), i->second.data.elements * sizeof(typename type_decode<T>::type_t));
        // Store data
        memcpy(reinterpret_cast<T*>(i->second.data.ptr), value.data(), i->second.data.elements * sizeof(typename type_decode<T>::type_t));
        return rtn;
    }
    THROW exception::InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::set().",
        name.c_str());
}
#endif
template<typename T, EnvironmentManager::size_type I, EnvironmentManager::size_type J, EnvironmentManager::size_type K, EnvironmentManager::size_type W>
void EnvironmentDescription::newMacroProperty(const std::string& name) {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment macro property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::newMacroProperty().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental macro properties");
    static_assert(I > 0, "Environment macro properties must have a length greater than 0 in the first axis.");
    static_assert(J > 0, "Environment macro properties must have a length greater than 0 in the second axis.");
    static_assert(K > 0, "Environment macro properties must have a length greater than 0 in the third axis.");
    static_assert(W > 0, "Environment macro properties must have a length greater than 0 in the fourth axis.");
    if (macro_properties.find(name) != macro_properties.end()) {
        THROW exception::DuplicateEnvProperty("Environmental macro property with name '%s' already exists, "
            "in EnvironmentDescription::newMacroProperty().",
            name.c_str());
    }
    macro_properties.emplace(name, MacroPropData(typeid(T), sizeof(T), { I, J, K, W }));
}
#ifdef SWIG
template<typename T>
void EnvironmentDescription::newMacroProperty_swig(const std::string& name, EnvironmentManager::size_type I, EnvironmentManager::size_type J, EnvironmentManager::size_type K, EnvironmentManager::size_type W) {
    if (!name.empty() && name[0] == '_') {
        THROW exception::ReservedName("Environment macro property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDescription::newMacroProperty().");
    }
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental macro properties");
    if (I <= 0) {
        THROW exception::DuplicateEnvProperty("Environmental macro property with name '%s' must have a length greater than 0 in the first axis, "
            "in EnvironmentDescription::newMacroProperty().",
            name.c_str());
    } else if (J <= 0) {
        THROW exception::DuplicateEnvProperty("Environmental macro property with name '%s' must have a length greater than 0 in the second axis, "
            "in EnvironmentDescription::newMacroProperty().",
            name.c_str());
    } else if (K <= 0) {
        THROW exception::DuplicateEnvProperty("Environmental macro property with name '%s' must have a length greater than 0 in the third axis, "
            "in EnvironmentDescription::newMacroProperty().",
            name.c_str());
    } else if (W <= 0) {
        THROW exception::DuplicateEnvProperty("Environmental macro property with name '%s' must have a length greater than 0 in the fourth axis, "
            "in EnvironmentDescription::newMacroProperty().",
            name.c_str());
    } else if (macro_properties.find(name) != macro_properties.end()) {
        THROW exception::DuplicateEnvProperty("Environmental macro property with name '%s' already exists, "
            "in EnvironmentDescription::newMacroProperty().",
            name.c_str());
    }
    macro_properties.emplace(name, MacroPropData(typeid(T), sizeof(T), { I, J, K, W }));
}
#endif
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDESCRIPTION_H_
