#include <map>
#include <cassert>
#include <memory>
#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_ENVIRONMENTMANAGER_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_ENVIRONMENTMANAGER_CUH_

#include <cuda_runtime.h>

#include <cstddef>
#include <unordered_map>
#include <array>
#include <string>
#include <type_traits>
#include <list>
#include <utility>
#include <typeindex>
#include <set>
#include <vector>

#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/runtime/cuRVE/curve.h"

struct SubEnvironmentData;
class EnvironmentDescription;
class CUDASimulation;
class CUDAAgent;

/**
 * Singleton manager for managing environment properties storage in constant memory
 * This is an internal class, that should not be accessed directly by modellers
 * @see EnvironmentDescription For describing the initial state of a model's environment properties
 * @see AgentEnvironment For reading environment properties during agent functions on the device
 * @see HostEnvironment For accessing environment properties during host functions
 * @note Not thread-safe
 */
class EnvironmentManager {
    /**
     * Uses instance to for RTC compilation
     */
    friend class CUDAAgent;
    /**
     * Uses instance to initialise a models environment properties on the device
     */
    friend class CUDASimulation;
    /**
     * Uses instance to access env properties in host functions
     */
    friend class HostEnvironment;
    /**
     * Accesses pointer to hc_buffer
     */
    friend class DefragProp;
    /**
     * Accesses properties to find all of a model's vars
     */
    friend class xmlWriter;
    friend class xmlReader;
    friend class jsonWriter;
    friend class jsonReader;
    friend class jsonReader_impl;
    /**
     * CUDASimulation instance id and Property name
     */
    typedef std::pair<unsigned int, std::string> NamePair;
    struct NamePairHash {
        size_t operator()(const NamePair& k) const {
            return std::hash<unsigned int>()(k.first) ^
                (std::hash<std::string>()(k.second) << 1);
        }
    };

 public:
    /**
     * Max amount of space that can be used for storing environmental properties
     */
    static const size_t MAX_BUFFER_SIZE = 10 * 1024;  // 10KB
    /**
     * Offset relative to c_buffer
     * Length in bytes
     */
    typedef unsigned int size_type;
    /**
     * Used to group items required by freeFragments 
     */
    typedef std::pair<ptrdiff_t, size_t> OffsetLen;
    /**
     * Gives names to indices of OffsetLen
     */
    enum OL {
        OFFSET = 0,
        LEN = 1,
    };
    /**
     * Used to group items required by properties
     */
    struct EnvProp {
        /**
         * @param _offset Offset into c_buffer/hc_buffer
         * @param _length Length of associated storage
         * @param _isConst Is the stored data constant
         * @param _elements How many elements does the stored data contain (1 if not array)
         * @param _type Type of propert (from typeid())
         */
        EnvProp(const ptrdiff_t &_offset, const size_t &_length, const bool &_isConst, const size_type &_elements, const std::type_index &_type)
            : offset(_offset),
            length(_length),
            isConst(_isConst),
            elements(_elements),
            type(_type) {}
        ptrdiff_t offset;
        size_t length;
        bool isConst;
        size_type elements;
        const std::type_index type;
        ptrdiff_t rtc_offset = 0;  // This is set by buildRTCOffsets();
    };
    /**
     * Used to represent properties of a mapped environment property
     */
    struct MappedProp {
        /**
         * @param _masterProp Master property of mapping
         * @param _isConst Is the stored data constant
         */
        MappedProp(const NamePair &_masterProp, const bool &_isConst)
            : masterProp(_masterProp),
            isConst(_isConst) {}
        const NamePair masterProp;
        const bool isConst;
    };
    /**
     * This structure is a clone of EnvProp
     * However, instead of offset (which points to an offset into hc_buffer)
     * data is avaialable, which points to host memory
     */
    struct DefragProp {
        /**
         * @param ep Environment property to clone
         * @note ep.offset is converted to a host pointer by adding to hc_buffer
         */
        explicit DefragProp(const EnvProp &ep)
            :data(EnvironmentManager::getInstance().hc_buffer + ep.offset),
            length(ep.length),
            isConst(ep.isConst),
            elements(ep.elements),
            type(ep.type) { }
        /**
        * @param _data Pointer to the data in host memory
        * @param _length Length of associated storage
        * @param _isConst Is the stored data constant
        * @param _elements How many elements does the stored data contain (1 if not array)
        * @param _type Type of propert (from typeid())
        */
        DefragProp(void *_data, const size_t &_length, const bool &_isConst, const size_type &_elements, const std::type_index &_type)
            : data(_data),
            length(_length),
            isConst(_isConst),
            elements(_elements),
            type(_type) { }
        void *data;
        size_t length;
        bool isConst;
        size_type elements;
        const std::type_index type;
    };
    /**
     * Struct used by rtc_caches
     * Represents a personalised constant cache buffer for a single CUDASimulation instance
     * These are shared by submodels
     */
    struct RTCEnvPropCache {
        /**
         * Host copy of the device memory pointed to by c_buffer
         */
        char hc_buffer[MAX_BUFFER_SIZE];
        /**
         * Offset relative to c_buffer, where no more data has been stored
         */
        ptrdiff_t nextFree = 0;
    };
    /**
     * Typedef for the map used for defragementation
     * The map is ordered by key of type size, therefore a reverse sort creates aligned data
     */
    typedef std::multimap<size_t, std::pair<const NamePair, DefragProp>> DefragMap;
    /**
     * Activates a models environment properties, by adding them to constant cache
     * @param instance_id instance_id of the CUDASimulation instance the properties are attached to
     * @param desc environment properties description to use
     */
    void init(const unsigned int &instance_id, const EnvironmentDescription &desc);
    /**
     * Submodel variant of init()
     * Activates a models unmapped environment properties, by adding them to constant cache
     * Maps a models mapped environment properties to their master property
     * @param instance_id instance_id of the CUDASimulation instance the properties are attached to
     * @param master_instance_id instance_id of the CUDASimulation instance of the parent of the submodel
     * @param desc environment properties description to use
     */
    void init(const unsigned int &instance_id, const EnvironmentDescription &desc, const unsigned int &master_instance_id, const SubEnvironmentData &mapping);
    /**
     * RTC functions hold their own unique constants for environment variables. This function copies all environment variable to the RTC copies.
     * It can not be incorporated into init() as init will be called before RTC functions have been compiled.
     * Uses the already populated Environment data from the cuda_model rather than environmentDescription.
     * @param cuda_model the cuda model being initialised.
     */
    void initRTC(const CUDASimulation &cuda_model);
    /**
     * Deactives all environmental properties linked to the named model from constant cache
     * @param instance_id instance_id of the CUDASimulation instance the properties are attached to
     */
    void free(const unsigned int &instance_id);
    /**
     * Adds a new environment property
     * @param name name used for accessing the property
     * @param value stored value of the property
     * @param isConst If set to true, it is not possible to change the value
     * @tparam T Type of the environmental property to be created
     * @throws DuplicateEnvProperty If a property of the same name already exists
     */
    template<typename T>
    void add(const NamePair &name, const T &value, const bool &isConst = false);
    /**
     * Convenience method: Adds a new environment property
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param var_name name used for accessing the property
     * @param value stored value of the property
     * @param isConst If set to true, it is not possible to change the value
     * @tparam T Type of the environmental property to be created
     * @throws DuplicateEnvProperty If a property of the same name already exists
     * @see add(const NamePair &, const T &, const bool &)
     */
    template<typename T>
    void add(const unsigned int &instance_id, const std::string &var_name, const T &value, const bool &isConst = false);
    /**
     * Adds a new environment property array
     * @param name name used for accessing the property
     * @param value stored value of the property
     * @param isConst If set to true, it is not possible to change the value
     * @tparam T Type of the environmental property array to be created
     * @tparam N Length of the environmental property array to be created
     * @throws DuplicateEnvProperty If a property of the same name already exists
     */
    template<typename T, size_type N>
    void add(const NamePair &name, const std::array<T, N> &value, const bool &isConst = false);
    /**
     * Convenience method: Adds a new environment property array
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param var_name name used for accessing the property
     * @param value stored value of the property
     * @param isConst If set to true, it is not possible to change the value
     * @tparam T Type of the environmental property array to be created
     * @tparam N Length of the environmental property array to be created
     * @throws DuplicateEnvProperty If a property of the same name already exists
     * @see add(const NamePair &, const std::array<T, N> &, const bool &)
     */
    template<typename T, size_type N>
    void add(const unsigned int &instance_id, const std::string &var_name, const std::array<T, N> &value, const bool &isConst = false);
    /**
     * Sets an environment property
     * @param name name used for accessing the property
     * @param value value to set the property
     * @tparam T Type of the environmental property array to be created
     * @return Returns the previous value
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws ReadOnlyEnvProperty If the named property is marked as const
     */
    template<typename T>
    T set(const NamePair &name, const T &value);
    /**
     * Convenience method: Sets an environment property
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param var_name name used for accessing the property
     * @param value value to set the property
     * @tparam T Type of the environmental property array to be created
     * @return Returns the previous value
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws ReadOnlyEnvProperty If the named property is marked as const
     * @see add(const NamePair &, const T &)
     */
    template<typename T>
    T set(const unsigned int &instance_id, const std::string &var_name, const T &value);
    /**
     * Sets an environment property array
     * @param name name used for accessing the property array
     * @param value value to set the property array
     * @tparam T Type of the environmental property array to be created
     * @tparam N Length of the environmental property array to be created
     * @return Returns the previous value
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws ReadOnlyEnvProperty If the named property is marked as const
     */
    template<typename T, size_type N>
    std::array<T, N> set(const NamePair &name, const std::array<T, N> &value);
    /**
     * Convenience method: Sets an environment property array
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param var_name name used for accessing the property
     * @param value value to set the property array
     * @tparam T Type of the environmental property array to be created
     * @tparam N Length of the environmental property array to be created
     * @return Returns the previous value
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws ReadOnlyEnvProperty If the named property is marked as const
     * @see set(const NamePair &, const std::array<T, N> &)
     */
    template<typename T, size_type N>
    std::array<T, N> set(const unsigned int &instance_id, const std::string &var_name, const std::array<T, N> &value);
#ifdef SWIG
    /**
     * Sets an environment property array
     * @param name name used for accessing the property array
     * @param value value to set the property array
     * @tparam T Type of the environmental property array to be created
     * @return Returns the previous value
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws ReadOnlyEnvProperty If the named property is marked as const
     */
    template<typename T>
    std::vector<T> setArray(const NamePair &name, const std::vector<T> &value);
#endif
    /**
     * Sets an element of an environment property array
     * @param name name used for accessing the property array
     * @param index Index of the element within the array
     * @param value value to set the element of the property array
     * @tparam T Type of the environmental property array to be created
     * @return Returns the previous value
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     */
    template<typename T>
    T set(const NamePair &name, const size_type &index, const T &value);
    /**
     * Convenience method: Sets an element of an environment property array
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param var_name name used for accessing the property
     * @param index Index of the element within the array
     * @param value value to set the element of the property array
     * @tparam T Type of the environmental property array to be created
     * @return Returns the previous value
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     * @see set(const NamePair &, const size_type &, const T &)
     */
    template<typename T>
    T set(const unsigned int &instance_id, const std::string &var_name, const size_type &index, const T &value);
    /**
     * Gets an environment property
     * @param name name used for accessing the property
     * @tparam T Type of the environmental property array to be created
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @see get(const NamePair &)
     */
    template<typename T>
    T get(const NamePair &name);
    /**
     * Convenience method: Gets an environment property
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param var_name name used for accessing the property
     * @tparam T Type of the environmental property array to be created
     * @throws InvalidEnvProperty If a property of the name does not exist
     */
    template<typename T>
    T get(const unsigned int &instance_id, const std::string &var_name);
    /**
     * Gets an environment property array
     * @param name name used for accessing the property array
     * @tparam T Type of the environmental property array to be created
     * @tparam N Length of the environmental property array to be created
     * @throws InvalidEnvProperty If a property array of the name does not exist
     */
    template<typename T, size_type N>
    std::array<T, N> get(const NamePair &name);
    /**
     * Convenience method: Gets an environment property array
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param var_name name used for accessing the property
     * @tparam T Type of the environmental property array to be created
     * @tparam N Length of the environmental property array to be created
     * @throws InvalidEnvProperty If a property array of the name does not exist
     * @see get(const NamePair &)
     */
    template<typename T, size_type N>
    std::array<T, N> get(const unsigned int &instance_id, const std::string &var_name);
    /**
     * Gets an element of an environment property array
     * @param name name used for accessing the property array
     * @param index Index of the element within the array
     * @tparam T Type of the value to be returned
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     */
    template<typename T>
    T get(const NamePair &name, const size_type &index);
#ifdef SWIG
    /**
     * Convenience method: Gets an environment property array
     * @param name name used for accessing the property array
     * @tparam T Type of the environmental property array to be created
     * @throws InvalidEnvProperty If a property array of the name does not exist
     * @see get(const NamePair &)
     */
    template<typename T>
    std::vector<T> getArray(const NamePair& name);
#endif
    /**
     * Convenience method: Gets an element of an environment property array
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param var_name name used for accessing the property
     * @tparam T Type of the value to be returned
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     * @see get(const NamePair &, const size_type &)
     */
    template<typename T>
    T get(const unsigned int &instance_id, const std::string &var_name, const size_type &index);
    /**
     * Removes an environment property
     * @param name name used for accessing the property
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @note This may be used to remove and recreate environment properties (and arrays) marked const
     */
    void remove(const NamePair &name);
    /**
     * Convenience method: Removes an environment property
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param var_name name used for accessing the property
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @note This may be used to remove and recreate environment properties (and arrays) marked const
     * @see remove(const NamePair &)
     */
    void remove(const unsigned int &instance_id, const std::string &var_name);
    /**
     * Returns all environment properties owned by a model to their default values
     * This means that properties inherited by a submodel will not be reset to their default values
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param desc The environment description (this is where the defaults are pulled from)
     * @todo This is not a particularly efficient implementation, as it updates them all individually.
     */
    void resetModel(const unsigned int &instance_id, const EnvironmentDescription &desc);
    /**
     * Returns whether the named env property exists
     * @param name name used for accessing the property
     */
    inline bool contains(const NamePair &name) const { return properties.find(name) != properties.end() || mapped_properties.find(name) != mapped_properties.end(); }
    /**
     * Convenience method: Returns whether the named env property exists
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param var_name name used for accessing the property
     * @see contains(const NamePair &)
     */
    inline bool contains(const unsigned int &instance_id, const std::string &var_name) const { return contains(toName(instance_id, var_name)); }
    /**
     * Returns whether the named env property is marked as const
     * @param name name used for accessing the property
     * @return true if the var is marked as constant (cannot be changed during simulation)
     * @throws InvalidEnvProperty If a property of the name does not exist
     */
    inline bool isConst(const NamePair &name) const {
        auto a = properties.find(name);
        if (a != properties.end())
            return a->second.isConst;
        const auto b = mapped_properties.find(name);
        if (b != mapped_properties.end()) {
            return b->second.isConst;
        }
        THROW InvalidEnvProperty("Environmental property with name '%u:%s' does not exist, "
            "in EnvironmentManager::isConst().",
            name.first, name.second.c_str());
    }
    /**
     * Convenience method: Returns whether the named env property is marked as const
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param var_name name used for accessing the property
     * @return true if the var is marked as constant (cannot be changed during simulation)
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @see isConst(const NamePair &)
     */
    inline bool isConst(const unsigned int &instance_id, const std::string &var_name) const { return isConst(toName(instance_id, var_name)); }
    /**
     * Returns the number of elements of the named env property (1 if not an array)
     * @param name name used for accessing the property
     * @throws InvalidEnvProperty If a property of the name does not exist
     */
    inline size_type length(const NamePair &name) const {
        auto a = properties.find(name);
        if (a != properties.end())
            return a->second.elements;
        const auto b = mapped_properties.find(name);
        if (b != mapped_properties.end()) {
            a = properties.find(b->second.masterProp);
            if (a != properties.end())
                return a->second.elements;
            THROW InvalidEnvProperty("Mapped environmental property with name '%u:%s' maps to missing property with name '%u:%s', "
                "in EnvironmentManager::type().",
                name.first, name.second.c_str(), b->second.masterProp.first, b->second.masterProp.second.c_str());
        }
        THROW InvalidEnvProperty("Environmental property with name '%u:%s' does not exist, "
            "in EnvironmentManager::length().",
            name.first, name.second.c_str());
    }
    /**
     * Convenience method: Returns the number of elements of the named env property (1 if not an array)
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param var_name name used for accessing the property
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @see length(const NamePair &)
     */
    inline size_type length(const unsigned int &instance_id, const std::string &var_name) const { return length(toName(instance_id, var_name)); }
    /**
     * Returns the variable type of named env property
     * @param name name used for accessing the property
     * @throws InvalidEnvProperty If a property of the name does not exist
     */
    inline std::type_index type(const NamePair &name) const {
        auto a = properties.find(name);
        if (a != properties.end())
            return a->second.type;
        const auto b = mapped_properties.find(name);
        if (b != mapped_properties.end()) {
            a = properties.find(b->second.masterProp);
            if (a != properties.end())
                return a->second.type;
            THROW InvalidEnvProperty("Mapped environmental property with name '%u:%s' maps to missing property with name '%u:%s', "
                "in EnvironmentManager::type().",
                name.first, name.second.c_str(), b->second.masterProp.first, b->second.masterProp.second.c_str());
        }
        THROW InvalidEnvProperty("Environmental property with name '%u:%s' does not exist, "
            "in EnvironmentManager::type().",
            name.first, name.second.c_str());
    }
    /**
     * Convenience method: Returns the variable type of named env property
     * @param instance_id instance_id of the CUDASimulation instance the property is attached to
     * @param var_name name used for accessing the property
     * @throws InvalidEnvProperty If a property of the name does not exist
     * @see type(const NamePair &)
     */
    inline std::type_index type(const unsigned int &instance_id, const std::string &var_name) const { return type(toName(instance_id, var_name)); }
    /**
     * Returns the available space remaining (bytes) for storing environmental properties
     */
    inline size_t freeSpace() const { return m_freeSpace; }
    /**
     * This is the string used to generate CURVE_NAMESPACE_HASH
     */
    static const char CURVE_NAMESPACE_STRING[23];
    /**
     * Hash never changes, so we store a copy at creation
     * Also ensure the device constexpr version matches
     */
    const Curve::NamespaceHash CURVE_NAMESPACE_HASH;
    /**
     * Returns read-only access to the properties map
     */
    const std::unordered_map<NamePair, EnvProp, NamePairHash> &getPropertiesMap() const {
        return properties;
    }
    /**
     * Returns readonly access to mapped properties
     */    
    const std::unordered_map<NamePair, MappedProp, NamePairHash> &getMappedProperties() const {
        return mapped_properties;
    }
    const void * getHostBuffer() const {
        return hc_buffer;
    }
    /**
     * Updates the copy of the environment property cache on the device
     * @param instance_id Used to update the specified instance's rtc too
     */
    void updateDevice(const unsigned int &instance_id);

 private:
    /**
     * Joins the two strings into a std::pair
     * @param instance_id becomes first item of pair
     * @param var_name becomes second item of pair
     * @return Returns std::make_pair(instance_id, var_name)
     */
    static NamePair toName(const unsigned int &instance_id, const std::string &var_name);
    /**
     * Returns the sum of the curve variable hash for the two items within name
     * @param name Pair of the two items to produce the curve value hash
     * @note Not static, because eventually we might need to use curve singleton
     */
    Curve::VariableHash toHash(const NamePair &name) const;
    /**
     * Common add handler
     */
    void add(const NamePair &name, const char *ptr, const size_t &len, const bool &isConst, const size_type &elements, const std::type_index &type);
    /**
     * Cleanup freeFragments
     * @param mergeProps Used by init to defragement whilst merging in new data
     * @param newmaps Namepairs of newly mapped properties, yet to to be setup (essentially ones not yet registered in curve)
     * @note any EnvPROP
     */
    void defragment(const DefragMap * mergeProps = nullptr, std::set<NamePair> newmaps = {});
    /**
     * This is the RTC version of defragment()
     * RTC Constant offsets are fixed at RTC time, and exist in their own constant block.
     * Therefore if the main constant cache is defragmented, and offsets change, the RTC constants will be incorrect.
     * Which fix this by maintaining a seperate constant cache per CUDASimulation instance
     * @param instance_id Instance id of the cuda agent model that owns the properties
     * @param master_instance_id Instance id of the parent model (If this isn't a submodel, pass instance_id here too
     * @param mergeProperties Ordered map of properties to be stored.
     */
    void buildRTCOffsets(const unsigned int &instance_id, const unsigned int &master_instance_id, const DefragMap &mergeProperties);
    /**
     * Useful for adding individual variables to RTC cache later on
     */
    void addRTCOffset(const NamePair &name);
    /**
     * Device pointer to the environment property buffer in __constant__ memory
     */
    const char *c_buffer;
    /**
     * Host copy of the device memory pointed to by c_buffer
     */
    char hc_buffer[MAX_BUFFER_SIZE];
    /**
     * Offset relative to c_buffer, where no more data has been stored
     */
    ptrdiff_t nextFree;
    /**
     * Unused space within c_buffer, including gaps in freeFragments
     */
    size_t m_freeSpace;
    /**
     * List of fragments remaining from deleted environment variables
     */
    std::list<OffsetLen> freeFragments;
    /**
     * Host copy of data related to each stored property
     */
    std::unordered_map<NamePair, EnvProp, NamePairHash> properties;
    /**
     * This lists all currently mapped properties, so their access can be redirected to the appropriate property
     */
    std::unordered_map<NamePair, MappedProp, NamePairHash> mapped_properties;
    /**
     * Map of model name to CUDASimulation for use in updating RTC values
     */
    std::unordered_map<unsigned int, const CUDASimulation&> cuda_agent_models;
    /**
     * Map of RTC caches per CUDASimulation instance
     * They are shared by submodels
     */
    std::unordered_map<unsigned int, std::shared_ptr<RTCEnvPropCache>> rtc_caches;
    /**
     * Flag indicating that curve has/hasn't been initialised yet on a device.
     */
    bool deviceInitialised;
    /*
     * Convenience fn for managing deviceRequiresUpdate
     * @param instance_id Sim instance id, UINT_MAX sets all
     */
    void setDeviceRequiresUpdateFlag(const unsigned int &instance_id = UINT_MAX);
    /**
     * These flags control what happens when updateDevice() is called
     * Their primary purpose is to cause the device memory to updated as lazily as possible
     */
    struct EnvUpdateFlags {
        /**
         * Update the device constant cache for main C env var storage
         */
        bool c_update_required = true;
        /**
         * Update the RTC environment cache for a specific CUDASimulation instance
         */
        bool rtc_update_required = true;
        /**
         * Additionally register all variables inside CURVE
         * This should only be triggered after a device reset, when EnvironmentManager::purge()  has been called
         * As properties are registered with curve when they are first added to EnvironmentManager
         **/
        bool curve_registration_required = false;
    };
    /**
     * Flag indicating whether the device copy is upto date
     * sim_instance_id:(C needs update, RTC needs update)
     */
    std::unordered_map<unsigned int, EnvUpdateFlags> deviceRequiresUpdate;
    /**
     * Function to initialise device-side portions of the environment manager
     */
    void initialiseDevice();
    /**
     * Remainder of class is singleton pattern
     */
    EnvironmentManager();
    /**
     * Wipes out host mirrors of device memory
     * Only really to be used after calls to cudaDeviceReset()
     * @note Only currently used after some tests
     */
    void purge();

 protected:
    /**
     * Returns the EnvironmentManager singleton instance
     */
    static EnvironmentManager& getInstance() {
        static EnvironmentManager instance;  // Guaranteed to be destroyed.
        return instance;                     // Instantiated on first use.
    }

    const CUDASimulation& getCUDASimulation(const unsigned int &instance_id);
    /**
     * Update the copy of the env var that exists in the rtc_cache to match the main cache
     * @param name namepair of the variable to be updated
     */
    void updateRTCValue(const NamePair &name);

 public:
    // Public deleted creates better compiler errors
    EnvironmentManager(EnvironmentManager const&) = delete;
    void operator=(EnvironmentManager const&) = delete;
};

/**
 * Constructors
 */
template<typename T>
void EnvironmentManager::add(const NamePair &name, const T &value, const bool &isConst) {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    if (contains(name)) {
        THROW DuplicateEnvProperty("Environmental property with name '%u:%s' already exists, "
            "in EnvironmentManager::add().",
            name.first, name.second.c_str());
    }
    add(name, reinterpret_cast<const char*>(&value), sizeof(T), isConst, 1, typeid(T));
}
template<typename T>
void EnvironmentManager::add(const unsigned int &instance_id, const std::string &var_name, const T &value, const bool &isConst) {
    add<T>(toName(instance_id, var_name), value, isConst);
}
template<typename T, EnvironmentManager::size_type N>
void EnvironmentManager::add(const NamePair &name, const std::array<T, N> &value, const bool &isConst) {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    if (contains(name)) {
        THROW DuplicateEnvProperty("Environmental property with name '%u:%s' already exists, "
            "in EnvironmentManager::add().",
            name.first, name.second.c_str());
    }
    add(name, reinterpret_cast<const char*>(value.data()), N * sizeof(T), isConst, N, typeid(T));
}
template<typename T, EnvironmentManager::size_type N>
void EnvironmentManager::add(const unsigned int &instance_id, const std::string &var_name, const std::array<T, N> &value, const bool &isConst) {
    add<T, N>(toName(instance_id, var_name), value, isConst);
}

/**
 * Setters
 */
template<typename T>
T EnvironmentManager::set(const NamePair &name, const T &value) {
    const std::type_index typ_id = type(name);
    if (typ_id != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environmental property ('%u:%s') type (%s) does not match template argument T (%s), "
            "in EnvironmentManager::set().",
            name.first, name.second.c_str(), typ_id.name(), typeid(T).name());
    }
    if (isConst(name)) {
        THROW ReadOnlyEnvProperty("Environmental property ('%u:%s') is marked as const and cannot be changed, "
            "in EnvironmentManager::set().",
            name.first, name.second.c_str());
    }
    // Copy old data to return
    T rtn = get<T>(name);
    // Find property offset
    ptrdiff_t buffOffset = 0;
    const auto a = properties.find(name);
    if (a != properties.end()) {
        buffOffset = a->second.offset;
    } else {
        buffOffset = properties.at(mapped_properties.at(name).masterProp).offset;
    }
    // Store data
    memcpy(hc_buffer + buffOffset, &value, sizeof(T));
    // Do rtc too
    updateRTCValue(name);
    // Set device update flag
    setDeviceRequiresUpdateFlag(name.first);

    return rtn;
}
template<typename T>
T EnvironmentManager::set(const unsigned int &instance_id, const std::string &var_name, const T &value) {
    return set<T>(toName(instance_id, var_name), value);
}
template<typename T, EnvironmentManager::size_type N>
std::array<T, N> EnvironmentManager::set(const NamePair &name, const std::array<T, N> &value) {
    const std::type_index typ_id = type(name);
    if (typ_id != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environmental property array ('%u:%s') type (%s) does not match template argument T (%s), "
            "in EnvironmentManager::set().",
            name.first, name.second.c_str(), typ_id.name(), typeid(T).name());
    }
    if (isConst(name)) {
        THROW ReadOnlyEnvProperty("Environmental property array ('%u:%s') is marked as const and cannot be changed, "
            "in EnvironmentManager::set().",
            name.first, name.second.c_str());
    }
    const size_type array_len = length(name);
    if (array_len != N) {
        THROW OutOfBoundsException("Length of named environmental property array (%u) does not match template argument N (%u)! "
            "in EnvironmentManager::set().",
            array_len, N);
    }
    // Find property offset
    ptrdiff_t buffOffset = 0;
    const auto a = properties.find(name);
    if (a != properties.end()) {
        buffOffset = a->second.offset;
    } else {
        buffOffset = properties.at(mapped_properties.at(name).masterProp).offset;
    }
    // Copy old data to return
    std::array<T, N> rtn = get<T, N>(name);
    // Store data
    memcpy(hc_buffer + buffOffset, value.data(), N * sizeof(T));
    // Do rtc too
    updateRTCValue(name);
    // Set device update flag
    setDeviceRequiresUpdateFlag(name.first);

    return rtn;
}
template<typename T, EnvironmentManager::size_type N>
std::array<T, N> EnvironmentManager::set(const unsigned int &instance_id, const std::string &var_name, const std::array<T, N> &value) {
    return set<T, N>(toName(instance_id, var_name), value);
}
#ifdef SWIG
template<typename T>
std::vector<T> EnvironmentManager::setArray(const NamePair& name, const std::vector<T>& value) {
    const std::type_index typ_id = type(name);
    if (typ_id != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environmental property array ('%u:%s') type (%s) does not match template argument T (%s), "
            "in EnvironmentManager::set().",
            name.first, name.second.c_str(), typ_id.name(), typeid(T).name());
    }
    if (isConst(name)) {
        THROW ReadOnlyEnvProperty("Environmental property array ('%u:%s') is marked as const and cannot be changed, "
            "in EnvironmentManager::set().",
            name.first, name.second.c_str());
    }
    const size_type array_len = length(name);
    if (array_len != value.size()) {
        THROW OutOfBoundsException("Length of named environmental property array (%u) does not match length of provided array (%llu)! "
            "in EnvironmentManager::set().",
            array_len, value.size());
    }
    // Find property offset
    ptrdiff_t buffOffset = 0;
    const auto a = properties.find(name);
    if (a != properties.end()) {
        buffOffset = a->second.offset;
    } else {
        buffOffset = properties.at(mapped_properties.at(name).masterProp).offset;
    }
    // Copy old data to return
    std::vector<T> rtn(value.size());
    if (a != properties.end()) {
        memcpy(rtn.data(), reinterpret_cast<T*>(hc_buffer + buffOffset), array_len * sizeof(T));
    } else {
        memcpy(rtn.data(), reinterpret_cast<T*>(hc_buffer + buffOffset), array_len * sizeof(T));
    }
    // Store data
    memcpy(hc_buffer + buffOffset, value.data(), value.size() * sizeof(T));
    // Do rtc too
    updateRTCValue(name);
    // Set device update flag
    setDeviceRequiresUpdateFlag(name.first);

    return rtn;
}
#endif
template<typename T>
T EnvironmentManager::set(const NamePair &name, const size_type &index, const T &value) {
    const std::type_index typ_id = type(name);
    if (typ_id != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environmental property array ('%u:%s') type (%s) does not match template argument T (%s), "
            "in EnvironmentManager::set().",
            name.first, name.second.c_str(), typ_id.name(), typeid(T).name());
    }
    if (isConst(name)) {
        THROW ReadOnlyEnvProperty("Environmental property array ('%u:%s') is marked as const and cannot be changed, "
            "in EnvironmentManager::set().",
            name.first, name.second.c_str());
    }
    const size_type array_len = length(name);
    if (index >= array_len) {
        THROW OutOfBoundsException("Index(%u) exceeds named environmental property array's length (%u), "
            "in EnvironmentManager::set().",
            index, array_len);
    }
    // Find property offset
    ptrdiff_t buffOffset = 0;
    const auto a = properties.find(name);
    if (a != properties.end()) {
        buffOffset = a->second.offset + index * sizeof(T);
    } else {
        const auto &master_name = mapped_properties.at(name).masterProp;
        buffOffset = properties.at(master_name).offset + index * sizeof(T);
    }
    // Copy old data to return
    T rtn = *reinterpret_cast<T*>(hc_buffer + buffOffset);
    // Store data
    memcpy(hc_buffer + buffOffset, &value, sizeof(T));
    // Do rtc too
    updateRTCValue(name);
    // Set device update flag
    setDeviceRequiresUpdateFlag(name.first);

    return rtn;
}
template<typename T>
T EnvironmentManager::set(const unsigned int &instance_id, const std::string &var_name, const size_type &index, const T &value) {
    return set<T>(toName(instance_id, var_name), index, value);
}

/**
 * Getters
 */
template<typename T>
T EnvironmentManager::get(const NamePair &name) {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    std::type_index typ_id = type(name);
    if (typ_id != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environmental property ('%u:%s') type (%s) does not match template argument T (%s), "
            "in EnvironmentManager::get().",
            name.first, name.second.c_str(), typ_id.name(), typeid(T).name());
    }
    // Copy old data to return
    const auto a = properties.find(name);
    if (a != properties.end())
        return *reinterpret_cast<T*>(hc_buffer + a->second.offset);
    return *reinterpret_cast<T*>(hc_buffer + properties.at(mapped_properties.at(name).masterProp).offset);
}
template<typename T>
T EnvironmentManager::get(const unsigned int &instance_id, const std::string &var_name) {
    return get<T>(toName(instance_id, var_name));
}
template<typename T, EnvironmentManager::size_type N>
std::array<T, N> EnvironmentManager::get(const NamePair &name) {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    const std::type_index typ_id = type(name);
    if (typ_id != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environmental property array ('%u:%s') type (%s) does not match template argument T (%s), "
            "in EnvironmentManager::get().",
            name.first, name.second.c_str(), typ_id.name(), typeid(T).name());
    }
    const size_type array_len = length(name);
    if (array_len != N) {
        THROW OutOfBoundsException("Length of named environmental property array (%u) does not match template argument N (%u)! "
            "in EnvironmentManager::get().",
            array_len, N);
    }
    // Copy old data to return
    std::array<T, N> rtn;
    const auto a = properties.find(name);
    if (a != properties.end()) {
        memcpy(rtn.data(), reinterpret_cast<T*>(hc_buffer + a->second.offset), N * sizeof(T));
    } else {
        memcpy(rtn.data(), reinterpret_cast<T*>(hc_buffer + properties.at(mapped_properties.at(name).masterProp).offset), N * sizeof(T));
    }
    return rtn;
}
template<typename T, EnvironmentManager::size_type N>
std::array<T, N> EnvironmentManager::get(const unsigned int &instance_id, const std::string &var_name) {
    return get<T, N>(toName(instance_id, var_name));
}
template<typename T>
T EnvironmentManager::get(const NamePair &name, const size_type &index) {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    const std::type_index typ_id = type(name);
    if (typ_id != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environmental property array ('%u:%s') type (%s) does not match template argument T (%s), "
            "in EnvironmentManager::get().",
            name.first, name.second.c_str(), typ_id.name(), typeid(T).name());
    }
    const size_type array_len = length(name);
    if (index >= array_len) {
        THROW OutOfBoundsException("Index(%u) exceeds named environmental property array's length (%u), "
            "in EnvironmentManager::set().",
            index, array_len);
    }
    // Copy old data to return
    const auto a = properties.find(name);
    if (a != properties.end())
        return *reinterpret_cast<T*>(hc_buffer + a->second.offset + index * sizeof(T));
    return *reinterpret_cast<T*>(hc_buffer + properties.at(mapped_properties.at(name).masterProp).offset + index * sizeof(T));
}
#ifdef SWIG
template<typename T>
std::vector<T> EnvironmentManager::getArray(const NamePair& name) {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,
        "Only arithmetic types can be used as environmental properties");
    const std::type_index typ_id = type(name);
    if (typ_id != std::type_index(typeid(T))) {
        THROW InvalidEnvPropertyType("Environmental property array ('%u:%s') type (%s) does not match template argument T (%s), "
            "in EnvironmentManager::get().",
            name.first, name.second.c_str(), typ_id.name(), typeid(T).name());
    }
    const size_type array_len = length(name);
    // Copy old data to return
    std::vector<T> rtn(static_cast<size_t>(array_len));
    const auto a = properties.find(name);
    if (a != properties.end()) {
        memcpy(rtn.data(), reinterpret_cast<T*>(hc_buffer + a->second.offset), array_len * sizeof(T));
    } else {
        memcpy(rtn.data(), reinterpret_cast<T*>(hc_buffer + properties.at(mapped_properties.at(name).masterProp).offset), array_len * sizeof(T));
    }
    return rtn;
}
#endif
template<typename T>
T EnvironmentManager::get(const unsigned int &instance_id, const std::string &var_name, const size_type &index) {
    return get<T>(toName(instance_id, var_name), index);
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_ENVIRONMENTMANAGER_CUH_
