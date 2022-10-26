#ifndef INCLUDE_FLAMEGPU_RUNTIME_UTILITY_ENVIRONMENTMANAGER_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_UTILITY_ENVIRONMENTMANAGER_CUH_

#include <cuda_runtime.h>

#include <map>
#include <string>
#include <memory>
#include <set>
#include <unordered_map>
#include <functional>
#include <utility>
#include <vector>

#include "flamegpu/defines.h"
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/runtime/detail/curve/HostCurve.cuh"
#include "flamegpu/util/type_decode.h"
#include "flamegpu/util/Any.h"

namespace flamegpu {
struct SubEnvironmentData;
class EnvironmentDescription;
class CUDASimulation;

/**
 * This class manages the regular (not macro) environment properties for a single simulation instance
 *
 * It packs properties into a compact cache, which can be copied to device memory for use with DeviceEnvironment
 * When properties are mapped between submodels, a weak_ptr is held in the mapping where necessary to allow updates to propagate
 * Propogation is carried out, by first following the chain up through parents, and then updating all direct children of the property from that parent
 */
class EnvironmentManager : public std::enable_shared_from_this<EnvironmentManager> {
    /**
     * CUDASimulation::simulate(const RunPlan&) requires access to EnvironmentManager::setPropertyDirect()
     * CUDASimulation::initEnvironmentMgr() requires access to EnvironmentManager::properties and EnvironmentManager::setPropertyDirect()
     * The latter could probably be moved into EnvironmentManager (behind a private method)
     */
    friend class CUDASimulation;

 private:
    /**
     * Used to group items required by properties
     */
    struct EnvProp {
        /**
         * @param _offset Offset into c_buffer/hc_buffer
         * @param _length Length of associated storage
         * @param _isConst Is the stored data constant
         * @param _elements How many elements does the stored data contain (1 if not array)
         * @param _type Type of property (from typeid())
         */
        EnvProp(const ptrdiff_t& _offset, const size_t _length, const bool _isConst, const size_type _elements, const std::type_index& _type)
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
    };
    /**
     * Used to represent properties of a mapped environment property
     */
    struct MappedProp {
        /**
         * @param _remoteName Remote property of mapping
         * @param _remoteEnv Will be used to forward updates (otherwise assumed the mapping is const at this end)
         * @param _isConst If true, updates will not propagated
         */
        MappedProp(const std::string& _remoteName, const std::shared_ptr<EnvironmentManager> &_remoteEnv, bool _isConst = false)
            : remoteName(_remoteName)
            , remoteEnv(_remoteEnv)
            , isConst(_isConst) {}
        const std::string remoteName;
        const std::weak_ptr<EnvironmentManager> remoteEnv;
        const bool isConst;
    };
    /**
     * This structure is a clone of EnvProp
     * However, instead of offset (which points to an offset into hc_buffer)
     * data is available, which points to host memory
     */
    struct DefragProp {
        /**
        * @param _data Pointer to the data in host memory
        * @param _length Length of associated storage
        * @param _isConst Is the stored data constant
        * @param _elements How many elements does the stored data contain (1 if not array)
        * @param _type Type of property (from typeid())
        */
        DefragProp(void *_data, const size_t _length, bool _isConst, const size_type _elements, const std::type_index &_type)
            : data(_data),
            length(_length),
            isConst(_isConst),
            elements(_elements),
            type(_type),
            offset(0) { }
        void *data;
        size_t length;
        bool isConst;
        size_type elements;
        const std::type_index type;
        ptrdiff_t offset;
    };
    /**
     * Transparent operators for DefragMap
     * This allows them to be secondarily ordered based on name if size is equal
     */
    friend bool operator<(const std::pair<size_t, std::string>& fk, const size_t lk) { return fk.first < lk; }
    friend bool operator<(const size_t lk, const std::pair<size_t, std::string>& fk) { return lk < fk.first; }
    friend bool operator<(const std::pair<size_t, std::string>& fk1, const std::pair<size_t, std::string>& fk2) {
        if (fk1.first == fk2.first) {
            // If size is equal, order by name
            return fk1.second < fk2.second;
        }
        return fk1.first < fk2.first;
    }
    /**
     * Typedef for the map used for defragmentation
     * The map is ordered by key of type size, therefore a reverse sort creates aligned data
     * Specify a transparent operator, to allow us to operate only over size_t part of key
     */
    typedef std::multimap<std::pair<size_t, std::string>, DefragProp, std::less<>> DefragMap;

 public:
    /**
     * Initialises a model's environment property cache
     * @param desc environment properties description to use
    */
     [[nodiscard]] static std::shared_ptr<EnvironmentManager> create(const EnvironmentDescription& desc) {
        std::shared_ptr<EnvironmentManager> rtn(new EnvironmentManager());  // Can't use make_shared with private constructor!
        rtn->init(desc);
        return rtn;
    }
    /**
     * Initialises a submodel's environment property cache
     * Links a submodel's mapped environment properties with their master property
     * @param desc environment properties description to use
     * @param parent_environment EnvironmentManager of the parent of the submodel, used to initialise mappings
     * @param mapping Metadata for which environment properties are mapped between master and submodels
     */
     [[nodiscard]] static std::shared_ptr<EnvironmentManager> create(const EnvironmentDescription& desc, const std::shared_ptr<EnvironmentManager>& parent_environment, const SubEnvironmentData& mapping) {
        std::shared_ptr<EnvironmentManager> rtn(new EnvironmentManager());  // Can't use make_shared with private constructor!
        rtn->init(desc, parent_environment, mapping);
        return rtn;
    }
    ~EnvironmentManager();
    EnvironmentManager(EnvironmentManager const&) = delete;
    void operator=(EnvironmentManager const&) = delete;
    /**
     * Sets an environment property
     * @param name name used for accessing the property
     * @param value value to set the property
     * @tparam T Type of the environmental property array to be created
     * @return Returns the previous value
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::ReadOnlyEnvProperty If the named property is marked as const
     */
    template<typename T>
    T setProperty(const std::string& name, T value);
    /**
     * Sets an environment property array
     * @param name name used for accessing the property array
     * @param value value to set the property array
     * @tparam T Type of the environmental property array to be created
     * @tparam N Length of the environmental property array to be created
     * @return Returns the previous value
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::ReadOnlyEnvProperty If the named property is marked as const
     */
    template<typename T, size_type N>
    std::array<T, N> setProperty(const std::string& name, const std::array<T, N>& value);
    /**
     * Sets an element of an environment property array
     * @param name name used for accessing the property array
     * @param index Index of the element within the array
     * @param value value to set the element of the property array
     * @tparam T Type of the environmental property array to be created
     * @tparam N (Optional) The length of the array variable, available for parity with other APIs, checked if provided
     * @return Returns the previous value
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     */
    template<typename T, size_type N = 0>
    T setProperty(const std::string& name,  size_type index, T value);
#ifdef SWIG
    /**
     * Sets an environment property array
     * @param name name used for accessing the property array
     * @param value value to set the property array
     * @tparam T Type of the environmental property array to be created
     * @return Returns the previous value
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws exception::ReadOnlyEnvProperty If the named property is marked as const
     */
    template<typename T>
    std::vector<T> setPropertyArray(const std::string& name, const std::vector<T>& value);
#endif
    /**
     * Gets an environment property
     * @param name name used for accessing the property
     * @tparam T Type of the environmental property array to be created
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @see get(const NamePair &)
     */
    template<typename T>
    T getProperty(const std::string& name);
    /**
     * Gets an environment property array
     * @param name name used for accessing the property array
     * @tparam T Type of the environmental property array to be created
     * @tparam N Length of the environmental property array to be created
     * @throws exception::InvalidEnvProperty If a property array of the name does not exist
     */
    template<typename T, size_type N>
    std::array<T, N> getProperty(const std::string& name);
    /**
     * Gets an element of an environment property array
     * @param name name used for accessing the property array
     * @param index Index of the element within the array
     * @tparam T Type of the value to be returned
     * @tparam N (Optional) The length of the array variable, available for parity with other APIs, checked if provided
     * @throws exception::InvalidEnvProperty If a property of the name does not exist
     * @throws std::out_of_range
     */
    template<typename T, size_type N = 0>
    T getProperty(const std::string& name, size_type index);
#ifdef SWIG
    /**
     * Convenience method: Gets an environment property array
     * @param name name used for accessing the property array
     * @tparam T Type of the environmental property array to be created
     * @throws exception::InvalidEnvProperty If a property array of the name does not exist
     * @see get(const NamePair &)
     */
    template<typename T>
    std::vector<T> getPropertyArray(const std::string& name);
#endif
    /**
     * Returns all environment properties owned by a model to their default values
     * This means that properties inherited by a submodel will not be reset to their default values
     * @param desc The environment description (this is where the defaults are pulled from)
     * @todo This is not a particularly efficient implementation, as it updates them all individually.
     */
    void resetModel(const EnvironmentDescription& desc);
    /**
     * Copies the environment property cache to a device buffer
     * @param stream Cuda stream to perform memcpys on
     */
    void updateDevice_async(cudaStream_t stream) const;
    /**
     * Returns the minimum buffer size required to call updateDevice_async()
     *
     * @return The size of h_buffer
     */
    size_t getBufferLen() const { return h_buffer_len; }
    /**
     * Used by IO methods to efficiently access environment
     */
    const void* getHostBuffer() const {
        return h_buffer;
    }
    /**
     * Used by agent functions to access the environment
     */
    const void* getDeviceBuffer() const {
        return d_buffer;
    }
    /**
     * Returns the full map of properties
     */
    const std::unordered_map<std::string, EnvProp> &getPropertiesMap() const {
        return properties;
    }

 private:
    /**
     * static EnvironmentDescription::create(const EnvironmentDescription&)
     */
    void init(const EnvironmentDescription& desc);
    /**
     * static EnvironmentDescription::create(const EnvironmentDescription&, const std::shared_ptr<EnvironmentManager>&, const SubEnvironmentData&)
     */
    void init(const EnvironmentDescription& desc, const std::shared_ptr<EnvironmentManager>& parent_environment, const SubEnvironmentData& mapping);
    /**
     * Called by the submodel variant of init() to notify the parent model of properties which are mapped
     * @param parent_name Name of the property in the callees environment
     * @param sub_name Name of the property in the callers environment
     * @param sub_environment Shared pointer to the callers environment, to provide the updates
     */
    void linkMappedProperty(const std::string &parent_name, const std::string& sub_name, const std::shared_ptr<EnvironmentManager>& sub_environment);
    /**
     * Propagate the value of the named property to any mapped environments
     *
     * Trace up through parents until no parent is available
     * Then set all child properties from this level
     *
     * @param property_name Name of the variable to be propagated
     * @param src_ptr Pointer to use as the source for the data to propagate
     */
    void propagateMappedPropertyValue(const std::string &property_name, const char *src_ptr);
    /**
     * Internal direct setter
     * This performs no safety checks, it assumes that src_ptr points to an appropriate buffer containing data of the right type/length
     * @param property_name Name of the property to set
     * @param src_ptr Pointer to buffer containing data to use as the source of the new value
     */
    void setPropertyDirect(const std::string& property_name, const char * src_ptr);
    /**
     * Locates and returns a reference to the metadata structure for the named property
     * If the values provides are incorrect an exception is raised
     * @param property_name Name of the variable to be located
     * @param setter If true, the property will be check to ensure it is not marked const
     * @param length If non-zero, the property will check the length of the base type array matches. The array length specified by the user (via the template parameter N)
     * @tparam T If non-void, the property which check the type T matches the the properties type
     * @throws exception::InvalidEnvPropertyType If the base type of T, does not match the base type of the specified property
     * @throws exception::OutOfBoundsException If the calculated length from T and length does not match the length of the specified property
     * @throws exception::ReadOnlyEnvProperty If setter is passed as true to a read only property
     */
    template<typename T>
    const EnvProp &findProperty(const std::string& property_name, bool setter, size_type length) const;
    /**
     * Returns the named property as an any generic
     * @param property_name Name of the variable to be returned
     */
    util::Any getPropertyAny(const std::string &property_name) const;
    /**
     * Host copy of the device memory pointed to by d_buffer
     */
    char *h_buffer = nullptr;
    char *d_buffer = nullptr;
    /**
     * True when the contents of d_buffer matches h_buffer, this is used to skip unnecessary cudaMemcpys
     */
    mutable bool d_buffer_ready = false;
    /**
     * Length of h_buffer
     */
    size_t h_buffer_len = 0;
    /**
     * Metadata related to each stored property
     */
    std::unordered_map<std::string, EnvProp> properties{};
    /**
     * This lists all currently mapped properties, between this environment and a child environment
     */
    std::multimap<std::string, MappedProp> mapped_child_properties{};
    /**
     * This lists all currently mapped properties, between this environment and a parent environment
     */
    std::map<std::string, MappedProp> mapped_parent_properties{};
    /**
     * Private default constructor, use the static factory to create
     * @see create()
     */
    EnvironmentManager() = default;
};

template<typename T>
const EnvironmentManager::EnvProp& EnvironmentManager::findProperty(const std::string& property_name, const bool setter, const size_type length) const {
    // Limited to Arithmetic types
    // Compound types would allow host pointers inside structs to be passed
    static_assert(std::is_arithmetic<typename type_decode<T>::type_t>::value || std::is_enum<typename type_decode<T>::type_t>::value || std::is_void<typename type_decode<T>::type_t>::value,
        "Only arithmetic types can be used as environmental properties");
    const auto a = properties.find(property_name);
    if (a != properties.end()) {
        if (std::type_index(typeid(T)) != std::type_index(typeid(void)) && a->second.type != std::type_index(typeid(typename type_decode<T>::type_t))) {
            THROW exception::InvalidEnvPropertyType("Environmental property with name '%s', type (%s) does not match template argument T (%s), "
                "in EnvironmentManager::setProperty().",
                property_name.c_str(), a->second.type.name(), typeid(typename type_decode<T>::type_t).name());
        } else if (length && a->second.elements != type_decode<T>::len_t * length) {
            THROW exception::OutOfBoundsException("Environmental property with name '%s', base length (%u) does not match provided base length (%u), "
                "in EnvironmentManager::setProperty().",
                property_name.c_str(), a->second.elements, type_decode<T>::len_t * length);
        } else if (setter && a->second.isConst) {
            THROW exception::ReadOnlyEnvProperty("Environmental property with name '%s' is marked as const and cannot be changed, "
                "in EnvironmentManager::setProperty().",
                property_name.c_str());
        }
        // Check this here, rather than in 4 separate methods
        if (setter)
            d_buffer_ready = false;
        return a->second;
    }
    THROW exception::InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentManager::find().",
        property_name.c_str());
}
/**
 * Setters
 */
template<typename T>
T EnvironmentManager::setProperty(const std::string &name, const T value) {
    const EnvProp &prop = findProperty<T>(name, true, 1);
    // Copy old data to return
    T rtn;
    char* const dest_ptr = h_buffer + prop.offset;
    memcpy(&rtn, dest_ptr, sizeof(T));
    // Store data
    memcpy(dest_ptr, &value, sizeof(T));
    // Notify children
    propagateMappedPropertyValue(name, dest_ptr);
    return rtn;
}
template<typename T, flamegpu::size_type N>
std::array<T, N> EnvironmentManager::setProperty(const std::string &name, const std::array<T, N> &value) {
    const EnvProp& prop = findProperty<T>(name, true, N);
    // Copy old data to return
    std::array<T, N> rtn;
    char* const dest_ptr = h_buffer + prop.offset;
    memcpy(rtn.data(), dest_ptr, sizeof(T) * N);
    // Store data
    memcpy(dest_ptr, value.data(), sizeof(T) * N);
    // Notify children
    propagateMappedPropertyValue(name, dest_ptr);
    return rtn;
}
template<typename T, flamegpu::size_type N>
T EnvironmentManager::setProperty(const std::string &name, const size_type index, const T value) {
    const EnvProp& prop = findProperty<T>(name, true, 0);
    if (N && N != prop.elements / type_decode<T>::len_t) {
        THROW exception::OutOfBoundsException("Environmental property with name '%s', array length mismatch (%u != %u), "
            "in EnvironmentManager::setProperty().",
            name.c_str(), N, prop.elements / type_decode<T>::len_t);
    } else if (index >= prop.elements / type_decode<T>::len_t) {
        THROW exception::OutOfBoundsException("Environmental property with name '%s', index (%u) exceeds named environmental property array's length (%u), "
            "in EnvironmentManager::setProperty().",
            name.c_str(), index, prop.elements / type_decode<T>::len_t);
    }
    // Copy old data to return
    T rtn;
    char* const dest_ptr = h_buffer + prop.offset;
    memcpy(&rtn, dest_ptr + index * sizeof(T), sizeof(T));
    // Store data
    memcpy(dest_ptr + index * sizeof(T), &value, sizeof(T));
    // Notify children
    propagateMappedPropertyValue(name, dest_ptr);
    return rtn;
}
#ifdef SWIG
template<typename T>
std::vector<T> EnvironmentManager::setPropertyArray(const std::string& name, const std::vector<T>& value) {
    const EnvProp& prop = findProperty<T>(name, true, static_cast<unsigned int>(value.size()));
    // Copy old data to return
    std::vector<T> rtn;
    rtn.resize(value.size());
    char* const dest_ptr = h_buffer + prop.offset;
    memcpy(rtn.data(), dest_ptr, sizeof(T) * value.size());
    // Store data
    memcpy(dest_ptr, value.data(), sizeof(T) * value.size());
    // Notify children
    propagateMappedPropertyValue(name, dest_ptr);
    return rtn;
}
#endif
/**
 * Getters
 */
template<typename T>
T EnvironmentManager::getProperty(const std::string &name) {
    const EnvProp& prop = findProperty<T>(name, false, 1);
    // Copy data to return
    T rtn;
    memcpy(&rtn, h_buffer + prop.offset, sizeof(T));
    return rtn;
}
template<typename T, flamegpu::size_type N>
std::array<T, N> EnvironmentManager::getProperty(const std::string &name) {
    const EnvProp& prop = findProperty<T>(name, false, N);
    // Copy old data to return
    std::array<T, N> rtn;
    memcpy(rtn.data(), h_buffer + prop.offset, sizeof(T) * N);
    return rtn;
}
template<typename T, flamegpu::size_type N>
T EnvironmentManager::getProperty(const std::string &name, const size_type index) {
    const EnvProp& prop = findProperty<T>(name, false, 0);
    if (N && N != prop.elements / type_decode<T>::len_t) {
        THROW exception::OutOfBoundsException("Environmental property with name '%s', array length mismatch (%u != %u), "
            "in EnvironmentManager::getProperty().",
            name.c_str(), N, prop.elements / type_decode<T>::len_t);
    } else if (index >= prop.elements / type_decode<T>::len_t) {
        THROW exception::OutOfBoundsException("Environmental property with name '%s', index (%u) exceeds named environmental property array's length (%u), "
            "in EnvironmentManager::getProperty().",
            name.c_str(), index, prop.elements / type_decode<T>::len_t);
    }
    // Copy old data to return
    T rtn;
    memcpy(&rtn, h_buffer + prop.offset + index * sizeof(T), sizeof(T));
    return rtn;
}
#ifdef SWIG
template<typename T>
std::vector<T> EnvironmentManager::getPropertyArray(const std::string &name) {
    const EnvProp& prop = findProperty<T>(name, false, 0);
    // Copy old data to return
    const unsigned int N = prop.elements / type_decode<T>::len_t;
    std::vector<T> rtn;
    rtn.resize(N);
    memcpy(rtn.data(), h_buffer + prop.offset, sizeof(T) * N);
    return rtn;
}
#endif

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_UTILITY_ENVIRONMENTMANAGER_CUH_
