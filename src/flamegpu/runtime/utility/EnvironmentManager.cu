#include "flamegpu/runtime/utility/EnvironmentManager.cuh"

#include <cassert>
#include <memory>

#include "flamegpu/gpu/CUDAErrorChecking.cuh"
#include "flamegpu/runtime/utility/DeviceEnvironment.cuh"
#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/model/SubEnvironmentData.h"
#include "flamegpu/gpu/CUDASimulation.h"
#include "flamegpu/util/nvtx.h"

namespace flamegpu {

/**
 * Internal namespace to hide __constant__ declarations from modeller
 */
namespace detail {
    /**
     * Managed by HostEnvironment, holds all environment properties
     */
    __constant__ char c_envPropBuffer[EnvironmentManager::MAX_BUFFER_SIZE];
}  // namespace detail

std::mutex EnvironmentManager::instance_mutex;
const char EnvironmentManager::CURVE_NAMESPACE_STRING[23] = "ENVIRONMENT_PROPERTIES";

EnvironmentManager::EnvironmentManager() :
    CURVE_NAMESPACE_HASH(Curve::variableRuntimeHash(CURVE_NAMESPACE_STRING)),
    nextFree(0),
    m_freeSpace(EnvironmentManager::MAX_BUFFER_SIZE),
    freeFragments(),
    deviceInitialised(false) { }

void EnvironmentManager::purge() {
    std::unique_lock<std::shared_timed_mutex> lock(mutex);
    std::unique_lock<std::shared_timed_mutex> deviceRequiresUpdate_lock(deviceRequiresUpdate_mutex);
    deviceInitialised = false;
    for (auto &a : deviceRequiresUpdate) {
        a.second.c_update_required = true;
        a.second.rtc_update_required = true;
        a.second.curve_registration_required = true;
    }
    deviceRequiresUpdate_lock.unlock();
    // We are now able to only purge the device stuff after device reset?
    // freeFragments.clear();
    // m_freeSpace = EnvironmentManager::MAX_BUFFER_SIZE;
    // nextFree = 0;
    // cuda_agent_models.clear();
    // properties.clear();
    // mapped_properties.clear();
    // rtc_caches.clear();
    initialiseDevice();
}

void EnvironmentManager::init(const unsigned int &instance_id, const EnvironmentDescription &desc, bool isPureRTC) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex);
    // Error if reinit
    for (auto &&i : properties) {
        if (i.first.first == instance_id) {
            THROW exception::EnvDescriptionAlreadyLoaded("Environment description with same instance id '%u' is already loaded, "
                "in EnvironmentManager::init().",
                instance_id);
        }
    }

    // Add to device requires update map
    std::unique_lock<std::shared_timed_mutex> deviceRequiresUpdate_lock(deviceRequiresUpdate_mutex);
    deviceRequiresUpdate.emplace(instance_id, EnvUpdateFlags());
    deviceRequiresUpdate_lock.unlock();

    // Build a DefragMap to send to defragger method
    DefragMap orderedProperties;
    size_t newSize = 0;
    for (auto _i = desc.properties.begin(); _i != desc.properties.end(); ++_i) {
        const auto &i = _i->second;
        NamePair name = toName(instance_id, _i->first);
        DefragProp prop = DefragProp(i.data.ptr, i.data.length, i.isConst, i.data.elements, i.data.type);
        const size_t typeSize = i.data.length / i.data.elements;
        orderedProperties.emplace(std::make_pair(typeSize, name), prop);
        newSize += i.data.length;
    }
    if (newSize > m_freeSpace) {
        // Ran out of constant cache space! (this can only trigger when a DefragMap is passed)
        // Arguably this check should be performed by init()
        THROW exception::OutOfMemory("Insufficient EnvProperty memory to create new properties,"
            "in EnvironmentManager::init().");
    }
    // Defragment to rebuild it properly
    defragment(Curve::getInstance(), &orderedProperties, {}, isPureRTC);
    // Setup RTC version
    buildRTCOffsets(instance_id, instance_id, orderedProperties);
}
void EnvironmentManager::init(const unsigned int &instance_id, const EnvironmentDescription &desc, bool isPureRTC, const unsigned int &master_instance_id, const SubEnvironmentData &mapping) {
    assert(deviceRequiresUpdate.size());  // submodel init should never be called first, requires parent init first for mapping
    std::unique_lock<std::shared_timed_mutex> lock(mutex);

    // Add to device requires update map
    std::unique_lock<std::shared_timed_mutex> deviceRequiresUpdate_lock(deviceRequiresUpdate_mutex);
    deviceRequiresUpdate.emplace(instance_id, EnvUpdateFlags());
    deviceRequiresUpdate_lock.unlock();

    // Error if reinit
    for (auto &&i : properties) {
        if (i.first.first == instance_id) {
            THROW exception::EnvDescriptionAlreadyLoaded("Environment description with same instance id '%u' is already loaded, "
                "in EnvironmentManager::init().",
                instance_id);
        }
    }

    // Build a DefragMap of to send to defragger method
    DefragMap orderedProperties;
    size_t newSize = 0;
    std::set<NamePair> new_mapped_props;
    for (auto _i = desc.properties.begin(); _i != desc.properties.end(); ++_i) {
        auto prop_mapping = mapping.properties.find(_i->first);
        const auto &i = _i->second;
        NamePair name = toName(instance_id, _i->first);
        if (prop_mapping == mapping.properties.end()) {
            // Property is not mapped, so add to defrag map
            DefragProp prop = DefragProp(i.data.ptr, i.data.length, i.isConst, i.data.elements, i.data.type);
            const size_t typeSize = i.data.length / i.data.elements;
            orderedProperties.emplace(std::make_pair(typeSize, name), prop);
            newSize += i.data.length;
        } else {
            // Property is mapped, follow it's mapping upwards until we find the highest parent
            NamePair ultimateParent = toName(master_instance_id, prop_mapping->second);
            auto propFind = mapped_properties.find(ultimateParent);
            while (propFind != mapped_properties.end()) {
                ultimateParent = propFind->second.masterProp;
                propFind = mapped_properties.find(ultimateParent);
            }
            // Add to mapping list
            MappedProp mp = MappedProp(ultimateParent, i.isConst);
            mapped_properties.emplace(name, mp);
            new_mapped_props.emplace(name);
        }
    }
    if (newSize > m_freeSpace) {
        // Ran out of constant cache space! (this can only trigger when a DefragMap is passed)
        // Arguably this check should be performed by init()
        THROW exception::OutOfMemory("Insufficient EnvProperty memory to create new properties,"
            "in EnvironmentManager::init().");
    }
    // Defragment to rebuild it properly
    defragment(Curve::getInstance(), &orderedProperties, new_mapped_props, isPureRTC);
    // Setup RTC version
    buildRTCOffsets(instance_id, master_instance_id, orderedProperties);
}

void EnvironmentManager::initRTC(const CUDASimulation& cudaSimulation) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex);
    // check to ensure that model name is not already registered
    auto res = cuda_agent_models.find(cudaSimulation.getInstanceID());
    if (res != cuda_agent_models.end()) {
        THROW exception::UnknownInternalError("Agent model name '%s' already registered in initRTC()", cudaSimulation.getModelDescription().name.c_str());
    }
    // register model name
    cuda_agent_models.emplace(cudaSimulation.getInstanceID(), cudaSimulation);
}

void EnvironmentManager::initialiseDevice() {
    // Caller must lock mutex
    if (!deviceInitialised) {
        void *t_c_buffer = nullptr;
        gpuErrchk(cudaGetSymbolAddress(&t_c_buffer, detail::c_envPropBuffer));
        c_buffer = reinterpret_cast<char*>(t_c_buffer);
        // printf("Env Prop Constant Cache Buffer: %p - %p\n", c_buffer, c_buffer + MAX_BUFFER_SIZE);
        assert(CURVE_NAMESPACE_HASH == DeviceEnvironment::CURVE_NAMESPACE_HASH());  // Host and Device namespace const's do not match
        deviceInitialised = true;
    }
}
void EnvironmentManager::free(Curve &curve, const unsigned int &instance_id) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex);
    // Release regular properties
    for (auto &&i = properties.begin(); i != properties.end();) {
        if (i->first.first == instance_id) {
            // Release from CURVE
            Curve::VariableHash cvh = toHash(i->first);
            curve.unregisterVariableByHash(cvh);
            // Drop from properties map
            i = properties.erase(i);
        } else {
            ++i;
        }
    }
    // Release mapped properties
    for (auto &&i = mapped_properties.begin(); i != mapped_properties.end();) {
        if (i->first.first == instance_id) {
            // Release from CURVE
            Curve::VariableHash cvh = toHash(i->first);
            curve.unregisterVariableByHash(cvh);
            // Drop from properties map
            i = mapped_properties.erase(i);
        } else {
            ++i;
        }
    }
    // Defragment to clear up all the buffer items we didn't handle here
    defragment(curve);
    // Remove reference to cuda agent model used by RTC
    // This may not exist if the CUDAgent model has not been created (e.g. some tests which do not run the model)
    auto cam = cuda_agent_models.find(instance_id);
    if (cam != cuda_agent_models.end()) {
        cuda_agent_models.erase(cam);
    }
    // Sample applies to requires update map
    std::unique_lock<std::shared_timed_mutex> deviceRequiresUpdate_lock(deviceRequiresUpdate_mutex);
    auto dru = deviceRequiresUpdate.find(instance_id);
    if (dru != deviceRequiresUpdate.end()) {
        deviceRequiresUpdate.erase(dru);
    }
    deviceRequiresUpdate_lock.unlock();

    // Sample applies to rtc_caches
    auto rtcc = rtc_caches.find(instance_id);
    if (rtcc != rtc_caches.end()) {
        rtc_caches.erase(rtcc);
    }
}

EnvironmentManager::NamePair EnvironmentManager::toName(const unsigned int &instance_id, const std::string &var_name) {
    return std::make_pair(instance_id, var_name);
}

/**
 * @note Not static, because eventually we might need to use curve singleton
 */
Curve::VariableHash EnvironmentManager::toHash(const NamePair &name) const {
    Curve::VariableHash var_cvh = Curve::variableRuntimeHash(name.second.c_str());
    return CURVE_NAMESPACE_HASH + name.first + var_cvh;
}

void EnvironmentManager::newProperty(const NamePair &name, const char *ptr, const size_t &length, const bool &isConst, const size_type &elements, const std::type_index &type) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex);
    assert(elements > 0);
    const size_t typeSize = (length / elements);
    ptrdiff_t buffOffset = MAX_BUFFER_SIZE;
    // Allocate buffer space, using a free fragment
    for (auto it = freeFragments.begin(); it != freeFragments.end(); ++it) {
        const ptrdiff_t alignmentOffset = std::get<OFFSET>(*it) % typeSize;
        const ptrdiff_t alignmentFix = alignmentOffset != 0 ? typeSize - alignmentOffset : 0;
        if (std::get<LEN>(*it) + alignmentFix <= length) {
            // We can use this space!
            buffOffset = std::get<OFFSET>(*it) + alignmentFix;
            // Update freeFrags
            if (alignmentFix != 0) {
                freeFragments.push_back(OffsetLen(std::get<OFFSET>(*it), alignmentFix));
            }
            // Update nextFree
            if (std::get<LEN>(*it) == length) {
                // Remove
                freeFragments.erase(it);
            } else {
                // Shrink
                *it = { std::get<OFFSET>(*it) + length, std::get<LEN>(*it) - length };
            }
            break;
        }
    }
    // Allocate buffer space, using nextFree
    if (buffOffset == MAX_BUFFER_SIZE) {
        const ptrdiff_t alignmentOffset = nextFree % typeSize;
        const ptrdiff_t alignmentFix = alignmentOffset != 0 ? typeSize - alignmentOffset : 0;
        if (nextFree + alignmentFix + length < MAX_BUFFER_SIZE) {
            // Update freeFrags
            if (alignmentFix != 0) {
                freeFragments.push_back(OffsetLen(nextFree, alignmentFix));
            }
            // We can use this space!
            nextFree += alignmentFix;
            buffOffset = nextFree;
            nextFree += length;
        }
    }
    if (buffOffset == MAX_BUFFER_SIZE) {  // buffOffset hasn't changed from init value
        // defragment() and retry using nextFree
        defragment(Curve::getInstance());
        const ptrdiff_t alignmentOffset = nextFree % typeSize;
        const ptrdiff_t alignmentFix = alignmentOffset != 0 ? typeSize - alignmentOffset : 0;
        if (nextFree + alignmentFix + length < MAX_BUFFER_SIZE) {
            // Update freeFrags
            if (alignmentFix != 0) {
                freeFragments.push_back(OffsetLen(nextFree, alignmentFix));
            }
            // We can use this space!
            nextFree += alignmentFix;
            buffOffset = nextFree;
            nextFree += length;
        } else {
            // Ran out of constant cache space!
            THROW exception::OutOfMemory("Insufficient EnvProperty memory to create new property,"
                "in EnvironmentManager::add().");
        }
    }
    // Update FreeSpace
    m_freeSpace -= length;
    // Add to properties
    // printf("Constant '%s' created at offset: %llu, (%llu%%8), (%llu%%4)\n", name.c_str(), buffOffset, buffOffset % 8, buffOffset % 4);
    properties.emplace(name, EnvProp(buffOffset, length, isConst, elements, type));
    // Store data
    memcpy(hc_buffer + buffOffset, ptr, length);
    // Register in cuRVE
    Curve::VariableHash cvh = toHash(name);
    const auto CURVE_RESULT = Curve::getInstance().registerVariableByHash(cvh, reinterpret_cast<void*>(buffOffset), typeSize, elements);
    if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
        THROW exception::CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE"
            "in EnvironmentManager::add().");
    }
#ifdef _DEBUG
    if (CURVE_RESULT != static_cast<int>(cvh%Curve::MAX_VARIABLES)) {
        fprintf(stderr, "Curve Warning: Environment Property '%s' has a collision and may work improperly.\n", name.second.c_str());
    }
#endif
    addRTCOffset(name);
    setDeviceRequiresUpdateFlag();
}

#ifdef _DEBUG
void EnvironmentManager::defragment(Curve &curve, const DefragMap * mergeProperties, std::set<NamePair> newmaps, bool isPureRTC) {
#else
void EnvironmentManager::defragment(Curve & curve, const DefragMap * mergeProperties, std::set<NamePair> newmaps, bool) {
#endif
    // Do not lock mutex here, do it in the calling method
    auto device_lock = std::unique_lock<std::shared_timed_mutex>(device_mutex);
    // Build a multimap to sort the elements (to create natural alignment in compact form)
    DefragMap orderedProperties;
    for (auto &i : properties) {
        size_t typeLen = i.second.length / i.second.elements;
        orderedProperties.emplace(std::make_pair(typeLen, i.first), DefragProp(i.second));
    }
    // Include any merge elements
    if (mergeProperties) {
        orderedProperties.insert(mergeProperties->cbegin(), mergeProperties->cend());
    }
    // Lock device mutex here, as we begin to mess with curve
    // Clear freefrags, so we can refill it with alignment junk
    freeFragments.clear();
    size_t spareFrags = 0;
    // Rebuild properties map into temporary buffer
    std::unordered_map<NamePair, EnvProp, NamePairHash> t_properties;
    char t_buffer[MAX_BUFFER_SIZE];
    ptrdiff_t buffOffset = 0;
    // Iterate largest vars first
    for (auto _i = orderedProperties.rbegin(); _i != orderedProperties.rend(); ++_i) {
        size_t typeSize = _i->first.first;
        const NamePair &name = _i->first.second;
        auto &i = _i->second;
        // Handle alignment
        const ptrdiff_t alignmentOffset = buffOffset % typeSize;
        const ptrdiff_t alignmentFix = alignmentOffset != 0 ? typeSize - alignmentOffset : 0;
        if (alignmentOffset != 0) {
            freeFragments.push_back(OffsetLen(buffOffset, alignmentFix));
            buffOffset += alignmentFix;
            spareFrags += alignmentFix;
        }
        if (buffOffset + i.length <= MAX_BUFFER_SIZE) {
            // Setup constant in new position
            memcpy(t_buffer + buffOffset, i.data, i.length);
            t_properties.emplace(name, EnvProp(buffOffset, i.length, i.isConst, i.elements, i.type, i.rtc_offset));
            // Update cuRVE (There isn't an update, so unregister and reregister)  // TODO: curveGetVariableHandle()?
            Curve::VariableHash cvh = toHash(name);
            // Only unregister variable if it's already registered
            if (!mergeProperties) {  // Merge properties are only provided on 1st init, when vars can't be unregistered
                curve.unregisterVariableByHash(cvh);
            } else {
                // Can this var be found inside mergeProps
                auto range = mergeProperties->equal_range(_i->first);
                bool isFound = false;
                for (auto w = range.first; w != range.second; ++w) {
                    if (w->first.second == name) {
                        isFound = true;
                        break;
                    }
                }
                if (!isFound) {
                    curve.unregisterVariableByHash(cvh);
                }
            }
            const auto CURVE_RESULT = curve.registerVariableByHash(cvh, reinterpret_cast<void*>(buffOffset),
                typeSize, i.elements);
            if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
                THROW exception::CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE, "
                    "in EnvironmentManager::defragment().");
            }
#ifdef _DEBUG
            if (!isPureRTC && CURVE_RESULT != static_cast<int>(cvh%Curve::MAX_VARIABLES)) {
                fprintf(stderr, "Curve Warning: Environment Property '%s' has a collision and may work improperly.\n", name.second.c_str());
            }
#endif
            // Increase buffer offset length that has been added
            buffOffset += i.length;
        } else {
            // Ran out of constant cache space! (this can only trigger when a DefragMap is passed)
            // Arguably this check should be performed by init()
            THROW exception::OutOfMemory("Insufficient EnvProperty memory to create new properties, "
                "in EnvironmentManager::defragment().");
        }
    }
    // Replace stored properties with temp
    std::swap(properties, t_properties);
    // Replace buffer on host
    memcpy(hc_buffer, t_buffer, buffOffset);
    // Update m_freeSpace, nextFree
    nextFree = buffOffset;
    m_freeSpace = MAX_BUFFER_SIZE - buffOffset + spareFrags;
    // Update cub for any mapped properties
    for (auto &mp : mapped_properties) {
        // Generate hash for the subproperty name
        Curve::VariableHash cvh = toHash(mp.first);
        // Unregister the property if it's already been registered
        if (newmaps.find(mp.first) == newmaps.end()) {
            curve.unregisterVariableByHash(cvh);
        }
        // Find the location of the mappedproperty
        auto masterprop = properties.at(mp.second.masterProp);
        // Create the mapping
        const auto CURVE_RESULT = curve.registerVariableByHash(cvh, reinterpret_cast<void*>(masterprop.offset),
                masterprop.length / masterprop.elements, masterprop.elements);
        if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
            THROW exception::CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE, "
                "in EnvironmentManager::defragment().");
        }
#ifdef _DEBUG
        if (!isPureRTC && CURVE_RESULT != static_cast<int>(cvh%Curve::MAX_VARIABLES)) {
            fprintf(stderr, "Curve Warning: Environment Property '%s' has a collision and may work improperly.\n", mp.first.second.c_str());
        }
#endif
    }
    setDeviceRequiresUpdateFlag();
}

void EnvironmentManager::buildRTCOffsets(const unsigned int &instance_id, const unsigned int &master_instance_id, const DefragMap &orderedProperties) {
    // Do not lock mutex here, do it in the calling method
    // Actually begin
    if (instance_id == master_instance_id) {
        // Create a new cache
        std::shared_ptr<RTCEnvPropCache> cache = std::make_shared<RTCEnvPropCache>();
        // Add the properties, they are already ordered so we can just enforce alignment
        // As we add each property, set its rtc_offset value in main properties map
        for (auto _i = orderedProperties.rbegin(); _i != orderedProperties.rend(); ++_i) {
            auto &i = _i->second;
            size_t alignmentSize = _i->first.first;
            const NamePair &name = _i->first.second;
            // Handle alignment
            const ptrdiff_t alignmentOffset = cache->nextFree % alignmentSize;
            const ptrdiff_t alignmentFix = alignmentOffset != 0 ? alignmentSize - alignmentOffset : 0;
            cache->nextFree += alignmentFix;
            if (cache->nextFree + i.length <= MAX_BUFFER_SIZE) {
                // Setup constant in new position
                memcpy(cache->hc_buffer + cache->nextFree, i.data, i.length);
                properties.at(name).rtc_offset = cache->nextFree;
                // Increase buffer offset length that has been added
                cache->nextFree += i.length;
            } else {
                // Ran out of constant cache space! (this can only trigger when a DefragMap is passed)
                // Arguably this check should be performed by init()
                // This should never happen, it would be caught by defrag sooner
                THROW exception::OutOfMemory("Insufficient EnvProperty memory to create new properties, "
                    "in EnvironmentManager::buildRTCOffsets().");
            }
        }
        // Cache is complete, add it to cache map
        rtc_caches.emplace(instance_id, cache);
    } else {
        // Find the master cache
        std::shared_ptr<RTCEnvPropCache> &cache = rtc_caches.at(master_instance_id);
        // Add the properties, they are already ordered so we can just enforce alignment
        // As we add each property, set its rtc_offset value in main properties map
        for (auto _i = orderedProperties.rbegin(); _i != orderedProperties.rend(); ++_i) {
            const NamePair &name = _i->first.second;
            auto &i = _i->second;
            auto mi_it = mapped_properties.find(name);
            if (mi_it ==  mapped_properties.end()) {
                // Property is not mapped, add it to cache
                size_t alignmentSize = _i->first.first;
                // Handle alignment
                const ptrdiff_t alignmentOffset = cache->nextFree % alignmentSize;
                const ptrdiff_t alignmentFix = alignmentOffset != 0 ? alignmentSize - alignmentOffset : 0;
                cache->nextFree += alignmentFix;
                if (cache->nextFree + i.length <= MAX_BUFFER_SIZE) {
                    // Setup constant in new position
                    memcpy(cache->hc_buffer + cache->nextFree, i.data, i.length);
                    properties.at(name).rtc_offset = cache->nextFree;
                    // Increase buffer offset length that has been added
                    cache->nextFree += i.length;
                } else {
                    // Ran out of constant cache space! (this can only trigger when a DefragMap is passed)
                    // Arguably this check should be performed by init()
                    // This should never happen, it would be caught by defrag sooner
                    THROW exception::OutOfMemory("Insufficient EnvProperty memory to create new properties, "
                        "in EnvironmentManager::buildRTCOffsets().");
                }
            }
        }
        // Add a copy of cache for this instance_id to env manager
        rtc_caches.emplace(instance_id, cache);
    }
}
char * EnvironmentManager::getRTCCache(const unsigned int& instance_id) {
    auto it = rtc_caches.find(instance_id);
    if (it != rtc_caches.end())
        return it->second->hc_buffer;
    THROW exception::UnknownInternalError("Instance with id '%u' not registered in EnvironmentManager for use with RTC in EnvironmentManager::getRTCCache", instance_id);
}
void EnvironmentManager::addRTCOffset(const NamePair &name) {
    // Do not lock mutex here, do it in the calling method
    auto mi_it = mapped_properties.find(name);
    // Property is not mapped (it's not even currently possible to add mapped properties after the fact)
    if (mi_it ==  mapped_properties.end()) {
        auto &cache = rtc_caches.at(name.first);
        auto &p = properties.at(name);
        size_t alignmentSize = p.length > 64 ? 64 : p.length;  // This creates better alignment for small vectors
        // Handle alignment
        const ptrdiff_t alignmentOffset = cache->nextFree % alignmentSize;
        const ptrdiff_t alignmentFix = alignmentOffset != 0 ? alignmentSize - alignmentOffset : 0;
        cache->nextFree += alignmentFix;
        if (cache->nextFree + p.length <= MAX_BUFFER_SIZE) {
            // Setup constant in new position
            memcpy(cache->hc_buffer + cache->nextFree, hc_buffer + p.offset, p.length);
            p.rtc_offset = cache->nextFree;
            // Increase buffer offset length that has been added
            cache->nextFree += p.length;
        } else {
            THROW exception::OutOfMemory("Insufficient EnvProperty memory to create new properties, "
                "in EnvironmentManager::buildRTCOffsets().");
        }
    } else {
        THROW exception::OutOfMemory("Support for mapped (sub) properties is not currently implemented, "
            "in EnvironmentManager::addRTCOffset().");
    }
}

const CUDASimulation& EnvironmentManager::getCUDASimulation(const unsigned int &instance_id) {
    // Don't lock mutex here, lock it in the calling function
    auto res = cuda_agent_models.find(instance_id);
    if (res == cuda_agent_models.end()) {
        THROW exception::UnknownInternalError("Instance with id '%u' not registered in EnvironmentManager for use with RTC in EnvironmentManager::getCUDASimulation", instance_id);
    }
    return res->second;
}

void EnvironmentManager::updateRTCValue(const NamePair &name) {
    // Don't lock mutex here, lock it in the calling function
    // Grab the updated prop
    auto a = properties.find(name);
    if (a == properties.end()) {
        a = properties.find(mapped_properties.at(name).masterProp);
    }
    // Grab the main cache ptr for the prop
    void *main_ptr = hc_buffer + a->second.offset;
    // Grab the rtc cache ptr for the prop
    void *rtc_ptr = rtc_caches.at(name.first)->hc_buffer + a->second.rtc_offset;
    // Copy
    memcpy(rtc_ptr, main_ptr, a->second.length);

    // Now we must detect if the variable is mapped in any form
    // If this is the case, any rtc models which share the property must be flagged for update too
    {
        std::unique_lock<std::shared_timed_mutex> deviceRequiresUpdate_lock(deviceRequiresUpdate_mutex);
        // First check if it's the subvar
        auto mp_it = mapped_properties.find(name);
        const NamePair check = mp_it == mapped_properties.end() ? name : mp_it->second.masterProp;
        // Now check for any properties mapped to this variable
        for (auto mp : mapped_properties) {
            if (mp.second.masterProp == check) {
                // It's a hit, set flag to true
                deviceRequiresUpdate.at(check.first).rtc_update_required = true;
            }
        }
        deviceRequiresUpdate_lock.unlock();
    }
}

void EnvironmentManager::removeProperty(const NamePair &name) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex);
    // Unregister in cuRVE
    Curve::VariableHash cvh = toHash(name);
    Curve::getInstance().unregisterVariableByHash(cvh);
    // Update free space
    // Remove from properties map
    auto realprop = properties.find(name);
    if (realprop!= properties.end()) {
        auto i = realprop->second;
        // Cast is safe, length would need to be gigabytes, we only have 64KB constant cache
        if (i.offset + static_cast<uint32_t>(i.length) == nextFree) {
            // Rollback nextFree
            nextFree = i.offset;
        } else {
            // Notify free fragments
            freeFragments.push_back(OffsetLen(i.offset, i.length));
        }
        m_freeSpace += i.length;
        // Purge properties
        properties.erase(name);
    } else {
        mapped_properties.erase(name);
    }
    setDeviceRequiresUpdateFlag(name.first);
}
void EnvironmentManager::removeProperty(const unsigned int &instance_id, const std::string &var_name) {
    removeProperty({instance_id, var_name});
}

void EnvironmentManager::resetModel(const unsigned int &instance_id, const EnvironmentDescription &desc) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex);
    // Todo: Might want to change this, so EnvManager holds a copy of the default at init time
    // For every property, in the named model, which is not a mapped property
    for (auto &d : desc.getPropertiesMap()) {
        if (mapped_properties.find({instance_id, d.first}) == mapped_properties.end()) {
            // Find the local property data
            auto &p = properties.at({instance_id, d.first});
            // Set back to default value
            memcpy(hc_buffer + p.offset, d.second.data.ptr, d.second.data.length);
            // Do rtc too
            void *rtc_ptr = rtc_caches.at(instance_id)->hc_buffer + p.rtc_offset;
            memcpy(rtc_ptr, d.second.data.ptr, d.second.data.length);
            assert(d.second.data.length == p.length);
        }
    }
    setDeviceRequiresUpdateFlag(instance_id);
}
void EnvironmentManager::setDeviceRequiresUpdateFlag(const unsigned int &instance_id) {
    std::unique_lock<std::shared_timed_mutex> deviceRequiresUpdate_lock(deviceRequiresUpdate_mutex);
    // Don't lock mutex here, lock it in the calling function
    // Increment host version
    if (instance_id == UINT_MAX) {
        // Set required version for all, we have defragged
        for (auto &a : deviceRequiresUpdate) {
            a.second.c_update_required = true;
            a.second.rtc_update_required = true;
        }
    } else {
        // Set individual
        auto &flags = deviceRequiresUpdate.at(instance_id);
        flags.c_update_required = true;
        flags.rtc_update_required = true;
    }
}
void EnvironmentManager::updateDevice(const unsigned int &instance_id) {
    // Lock shared mutex of mutex in calling method first!!!
    // Device must be init first
    assert(deviceInitialised);
    std::unique_lock<std::shared_timed_mutex> deviceRequiresUpdate_lock(deviceRequiresUpdate_mutex);
    NVTX_RANGE("EnvironmentManager::updateDevice()");
    auto &flags = deviceRequiresUpdate.at(instance_id);
    auto &c_update_required = flags.c_update_required;
    auto &rtc_update_required = flags.rtc_update_required;
    auto &curve_registration_required = flags.curve_registration_required;
    if (c_update_required) {
        // Store data
        gpuErrchk(cudaMemcpy(reinterpret_cast<void*>(const_cast<char*>(c_buffer)), reinterpret_cast<void*>(const_cast<char*>(hc_buffer)), MAX_BUFFER_SIZE, cudaMemcpyHostToDevice));
        // Update C update flag for all instances
        for (auto &a : deviceRequiresUpdate) {
            a.second.c_update_required = false;
        }
    }
    if (rtc_update_required) {
        // RTC is nolonger updated here, it's always updated before the CurveRTCHost is pushed to device.
        // Update instance's rtc update flag
        rtc_update_required = false;
    }
    if (curve_registration_required) {
        auto &curve = Curve::getInstance();
        // Update cub for any not mapped properties
        for (auto &p : properties) {
            if (p.first.first == instance_id) {
                // Generate hash for the subproperty name
                Curve::VariableHash cvh = toHash(p.first);
                const auto &prop = p.second;
                // Create the mapping
                const auto CURVE_RESULT = curve.registerVariableByHash(cvh, reinterpret_cast<void*>(prop.offset),
                        prop.length / prop.elements, prop.elements);
                if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
                    THROW exception::CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE, "
                        "in EnvironmentManager::updateDevice().");
                }
#ifdef _DEBUG
                if (CURVE_RESULT != static_cast<int>(cvh%Curve::MAX_VARIABLES)) {
                    fprintf(stderr, "Curve Warning: Environment Property '%s' has a collision and may work improperly.\n", p.first.second.c_str());
                }
#endif
            }
        }
        // Update cub for any mapped properties
        for (auto &mp : mapped_properties) {
            if (mp.first.first == instance_id) {
                // Generate hash for the subproperty name
                Curve::VariableHash cvh = toHash(mp.first);
                const auto &masterprop = properties.at(mp.second.masterProp);
                // Create the mapping
                const auto CURVE_RESULT = curve.registerVariableByHash(cvh, reinterpret_cast<void*>(masterprop.offset),
                        masterprop.length / masterprop.elements, masterprop.elements);
                if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
                    THROW exception::CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE, "
                        "in EnvironmentManager::updateDevice().");
                }
#ifdef _DEBUG
                if (CURVE_RESULT != static_cast<int>(cvh%Curve::MAX_VARIABLES)) {
                    fprintf(stderr, "Curve Warning: Environment Property '%s' has a collision and may work improperly.\n", mp.first.second.c_str());
                }
#endif
            }
        }
        curve_registration_required = false;
    }
}


EnvironmentManager& EnvironmentManager::getInstance() {
    auto lock = std::unique_lock<std::mutex>(instance_mutex);  // Mutex to protect from two threads triggering the static instantiation concurrently
    static std::map<int, std::unique_ptr<EnvironmentManager>> instances = {};  // Instantiated on first use.
    int device_id = -1;
    gpuErrchk(cudaGetDevice(&device_id));
    // Can't use operator[] here, constructor is private
    const auto f = instances.find(device_id);
    if (f != instances.end())
        return *f->second;
    return *(instances.emplace(device_id, std::unique_ptr<EnvironmentManager>(new EnvironmentManager())).first->second);
}


util::Any EnvironmentManager::getPropertyAny(const unsigned int &instance_id, const std::string &var_name) const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex);
    const NamePair name = toName(instance_id, var_name);
    auto a = properties.find(name);
    if (a != properties.end())
        return util::Any(hc_buffer + a->second.offset, a->second.length, a->second.type, a->second.elements);
    const auto b = mapped_properties.find(name);
    if (b != mapped_properties.end()) {
        a = properties.find(b->second.masterProp);
        if (a != properties.end())
            return util::Any(hc_buffer + a->second.offset, a->second.length, a->second.type, a->second.elements);
        THROW exception::InvalidEnvProperty("Mapped environmental property with name '%u:%s' maps to missing property with name '%u:%s', "
            "in EnvironmentManager::getPropertyAny().",
            name.first, name.second.c_str(), b->second.masterProp.first, b->second.masterProp.second.c_str());
    }
    THROW exception::InvalidEnvProperty("Environmental property with name '%u:%s' does not exist, "
        "in EnvironmentManager::getPropertyAny().",
        name.first, name.second.c_str());
}

}  // namespace flamegpu
