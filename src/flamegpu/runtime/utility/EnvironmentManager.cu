#include "flamegpu/runtime/utility/EnvironmentManager.cuh"

#include <cassert>
#include <map>

#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/runtime/utility/DeviceEnvironment.cuh"
#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/model/SubEnvironmentData.h"
#include "flamegpu/gpu/CUDAAgentModel.h"

/**
 * Internal namespace to hide __constant__ declarations from modeller
 */
namespace flamegpu_internal {
    /**
     * Managed by HostEnvironment, holds all environment properties
     */
    __constant__ char c_envPropBuffer[EnvironmentManager::MAX_BUFFER_SIZE];
    /**
     * Managed by HostEnvironment, returned whenever a failure state is reached
     * Think this exists because we cant return a reference to a constexpr
     */
    __constant__ uint64_t c_deviceEnvErrorPattern;
}  // namespace flamegpu_internal

const char EnvironmentManager::CURVE_NAMESPACE_STRING[23] = "ENVIRONMENT_PROPERTIES";

EnvironmentManager::EnvironmentManager() :
    CURVE_NAMESPACE_HASH(Curve::variableRuntimeHash(CURVE_NAMESPACE_STRING)),
    nextFree(0),
    m_freeSpace(EnvironmentManager::MAX_BUFFER_SIZE),
    freeFragments(),
    deviceInitialised(false) { }

void EnvironmentManager::purge() {
    deviceInitialised = false;
    for (auto &a : deviceRequiresUpdate) {
        a.second.c_update_required = true;
        a.second.rtc_update_required = true;
        a.second.curve_registration_required = true;
    }
    // We are now able to only purge the device stuff after device reset?
    // freeFragments.clear();
    // m_freeSpace = EnvironmentManager::MAX_BUFFER_SIZE;
    // nextFree = 0;
    // cuda_agent_models.clear();
    // properties.clear();
    // mapped_properties.clear();
    // rtc_caches.clear();
}

void EnvironmentManager::init(const unsigned int &instance_id, const EnvironmentDescription &desc) {
    // Error if reinit
    for (auto &&i : properties) {
        if (i.first.first == instance_id) {
            THROW EnvDescriptionAlreadyLoaded("Environment description with same instance id '%u' is already loaded, "
                "in EnvironmentManager::init().",
                instance_id);
        }
    }

    // Add to device requires update map
    deviceRequiresUpdate.emplace(instance_id, EnvUpdateFlags());

    // Build a DefragMap to send to defragger method
    DefragMap orderedProperties;
    size_t newSize = 0;
    for (auto _i = desc.properties.begin(); _i != desc.properties.end(); ++_i) {
        const auto &i = _i->second;
        NamePair name = toName(instance_id, _i->first);
        DefragProp prop = DefragProp(i.data.ptr, i.data.length, i.isConst, i.elements, i.type);
        const size_t typeSize = i.data.length / i.elements;
        orderedProperties.emplace(typeSize, std::make_pair(name, prop));
        newSize += i.data.length;
    }
    if (newSize > m_freeSpace) {
        // Ran out of constant cache space! (this can only trigger when a DefragMap is passed)
        // Arguably this check should be performed by init()
        THROW OutOfMemory("Insufficient EnvProperty memory to create new properties,"
            "in EnvironmentManager::init().");
    }
    // Defragment to rebuild it properly
    defragment(&orderedProperties);
    // Setup RTC version
    buildRTCOffsets(instance_id, instance_id, orderedProperties);
}
void EnvironmentManager::init(const unsigned int &instance_id, const EnvironmentDescription &desc, const unsigned int &master_instance_id, const SubEnvironmentData &mapping) {
    assert(deviceRequiresUpdate.size());  // submodel init should never be called first, requires parent init first for mapping

    // Add to device requires update map
    deviceRequiresUpdate.emplace(instance_id, EnvUpdateFlags());

    // Error if reinit
    for (auto &&i : properties) {
        if (i.first.first == instance_id) {
            THROW EnvDescriptionAlreadyLoaded("Environment description with same instance id '%u' is already loaded, "
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
            DefragProp prop = DefragProp(i.data.ptr, i.data.length, i.isConst, i.elements, i.type);
            const size_t typeSize = i.data.length / i.elements;
            orderedProperties.emplace(typeSize, std::make_pair(name, prop));
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
        THROW OutOfMemory("Insufficient EnvProperty memory to create new properties,"
            "in EnvironmentManager::init().");
    }
    // Defragment to rebuild it properly
    defragment(&orderedProperties, new_mapped_props);
    // Setup RTC version
    buildRTCOffsets(instance_id, master_instance_id, orderedProperties);
}

void EnvironmentManager::initRTC(const CUDAAgentModel& cuda_model) {
    // check to ensure that model name is not already registered
    auto res = cuda_agent_models.find(cuda_model.getInstanceID());
    if (res != cuda_agent_models.end()) {
        THROW UnknownInternalError("Agent model name '%s' already registered in initRTC()", cuda_model.getModelDescription().name.c_str());
    }
    // register model name
    cuda_agent_models.emplace(cuda_model.getInstanceID(), cuda_model);
}

void EnvironmentManager::initialiseDevice() {
    if (!deviceInitialised) {
        void *t_c_buffer = nullptr;
        gpuErrchk(cudaGetSymbolAddress(&t_c_buffer, flamegpu_internal::c_envPropBuffer));
        c_buffer = reinterpret_cast<char*>(t_c_buffer);
        // printf("Env Prop Constant Cache Buffer: %p - %p\n", c_buffer, c_buffer + MAX_BUFFER_SIZE);
        assert(CURVE_NAMESPACE_HASH == DeviceEnvironment::CURVE_NAMESPACE_HASH());  // Host and Device namespace const's do not match
        // Setup device-side error pattern
        const uint64_t h_errorPattern = DeviceEnvironment::ERROR_PATTERN();
        gpuErrchk(cudaMemcpyToSymbol(flamegpu_internal::c_deviceEnvErrorPattern, reinterpret_cast<const void*>(&h_errorPattern), sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
        deviceInitialised = true;
    }
}
void EnvironmentManager::free(const unsigned int &instance_id) {
    Curve::getInstance().setNamespaceByHash(CURVE_NAMESPACE_HASH);
    // Release regular properties
    for (auto &&i = properties.begin(); i != properties.end();) {
        if (i->first.first == instance_id) {
            // Release from CURVE
            Curve::VariableHash cvh = toHash(i->first);
            Curve::getInstance().unregisterVariableByHash(cvh);
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
            Curve::getInstance().unregisterVariableByHash(cvh);
            // Drop from properties map
            i = mapped_properties.erase(i);
        } else {
            ++i;
        }
    }
    Curve::getInstance().setDefaultNamespace();
    // Defragment to clear up all the buffer items we didn't handle here
    defragment();
    // Remove reference to cuda agent model used by RTC
    // This may not exist if the CUDAgent model has not been created (e.g. some tests which do not run the model)
    auto cam = cuda_agent_models.find(instance_id);
    if (cam != cuda_agent_models.end()) {
        cuda_agent_models.erase(cam);
    }
    // Sample applies to requires update map
    auto dru = deviceRequiresUpdate.find(instance_id);
    if (dru != deviceRequiresUpdate.end()) {
        deviceRequiresUpdate.erase(dru);
    }
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
    Curve::VariableHash model_cvh = Curve::getInstance().variableRuntimeHash(name.first);
    Curve::VariableHash var_cvh = Curve::getInstance().variableRuntimeHash(name.second.c_str());
    return model_cvh + var_cvh;
}

void EnvironmentManager::add(const NamePair &name, const char *ptr, const size_t &length, const bool &isConst, const size_type &elements, const std::type_index &type) {
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
        defragment();
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
            THROW OutOfMemory("Insufficient EnvProperty memory to create new property,"
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
    Curve::getInstance().setNamespaceByHash(CURVE_NAMESPACE_HASH);
    Curve::VariableHash cvh = toHash(name);
    const auto CURVE_RESULT = Curve::getInstance().registerVariableByHash(cvh, reinterpret_cast<void*>(buffOffset), typeSize, elements);
    if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
        THROW CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE"
            "in EnvironmentManager::add().");
    }
#ifdef _DEBUG
    if (CURVE_RESULT != static_cast<int>((cvh+CURVE_NAMESPACE_HASH)%Curve::MAX_VARIABLES)) {
        fprintf(stderr, "Curve Warning: Environment Property '%s' has a collision and may work improperly.\n", name.second.c_str());
    }
#endif
    Curve::getInstance().setDefaultNamespace();
    addRTCOffset(name);
    setDeviceRequiresUpdateFlag();
}

void EnvironmentManager::defragment(const DefragMap * mergeProperties, std::set<NamePair> newmaps) {
    // Build a multimap to sort the elements (to create natural alignment in compact form)
    DefragMap orderedProperties;
    for (auto &i : properties) {
        size_t typeLen = i.second.length / i.second.elements;
        orderedProperties.emplace(typeLen, std::make_pair(i.first, DefragProp(i.second)));
    }
    // Include any merge elements
    if (mergeProperties) {
        orderedProperties.insert(mergeProperties->cbegin(), mergeProperties->cend());
    }
    // Clear freefrags, so we can refill it with alignment junk
    freeFragments.clear();
    size_t spareFrags = 0;
    // Rebuild properties map into temporary buffer
    std::unordered_map<NamePair, EnvProp, NamePairHash> t_properties;
    char t_buffer[MAX_BUFFER_SIZE];
    ptrdiff_t buffOffset = 0;
    Curve::getInstance().setNamespaceByHash(CURVE_NAMESPACE_HASH);
    // Iterate largest vars first
    for (auto _i = orderedProperties.rbegin(); _i != orderedProperties.rend(); ++_i) {
        size_t typeSize = _i->first;
        auto &i = _i->second;
        // Handle alignment
        const ptrdiff_t alignmentOffset = buffOffset % typeSize;
        const ptrdiff_t alignmentFix = alignmentOffset != 0 ? typeSize - alignmentOffset : 0;
        if (alignmentOffset != 0) {
            freeFragments.push_back(OffsetLen(buffOffset, alignmentFix));
            buffOffset += alignmentFix;
            spareFrags += alignmentFix;
        }
        if (buffOffset + i.second.length <= MAX_BUFFER_SIZE) {
            // Setup constant in new position
            memcpy(t_buffer + buffOffset, i.second.data, i.second.length);
            t_properties.emplace(i.first, EnvProp(buffOffset, i.second.length, i.second.isConst, i.second.elements, i.second.type));
            // Update cuRVE (There isn't an update, so unregister and reregister)  // TODO: curveGetVariableHandle()?
            Curve::VariableHash cvh = toHash(i.first);
            // Only unregister variable if it's already registered
            if (!mergeProperties) {  // Merge properties are only provided on 1st init, when vars can't be unregistered
                Curve::getInstance().unregisterVariableByHash(cvh);
            } else {
                // Can this var be found inside mergeProps
                auto range = mergeProperties->equal_range(_i->first);
                bool isFound = false;
                for (auto w = range.first; w != range.second; ++w) {
                    if (w->second.first == _i->second.first) {
                        isFound = true;
                        break;
                    }
                }
                if (!isFound) {
                    Curve::getInstance().unregisterVariableByHash(cvh);
                }
            }
            const auto CURVE_RESULT = Curve::getInstance().registerVariableByHash(cvh, reinterpret_cast<void*>(buffOffset),
                typeSize, i.second.elements);
            if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
                THROW CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE, "
                    "in EnvironmentManager::defragment().");
            }
#ifdef _DEBUG
            if (CURVE_RESULT != static_cast<int>((cvh+CURVE_NAMESPACE_HASH)%Curve::MAX_VARIABLES)) {
                fprintf(stderr, "Curve Warning: Environment Property '%s' has a collision and may work improperly.\n", i.first.second.c_str());
            }
#endif
            // Increase buffer offset length that has been added
            buffOffset += i.second.length;
        } else {
            // Ran out of constant cache space! (this can only trigger when a DefragMap is passed)
            // Arguably this check should be performed by init()
            THROW OutOfMemory("Insufficient EnvProperty memory to create new properties, "
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
            Curve::getInstance().unregisterVariableByHash(cvh);
        }
        // Find the location of the mappedproperty
        auto masterprop = properties.at(mp.second.masterProp);
        // Create the mapping
        const auto CURVE_RESULT = Curve::getInstance().registerVariableByHash(cvh, reinterpret_cast<void*>(masterprop.offset),
                masterprop.length / masterprop.elements, masterprop.elements);
        if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
            THROW CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE, "
                "in EnvironmentManager::defragment().");
        }
#ifdef _DEBUG
        if (CURVE_RESULT != static_cast<int>((cvh+CURVE_NAMESPACE_HASH)%Curve::MAX_VARIABLES)) {
            fprintf(stderr, "Curve Warning: Environment Property '%s' has a collision and may work improperly.\n", mp.first.second.c_str());
        }
#endif
    }
    Curve::getInstance().setDefaultNamespace();
    setDeviceRequiresUpdateFlag();
}

void EnvironmentManager::buildRTCOffsets(const unsigned int &instance_id, const unsigned int &master_instance_id, const DefragMap &orderedProperties) {
    // Actually begin
    if (instance_id == master_instance_id) {
        // Create a new cache
        std::shared_ptr<RTCEnvPropCache> cache = std::make_shared<RTCEnvPropCache>();
        // Add the properties, they are already ordered so we can just enforce alignment
        // As we add each property, set its rtc_offset value in main properties map
        for (auto _i = orderedProperties.rbegin(); _i != orderedProperties.rend(); ++_i) {
            auto &i = _i->second;
            size_t alignmentSize = _i->first;
            // Handle alignment
            const ptrdiff_t alignmentOffset = cache->nextFree % alignmentSize;
            const ptrdiff_t alignmentFix = alignmentOffset != 0 ? alignmentSize - alignmentOffset : 0;
            cache->nextFree += alignmentFix;
            if (cache->nextFree + i.second.length <= MAX_BUFFER_SIZE) {
                // Setup constant in new position
                memcpy(cache->hc_buffer + cache->nextFree, i.second.data, i.second.length);
                properties.at(i.first).rtc_offset = cache->nextFree;
                // Increase buffer offset length that has been added
                cache->nextFree += i.second.length;
            } else {
                // Ran out of constant cache space! (this can only trigger when a DefragMap is passed)
                // Arguably this check should be performed by init()
                // This should never happen, it would be caught by defrag sooner
                THROW OutOfMemory("Insufficient EnvProperty memory to create new properties, "
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
            auto &i = _i->second;
            auto mi_it = mapped_properties.find(i.first);
            if (mi_it ==  mapped_properties.end()) {
                // Property is not mapped, add it to cache
                size_t alignmentSize = _i->first;
                // Handle alignment
                const ptrdiff_t alignmentOffset = cache->nextFree % alignmentSize;
                const ptrdiff_t alignmentFix = alignmentOffset != 0 ? alignmentSize - alignmentOffset : 0;
                cache->nextFree += alignmentFix;
                if (cache->nextFree + i.second.length <= MAX_BUFFER_SIZE) {
                    // Setup constant in new position
                    memcpy(cache->hc_buffer + cache->nextFree, i.second.data, i.second.length);
                    properties.at(i.first).rtc_offset = cache->nextFree;
                    // Increase buffer offset length that has been added
                    cache->nextFree += i.second.length;
                } else {
                    // Ran out of constant cache space! (this can only trigger when a DefragMap is passed)
                    // Arguably this check should be performed by init()
                    // This should never happen, it would be caught by defrag sooner
                    THROW OutOfMemory("Insufficient EnvProperty memory to create new properties, "
                        "in EnvironmentManager::buildRTCOffsets().");
                }
            }
        }
        // Add a copy of cache for this instance_id to env manager
        rtc_caches.emplace(instance_id, cache);
    }
}
void EnvironmentManager::addRTCOffset(const NamePair &name) {
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
            THROW OutOfMemory("Insufficient EnvProperty memory to create new properties, "
                "in EnvironmentManager::buildRTCOffsets().");
        }
    } else {
        THROW OutOfMemory("Support for mapped (sub) properties is not currently implemented, "
            "in EnvironmentManager::addRTCOffset().");
    }
}

const CUDAAgentModel& EnvironmentManager::getCUDAAgentModel(const unsigned int &instance_id) {
    auto res = cuda_agent_models.find(instance_id);
    if (res == cuda_agent_models.end()) {
        THROW UnknownInternalError("Instance with id '%u' not registered in EnvironmentManager for use with RTC in EnvironmentManager::getCUDAAgentModel", instance_id);
    }
    return res->second;
}

void EnvironmentManager::updateRTCValue(const NamePair &name) {
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
    }
}

void EnvironmentManager::remove(const NamePair &name) {
    // Unregister in cuRVE
    Curve::getInstance().setNamespaceByHash(CURVE_NAMESPACE_HASH);
    Curve::VariableHash cvh = toHash(name);
    Curve::getInstance().unregisterVariableByHash(cvh);
    Curve::getInstance().setDefaultNamespace();
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
void EnvironmentManager::remove(const unsigned int &instance_id, const std::string &var_name) {
    remove({instance_id, var_name});
}

void EnvironmentManager::resetModel(const unsigned int &instance_id, const EnvironmentDescription &desc) {
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
    // Device must be init first
    initialiseDevice();

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
        // update RTC
        const CUDAAgentModel& cuda_agent_model = getCUDAAgentModel(instance_id);
        const auto &rtc_cache = rtc_caches.at(instance_id);
        cuda_agent_model.RTCUpdateEnvironmentVariables(rtc_cache->hc_buffer, rtc_cache->nextFree);
        // Update instance's rtc update flag
        rtc_update_required = false;
    }
    if (curve_registration_required) {
        Curve::getInstance().setNamespaceByHash(CURVE_NAMESPACE_HASH);
        // Update cub for any not mapped properties
        for (auto &p : properties) {
            if (p.first.first == instance_id) {
                // Generate hash for the subproperty name
                Curve::VariableHash cvh = toHash(p.first);
                const auto &prop = p.second;
                // Create the mapping
                const auto CURVE_RESULT = Curve::getInstance().registerVariableByHash(cvh, reinterpret_cast<void*>(prop.offset),
                        prop.length / prop.elements, prop.elements);
                if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
                    THROW CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE, "
                        "in EnvironmentManager::updateDevice().");
                }
#ifdef _DEBUG
                if (CURVE_RESULT != static_cast<int>((cvh+CURVE_NAMESPACE_HASH)%Curve::MAX_VARIABLES)) {
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
                const auto CURVE_RESULT = Curve::getInstance().registerVariableByHash(cvh, reinterpret_cast<void*>(masterprop.offset),
                        masterprop.length / masterprop.elements, masterprop.elements);
                if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
                    THROW CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE, "
                        "in EnvironmentManager::updateDevice().");
                }
#ifdef _DEBUG
                if (CURVE_RESULT != static_cast<int>((cvh+CURVE_NAMESPACE_HASH)%Curve::MAX_VARIABLES)) {
                    fprintf(stderr, "Curve Warning: Environment Property '%s' has a collision and may work improperly.\n", mp.first.second.c_str());
                }
#endif
            }
        }
        curve_registration_required = false;
        Curve::getInstance().setDefaultNamespace();
    }
}
