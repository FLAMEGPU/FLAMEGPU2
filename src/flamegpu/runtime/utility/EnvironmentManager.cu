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
    CURVE_NAMESPACE_HASH(Curve::getInstance().variableRuntimeHash(CURVE_NAMESPACE_STRING)),
    nextFree(0),
    m_freeSpace(EnvironmentManager::MAX_BUFFER_SIZE),
    freeFragments(),
    deviceInitialised(false) { }

void EnvironmentManager::purge() {
    deviceInitialised = false;
    freeFragments.clear();
    m_freeSpace = EnvironmentManager::MAX_BUFFER_SIZE;
    nextFree = 0;
    cuda_agent_models.clear();
    properties.clear();
    mapped_properties.clear();
}

void EnvironmentManager::init(const std::string& model_name, const EnvironmentDescription &desc) {
    // Initialise device portions of Environment manager
    initialiseDevice();
    // Error if reinit
    for (auto &&i : properties) {
        if (i.first.first == model_name) {
            THROW EnvDescriptionAlreadyLoaded("Environment description with same model name '%s' is already loaded, "
                "in EnvironmentManager::init().",
                model_name.c_str());
        }
    }
    // Build a DefragMap to send to defragger method
    DefragMap orderedProperties;
    size_t newSize = 0;
    for (auto _i = desc.properties.begin(); _i != desc.properties.end(); ++_i) {
        const auto &i = _i->second;
        NamePair name = toName(model_name, _i->first);
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
}
void EnvironmentManager::init(const std::string& model_name, const EnvironmentDescription &desc, const std::string& master_model_name, const SubEnvironmentData &mapping) {
    // Initialise device portions of Environment manager
    assert(deviceInitialised);  // submodel init should never be called first, requires parent init first for mapping
    // Error if reinit
    for (auto &&i : properties) {
        if (i.first.first == model_name) {
            THROW EnvDescriptionAlreadyLoaded("Environment description with same model name '%s' is already loaded, "
                "in EnvironmentManager::init().",
                model_name.c_str());
        }
    }

    // Build a DefragMap of to send to defragger method
    DefragMap orderedProperties;
    size_t newSize = 0;
    std::set<NamePair> new_mapped_props;
    for (auto _i = desc.properties.begin(); _i != desc.properties.end(); ++_i) {
        auto prop_mapping = mapping.properties.find(_i->first);
        const auto &i = _i->second;
        NamePair name = toName(model_name, _i->first);
        if (prop_mapping == mapping.properties.end()) {
            // Property is not mapped, so add to defrag map
            DefragProp prop = DefragProp(i.data.ptr, i.data.length, i.isConst, i.elements, i.type);
            const size_t typeSize = i.data.length / i.elements;
            orderedProperties.emplace(typeSize, std::make_pair(name, prop));
            newSize += i.data.length;
        } else {
            // Property is mapped, follow it's mapping upwards until we find the highest parent
            NamePair ultimateParent = toName(master_model_name, prop_mapping->second);
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
}

void EnvironmentManager::initRTC(const CUDAAgentModel& cuda_model) {
    // check to ensure that model name is not already registered
    auto res = cuda_agent_models.find(cuda_model.getModelDescription().name);
    if (res != cuda_agent_models.end()) {
        THROW UnknownInternalError("Agent model name '%s' already registered in initRTC()", cuda_model.getModelDescription().name.c_str());
    }
    // register model name
    cuda_agent_models.emplace(cuda_model.getModelDescription().name, cuda_model);

    // loop through environment properties, already registered by cuda_
    for (auto &p : properties) {
        if (p.first.first == cuda_model.getModelDescription().name) {
            auto var_name = p.first.second;
            auto src = hc_buffer + p.second.offset;
            auto length = p.second.length;
            // Register variable for use in any RTC functions
            cuda_model.RTCSetEnvironmentVariable(var_name.c_str(), src, length);
        }
    }
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
void EnvironmentManager::free(const std::string &model_name) {
    Curve::getInstance().setNamespaceByHash(CURVE_NAMESPACE_HASH);
    // Release regular properties
    for (auto &&i = properties.begin(); i != properties.end();) {
        if (i->first.first == model_name) {
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
        if (i->first.first == model_name) {
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
    auto cam = cuda_agent_models.find(model_name);
    if (cam != cuda_agent_models.end()) {
        cuda_agent_models.erase(cam);
    }
}

EnvironmentManager::NamePair EnvironmentManager::toName(const std::string &model_name, const std::string &var_name) {
    return std::make_pair(model_name, var_name);
}

/**
 * @note Not static, because eventually we might need to use curve singleton
 */
Curve::VariableHash EnvironmentManager::toHash(const NamePair &name) const {
    Curve::VariableHash model_cvh = Curve::getInstance().variableRuntimeHash(name.first.c_str());
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
    gpuErrchk(cudaMemcpy(reinterpret_cast<void*>(const_cast<char*>(c_buffer + buffOffset)), reinterpret_cast<void*>(hc_buffer + buffOffset), length, cudaMemcpyHostToDevice));
    // Register in cuRVE
    Curve::getInstance().setNamespaceByHash(CURVE_NAMESPACE_HASH);
    Curve::VariableHash cvh = toHash(name);
    const auto CURVE_RESULT = Curve::getInstance().registerVariableByHash(cvh, reinterpret_cast<void*>(const_cast<char*>(c_buffer + buffOffset)), typeSize, elements);
    if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
        THROW CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE"
            "in EnvironmentManager::add().");
    }
    Curve::getInstance().setDefaultNamespace();
}

void EnvironmentManager::defragment(DefragMap *mergeProperties, std::set<NamePair> newmaps) {
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
            const auto CURVE_RESULT = Curve::getInstance().registerVariableByHash(cvh, reinterpret_cast<void*>(const_cast<char*>(c_buffer + buffOffset)),
                typeSize, i.second.elements);
            if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
                THROW CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE, "
                    "in EnvironmentManager::defragment().");
            }
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
    // Replace buffer on device
    gpuErrchk(cudaMemcpy(reinterpret_cast<void*>(const_cast<char*>(c_buffer)), reinterpret_cast<void*>(hc_buffer), buffOffset, cudaMemcpyHostToDevice));
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
        const auto CURVE_RESULT = Curve::getInstance().registerVariableByHash(cvh, reinterpret_cast<void*>(const_cast<char*>(c_buffer + masterprop.offset)),
                masterprop.length / masterprop.elements, masterprop.elements);
        if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
            THROW CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE, "
                "in EnvironmentManager::defragment().");
        }
    }
    Curve::getInstance().setDefaultNamespace();
}

const CUDAAgentModel& EnvironmentManager::getCUDAAgentModel(std::string model_name) {
    auto res = cuda_agent_models.find(model_name);
    if (res == cuda_agent_models.end()) {
        THROW UnknownInternalError("Agent model name '%s' not registered in EnvironmentManager for use with RTC in EnvironmentManager::getCUDAAgentModel", model_name.c_str());
    }
    return res->second;
}

void EnvironmentManager::setRTCValue(const std::string &model_name, const std::string &variable_name, const void* src, size_t count, size_t offset) {
    const CUDAAgentModel& cuda_agent_model = getCUDAAgentModel(model_name);
    cuda_agent_model.RTCSetEnvironmentVariable(variable_name, src, count, offset);
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
}
void EnvironmentManager::remove(const std::string &model_name, const std::string &var_name) {
    remove({model_name, var_name});
}

void EnvironmentManager::resetModel(const std::string &model_name, const EnvironmentDescription &desc) {
    // Todo: Might want to change this, so EnvManager holds a copy of the default at init time
    // For every property, in the named model, which is not a mapped property
    for (auto &d : desc.getPropertiesMap()) {
        if (mapped_properties.find({model_name, d.first}) == mapped_properties.end()) {
            // Find the local property data
            auto &p = properties.at({model_name, d.first});
            // Set back to default value
            memcpy(hc_buffer + p.offset, d.second.data.ptr, d.second.data.length);
            assert(d.second.data.length == p.length);
            // Store data
            gpuErrchk(cudaMemcpy(reinterpret_cast<void*>(const_cast<char*>(c_buffer + p.offset)), reinterpret_cast<void*>(hc_buffer + p.offset), p.length, cudaMemcpyHostToDevice));
            // update RTC
            setRTCValue(model_name, d.first, hc_buffer + p.offset, p.length, p.offset);
        }
    }
}
