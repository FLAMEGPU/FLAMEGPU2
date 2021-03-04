#include "flamegpu/gpu/CUDAAgent.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <string>
// If MSVC earlier than VS 2019
#if defined(_MSC_VER) && _MSC_VER < 1920
#include <filesystem>
using std::tr2::sys::exists;
using std::tr2::sys::path;
#else
// VS2019 requires this macro, as building pre c++17 cant use std::filesystem
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
using std::experimental::filesystem::v1::exists;
using std::experimental::filesystem::v1::path;
#endif

#include "flamegpu/version.h"
#include "flamegpu/gpu/CUDAFatAgent.h"
#include "flamegpu/gpu/CUDAAgentStateList.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/gpu/CUDASimulation.h"

#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/runtime/cuRVE/curve_rtc.h"
#include "flamegpu/gpu/CUDAScatter.h"
#include "flamegpu/util/compute_capability.cuh"
#include "flamegpu/util/nvtx.h"

CUDAAgent::CUDAAgent(const AgentData& description, const CUDASimulation &_cuda_model)
    : agent_description(description)  // This is a master agent, so it must create a new fat_agent
    , fat_agent(std::make_shared<CUDAFatAgent>(agent_description))  // if we create fat agent, we're index 0
    , fat_index(0)
    , cuda_model(_cuda_model)
    , TOTAL_AGENT_VARIABLE_SIZE(calcTotalVarSize(description)) {
    // Generate state map from fat_agent
    auto fatstate_map = fat_agent->getStateMap(fat_index);
    for (auto &state : description.states) {
        // Find correct fat state
        auto fatstate = fatstate_map.at(state);
        // Construct a regular state map from this
        auto slimstate = std::make_shared<CUDAAgentStateList>(fatstate, *this, fat_index, agent_description);
        // Store in our map
        state_map.emplace(state, slimstate);
    }
}
CUDAAgent::CUDAAgent(
    const AgentData &description,
    const CUDASimulation &_cuda_model,
    const std::unique_ptr<CUDAAgent> &master_agent,
    const std::shared_ptr<SubAgentData> &mapping)
    : agent_description(description)
    , fat_agent(master_agent->getFatAgent())
    , fat_index(fat_agent->getMappedAgentCount())
    , cuda_model(_cuda_model)
    , TOTAL_AGENT_VARIABLE_SIZE(calcTotalVarSize(description)) {
    // This is next agent to be added to fat_agent, so it takes existing count
    // Pass required info, so fat agent can generate new buffers and mappings
    fat_agent->addSubAgent(agent_description, master_agent->getFatIndex(), mapping);
    // Generate state map from fat_agent
    auto fatstate_map = fat_agent->getStateMap(fat_index);
    for (auto &state : agent_description.states) {
        // Find correct fat state
        auto fatstate = fatstate_map.at(state);
        // Construct a regular state map from this
        auto slimstate = std::make_shared<CUDAAgentStateList>(fatstate, *this, fat_index, agent_description, mapping->states.find(state) != mapping->states.end(), mapping->variables);
        // Store in our map
        state_map.emplace(state, slimstate);
    }
}

void CUDAAgent::mapRuntimeVariables(const AgentFunctionData& func, const unsigned int &instance_id) const {
    // check the cuda agent state map to find the correct state list for functions starting state
    auto sm = state_map.find(func.initial_state);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::mapRuntimeVariables()",
            agent_description.name.c_str(), func.initial_state.c_str());
    }

    const Curve::VariableHash agent_hash = Curve::variableRuntimeHash(agent_description.name.c_str());
    const Curve::VariableHash func_hash = Curve::variableRuntimeHash(func.name.c_str());
    auto &curve = Curve::getInstance();
    const unsigned int agent_count = this->getStateSize(func.initial_state);
    // loop through the agents variables to map each variable name using cuRVE
    for (const auto &mmp : agent_description.variables) {
        // get a device pointer for the agent variable name
        void* d_ptr = sm->second->getVariablePointer(mmp.first);

        // map using curve
        const Curve::VariableHash var_hash = Curve::variableRuntimeHash(mmp.first.c_str());

        // get the agent variable size
        const size_t type_size = mmp.second.type_size * mmp.second.elements;

        // maximum population num
        if (func.func || func.condition) {
#ifdef _DEBUG
            const Curve::Variable cv = curve.registerVariableByHash(var_hash + agent_hash + func_hash + instance_id, d_ptr, type_size, agent_count);
            if (cv != static_cast<int>((var_hash + agent_hash + func_hash + instance_id)%Curve::MAX_VARIABLES)) {
                fprintf(stderr, "Curve Warning: Agent Function '%s' Variable '%s' has a collision and may work improperly.\n", func.name.c_str(), mmp.first.c_str());
            }
#else
            curve.registerVariableByHash(var_hash + agent_hash + func_hash + instance_id, d_ptr, type_size, agent_count);
#endif
        }
        // Map RTC variables to agent function (these must be mapped before each function execution as the runtime pointer may have changed to the swapping)
        if (!func.rtc_func_name.empty()) {
            // Copy data to rtc header cache
            auto& rtc_header = getRTCHeader(func.name);
            memcpy(rtc_header.getAgentVariableCachePtr(mmp.first.c_str()), &d_ptr, sizeof(void*));
        }

        // Map RTC variables to agent function conditions (these must be mapped before each function execution as the runtime pointer may have changed to the swapping)
        if (!func.rtc_func_condition_name.empty()) {
            // Copy data to rtc header cache
            std::string func_name = func.name + "_condition";
            auto& rtc_header = getRTCHeader(func_name);
            memcpy(rtc_header.getAgentVariableCachePtr(mmp.first.c_str()), &d_ptr, sizeof(void*));
        }
    }
}

void CUDAAgent::unmapRuntimeVariables(const AgentFunctionData& func, const unsigned int &instance_id) const {
    // Skip if RTC
    if (!(func.func || func.condition))
        return;
    // check the cuda agent state map to find the correct state list for functions starting state
    const auto &sm = state_map.find(func.initial_state);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::unmapRuntimeVariables()",
            agent_description.name.c_str(), func.initial_state.c_str());
    }

    const Curve::VariableHash agent_hash = Curve::variableRuntimeHash(agent_description.name.c_str());
    const Curve::VariableHash func_hash = Curve::variableRuntimeHash(func.name.c_str());
    // loop through the agents variables to map each variable name using cuRVE
    for (const auto &mmp : agent_description.variables) {
        // get a device pointer for the agent variable name
        // void* d_ptr = sm->second->getAgentListVariablePointer(mmp.first);

        // unmap using curve
        const Curve::VariableHash var_hash = Curve::variableRuntimeHash(mmp.first.c_str());
        Curve::getInstance().unregisterVariableByHash(var_hash + agent_hash + func_hash + instance_id);
    }

    // No current need to unmap RTC variables as they are specific to the agent functions and thus do not persist beyond the scope of a single function
}

void CUDAAgent::setPopulationData(const AgentVector& population, const std::string& state_name, CUDAScatter& scatter, const unsigned int& streamId, const cudaStream_t& stream) {
    // Validate agent state
    auto our_state = state_map.find(state_name);
    if (our_state == state_map.end()) {
        if (state_name == ModelData::DEFAULT_STATE) {
            THROW InvalidAgentState("Agent '%s' does not use the default state, so the state must be passed explicitly, "
                "in CUDAAgent::setPopulationData()",
                state_name.c_str(), population.getAgentName().c_str());
        } else {
            THROW InvalidAgentState("State '%s' was not found in agent '%s', "
                "in CUDAAgent::setPopulationData()",
                state_name.c_str(), population.getAgentName().c_str());
        }
    }
    // Copy population data
    // This call hierarchy validates agent desc matches
    our_state->second->setAgentData(population, scatter, streamId, stream);
}
void CUDAAgent::getPopulationData(AgentVector& population, const std::string& state_name) const {
    // Validate agent state
    auto our_state = state_map.find(state_name);
    if (our_state == state_map.end()) {
        if (state_name == ModelData::DEFAULT_STATE) {
            THROW InvalidAgentState("Agent '%s' does not use the default state, so the state must be passed explicitly, "
                "in CUDAAgent::getPopulationData()",
                state_name.c_str(), population.getAgentName().c_str());
        } else {
            THROW InvalidAgentState("State '%s' was not found in agent '%s', "
                "in CUDAAgent::getPopulationData()",
                state_name.c_str(), population.getAgentName().c_str());
        }
    }
    // Copy population data
    // This call hierarchy validates agent desc matches
    our_state->second->getAgentData(population);
}
/**
 * Returns the number of alive and active agents in the named state
 */
unsigned int CUDAAgent::getStateSize(const std::string &state) const {
    // check the cuda agent state map to find the correct state list for functions starting state
    const auto &sm = state_map.find(state);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found, "
            "in CUDAAgent::getStateSize()",
            agent_description.name.c_str(), state.c_str());
    }
    return sm->second->getSize();
}
/**
 * Returns the number of alive and active agents in the named state
 */
unsigned int CUDAAgent::getStateAllocatedSize(const std::string &state) const {
    // check the cuda agent state map to find the correct state list for functions starting state
    const auto &sm = state_map.find(state);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found, "
            "in CUDAAgent::getStateAllocatedSize()",
            agent_description.name.c_str(), state.c_str());
    }
    return sm->second->getAllocatedSize();
}
void CUDAAgent::resizeState(const std::string& state, const unsigned int& minimumSize, const bool& retainData) {
    // check the cuda agent state map to find the correct state list
    const auto& sm = state_map.find(state);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found, "
            "in CUDAAgent::getStateAllocatedSize()",
            agent_description.name.c_str(), state.c_str());
    }
    sm->second->resize(minimumSize, retainData);
}

void CUDAAgent::setStateAgentCount(const std::string& state, const unsigned int& newSize) {
    // check the cuda agent state map to find the correct state list
    const auto& sm = state_map.find(state);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found, "
            "in CUDAAgent::getStateAllocatedSize()",
            agent_description.name.c_str(), state.c_str());
    }
    sm->second->setAgentCount(newSize);
}
const AgentData &CUDAAgent::getAgentDescription() const {
    return agent_description;
}
void *CUDAAgent::getStateVariablePtr(const std::string &state_name, const std::string &variable_name) {
    // check the cuda agent state map to find the correct state list for functions starting state
    const auto &sm = state_map.find(state_name);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found, "
            "in CUDAAgent::getStateVariablePtr()",
            agent_description.name.c_str(), state_name.c_str());
    }
    return sm->second->getVariablePointer(variable_name);
}
/**
 * Processes agent death, this call is forwarded to the fat agent
 * All disabled agents are scattered to swap
 * Only alive agents with deathflag are scattered
 * @param func The agent function condition being processed
 * @param streamId The index of the agent function within the current layer
 * @see CUDAFatAgent::processDeath(const unsigned int &, const std::string &, const unsigned int &)
 */
void CUDAAgent::processDeath(const AgentFunctionData& func, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    // Optionally process agent death
    if (func.has_agent_death) {
        // Agent death operates on all mapped vars, so handled by fat agent
        fat_agent->processDeath(fat_index, func.initial_state, scatter, streamId, stream);
    }
}
/**
 * Transitions all active agents from the source state to the destination state
 * @param _src The source state
 * @param _dest The destination state
 * @param streamId The index of the agent function within the current layer
 * @see CUDAFatAgent::transitionState(const unsigned int &, const std::string &, const std::string &, const unsigned int &)
 */
void CUDAAgent::transitionState(const std::string &_src, const std::string &_dest, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    // All mapped vars need to transition too, so handled by fat agent
    fat_agent->transitionState(fat_index, _src, _dest, scatter, streamId, stream);
}
/**
 * Scatters agents based on their output of the agent function condition
 * Agents which failed the condition are scattered to the front and marked as disabled
 * Agents which pass the condition are scatterd to after the disabled agents
 * @param func The agent function condition being processed
 * @param streamId The index of the agent function within the current layer
 * @see CUDAFatAgent::processFunctionCondition(const unsigned int &, const unsigned int &)
 * @note Named state must not already contain disabled agents
 * @note The disabled agents are re-enabled using clearFunctionCondition(const std::string &)
 */
void CUDAAgent::processFunctionCondition(const AgentFunctionData& func, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    // Optionally process function condition
    if ((func.condition) || (!func.rtc_func_condition_name.empty())) {
        // Agent function condition operates on all mapped vars, so handled by fat agent
        fat_agent->processFunctionCondition(fat_index, func.initial_state, scatter, streamId, stream);
    }
}
void CUDAAgent::scatterHostCreation(const std::string &state_name, const unsigned int &newSize, char *const d_inBuff, const VarOffsetStruct &offsets, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    auto sm = state_map.find(state_name);
    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::scatterHostCreation()",
            agent_description.name.c_str(), state_name.c_str());
    }
    sm->second->scatterHostCreation(newSize, d_inBuff, offsets, scatter, streamId, stream);
}
void CUDAAgent::scatterSort(const std::string &state_name, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    auto sm = state_map.find(state_name);
    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::scatterHostCreation()",
            agent_description.name.c_str(), state_name.c_str());
    }
    sm->second->scatterSort(scatter, streamId, stream);
}
void CUDAAgent::mapNewRuntimeVariables(const CUDAAgent& func_agent, const AgentFunctionData& func, const unsigned int &maxLen, CUDAScatter &scatter, const unsigned int &instance_id, const unsigned int &streamId) {
    // Confirm agent output is set
    if (auto oa = func.agent_output.lock()) {
        // check the cuda agent state map to find the correct state list for functions starting state
        auto sm = state_map.find(func.agent_output_state);

        if (sm == state_map.end()) {
            THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
                "in CUDAAgent::mapNewRuntimeVariables()",
                agent_description.name.c_str(), func.agent_output_state.c_str());
        }
        // Notify scan flag that it might need resizing
        // We need a 3rd array, because a function might combine agent birth, agent death and message output
        scatter.Scan().resize(maxLen, CUDAScanCompaction::AGENT_OUTPUT, streamId);
        // Ensure the scan flag is zeroed
        scatter.Scan().zero(CUDAScanCompaction::AGENT_OUTPUT, streamId);

        // Request a buffer for new
        char *d_new_buffer = static_cast<char*>(fat_agent->allocNewBuffer(TOTAL_AGENT_VARIABLE_SIZE, maxLen, agent_description.variables.size()));

        // Store buffer so we can release it later
        {
            std::lock_guard<std::mutex> guard(newBuffsMutex);
            const auto rtn = newBuffs.emplace(func.initial_state, d_new_buffer);
            if (!rtn.second) {
                assert(false);  // Insertion happened (false if element already exists)
            }
        }

        // Init the buffer to default values for variables
        scatter.broadcastInit(
            streamId,
            0,
            agent_description.variables,
            d_new_buffer,
            maxLen, 0);

        // Map variables to curve
        const Curve::VariableHash _agent_birth_hash = Curve::variableRuntimeHash("_agent_birth");
        const Curve::VariableHash func_hash = Curve::variableRuntimeHash(func.name.c_str());
        auto &curve = Curve::getInstance();
        // loop through the agents variables to map each variable name using cuRVE
        for (const auto &mmp : agent_description.variables) {
            // map using curve
            const Curve::VariableHash var_hash = Curve::variableRuntimeHash(mmp.first.c_str());

            // get the agent variable size
            const size_t type_size = mmp.second.type_size * mmp.second.elements;

            // get a device pointer for the agent variable name
            void* d_ptr = d_new_buffer;

            // Move the pointer along for next variable
            d_new_buffer += type_size * maxLen;

            // 64 bit align the new buffer start
            if (reinterpret_cast<size_t>(d_new_buffer)%8) {
                d_new_buffer += 8 - (reinterpret_cast<size_t>(d_new_buffer)%8);
            }

            // maximum population num
            if (func.func) {
#ifdef _DEBUG
                const Curve::Variable cv = curve.registerVariableByHash(var_hash + (_agent_birth_hash ^ func_hash) + instance_id, d_ptr, type_size, maxLen);
                if (cv != static_cast<int>((var_hash + (_agent_birth_hash ^ func_hash) + instance_id)%Curve::MAX_VARIABLES)) {
                    fprintf(stderr, "Curve Warning: Agent Function '%s' New Agent Variable '%s' has a collision and may work improperly.\n", func.name.c_str(), mmp.first.c_str());
                }
#else
                curve.registerVariableByHash(var_hash + (_agent_birth_hash ^ func_hash) + instance_id, d_ptr, type_size, maxLen);
#endif
            } else  {
                // Map RTC variables (these must be mapped before each function execution as the runtime pointer may have changed to the swapping)
                // Copy data to rtc header cache
                auto& rtc_header = func_agent.getRTCHeader(func.name);
                memcpy(rtc_header.getNewAgentVariableCachePtr(mmp.first.c_str()), &d_ptr, sizeof(void*));
            }
        }
    }
}
void CUDAAgent::unmapNewRuntimeVariables(const AgentFunctionData& func, const unsigned int &instance_id) {
    // Confirm agent output is set
    if (auto oa = func.agent_output.lock()) {
        // Release new buffer
        {
            std::lock_guard<std::mutex> guard(newBuffsMutex);
            const auto d_buff = newBuffs.find(func.initial_state);
            if (d_buff != newBuffs.end()) {
                fat_agent->freeNewBuffer(d_buff->second);
                newBuffs.erase(d_buff);
            } else {
                assert(false);  // We don't have a new buffer reserved???
            }
        }
        // Skip if RTC
        if (!func.func)
            return;
        // Unmap curve
        const Curve::VariableHash _agent_birth_hash = Curve::variableRuntimeHash("_agent_birth");
        const Curve::VariableHash func_hash = Curve::variableRuntimeHash(func.name.c_str());
        auto &curve = Curve::getInstance();
        // loop through the agents variables to map each variable name using cuRVE
        for (const auto &mmp : agent_description.variables) {
            // unmap using curve
            const Curve::VariableHash var_hash = Curve::variableRuntimeHash(mmp.first.c_str());
            curve.unregisterVariableByHash(var_hash + (_agent_birth_hash ^ func_hash) + instance_id);

            // no need to unmap RTC variables
        }
    }
}

void CUDAAgent::scatterNew(const AgentFunctionData& func, const unsigned int &newSize, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    // Confirm agent output is set
    if (auto oa = func.agent_output.lock()) {
        auto sm = state_map.find(func.agent_output_state);
        if (sm == state_map.end()) {
            THROW InvalidStateName("Agent '%s' does not contain state '%s', "
                "in CUDAAgent::scatterNew()\n",
                agent_description.name.c_str(), func.agent_output_state.c_str());
        }
        // Find new buffer
        void *newBuff = nullptr;
        {
            std::lock_guard<std::mutex> guard(newBuffsMutex);
            const auto d_buff = newBuffs.find(func.initial_state);
            if (d_buff != newBuffs.end()) {
                newBuff = d_buff->second;
            }
        }
        if (!newBuff) {
            THROW InvalidAgentFunc("New buffer not present for function within init state: %s,"
                " in CUDAAgent::scatterNew()\n",
                func.initial_state.c_str());
        }
        sm->second->scatterNew(newBuff, newSize, scatter, streamId, stream);
    }
}
void CUDAAgent::clearFunctionCondition(const std::string &state) {
    fat_agent->setConditionState(fat_index, state, 0);
}

void CUDAAgent::addInstantitateRTCFunction(const AgentFunctionData& func, bool function_condition) {
    // Generate the dynamic curve header
    CurveRTCHost &curve_header = *rtc_header_map.emplace(function_condition ? func.name + "_condition" : func.name, std::make_unique<CurveRTCHost>()).first->second;
    // agent function hash
    Curve::NamespaceHash agentname_hash = Curve::variableRuntimeHash(this->getAgentDescription().name.c_str());
    Curve::NamespaceHash funcname_hash = Curve::variableRuntimeHash(func.name.c_str());
    Curve::NamespaceHash agent_func_name_hash = agentname_hash + funcname_hash;

    // set agent function variables in rtc curve
    for (const auto& mmp : func.parent.lock()->variables) {
        curve_header.registerAgentVariable(mmp.first.c_str(), agent_func_name_hash, mmp.second.type.name(), mmp.second.type_size, mmp.second.elements);
    }

    // for normal agent function (e.g. not an agent function condition) append messages and agent outputs
    if (!function_condition) {
        // Set input message variables in curve
        if (auto im = func.message_input.lock()) {
            // get the message input hash
            Curve::NamespaceHash msg_in_hash = Curve::variableRuntimeHash(im->name.c_str());
            for (auto msg_in_var : im->variables) {
                // register message variables using combined hash
                curve_header.registerMessageInVariable(msg_in_var.first.c_str(), msg_in_hash + agent_func_name_hash,
                msg_in_var.second.type.name(), msg_in_var.second.type_size, msg_in_var.second.elements, true, false);
            }
        }
        // Set output message variables in curve
        if (auto om = func.message_output.lock()) {
            // get the message input hash
            Curve::NamespaceHash msg_out_hash = Curve::variableRuntimeHash(om->name.c_str());
            for (auto msg_out_var : om->variables) {
                // register message variables using combined hash
                curve_header.registerMessageOutVariable(msg_out_var.first.c_str(), msg_out_hash + agent_func_name_hash,
                msg_out_var.second.type.name(), msg_out_var.second.type_size, msg_out_var.second.elements, false, true);
            }
        }
        // Set agent output variables in curve
        if (auto ao = func.agent_output.lock()) {
            // get the message input hash
            Curve::NamespaceHash agent_out_hash = Curve::variableRuntimeHash("_agent_birth");
            for (auto agent_out_var : ao->variables) {
                // register message variables using combined hash
                curve_header.registerNewAgentVariable(agent_out_var.first.c_str(), agent_out_hash + funcname_hash,
                agent_out_var.second.type.name(), agent_out_var.second.type_size, agent_out_var.second.elements, false, true);
            }
        }
    }

    // Set Environment variables in curve
    {
        // Scope the mutex
        auto lock = EnvironmentManager::getInstance().getSharedLock();
        const auto &prop_map = EnvironmentManager::getInstance().getPropertiesMap();
        for (auto p : prop_map) {
            if (p.first.first == cuda_model.getInstanceID()) {
                const char* variableName = p.first.second.c_str();
                const char* type = p.second.type.name();
                unsigned int elements = p.second.elements;
                ptrdiff_t offset = p.second.rtc_offset;
                curve_header.registerEnvVariable(variableName, cuda_model.getInstanceID(), offset, type, p.second.length/elements, elements);
            }
        }
        // Set mapped environment variables in curve
        for (const auto mp : EnvironmentManager::getInstance().getMappedProperties()) {
            if (mp.first.first == cuda_model.getInstanceID()) {
                auto p = prop_map.at(mp.second.masterProp);
                const char* variableName = mp.second.masterProp.second.c_str();
                const char* type = p.type.name();
                unsigned int elements = p.elements;
                ptrdiff_t offset = p.rtc_offset;
                curve_header.registerEnvVariable(variableName, cuda_model.getInstanceID(), offset, type, p.length/elements, elements);
            }
        }
    }

    // get the dynamically generated header from curve rtc
    std::string curve_dynamic_header = curve_header.getDynamicHeader();

    // output to disk if OUTPUT_RTC_DYNAMIC_FILES macro is set
#ifdef OUTPUT_RTC_DYNAMIC_FILES
        // create string for agent function implementation
        std::string func_impl = std::string(func.rtc_func_name).append("_impl");
        // curve
        std::ofstream file_curve_rtc_header;
        std::string file_curve_rtc_header_filename = func_impl.c_str();
        if (function_condition)
            file_curve_rtc_header_filename.append("_condition");
        file_curve_rtc_header_filename.append("_curve_rtc_dynamic.h");
        file_curve_rtc_header.open(file_curve_rtc_header_filename);
        // Remove first line as it is the filename, which misaligns profiler
        std::string out_s = curve_dynamic_header;
        out_s.erase(0, out_s.find("\n") + 1);
        file_curve_rtc_header << out_s;
        file_curve_rtc_header.close();
        // agent function
        std::ofstream agent_function_file;
        std::string agent_function_filename = func_impl.c_str();
        if (function_condition)
            agent_function_filename.append("_condition");
        agent_function_filename.append(".cu");
        agent_function_file.open(agent_function_filename);
        // Remove first line as it is the filename, which misaligns profiler
        out_s = func.rtc_source;
        out_s.erase(0, out_s.find("\n") + 1);
        agent_function_file << out_s;
        agent_function_file.close();
#endif

    JitifyCache &jitify = JitifyCache::getInstance();
    // switch between normal agent function and agent function condition
    if (!function_condition) {
        const std::string t_func_impl = std::string(func.rtc_func_name).append("_impl");
        const std::vector<std::string> template_args = { t_func_impl.c_str(), func.msg_in_type.c_str(), func.msg_out_type.c_str() };
        auto kernel_inst = jitify.loadKernel(func.rtc_func_name, template_args, func.rtc_source, curve_dynamic_header);
        // add kernel instance to map
        rtc_func_map.insert(CUDARTCFuncMap::value_type(func.name, std::move(kernel_inst)));
    } else {
        const std::string t_func_impl = std::string(func.rtc_func_condition_name).append("_cdn_impl");
        const std::vector<std::string> template_args = { t_func_impl.c_str() };
        auto kernel_inst = jitify.loadKernel(func.rtc_func_name + "_condition", template_args, func.rtc_condition_source, curve_dynamic_header);
        // add kernel instance to map
        rtc_func_map.insert(CUDARTCFuncMap::value_type(func.name + "_condition", std::move(kernel_inst)));
    }
}

const jitify::experimental::KernelInstantiation& CUDAAgent::getRTCInstantiation(const std::string &function_name) const {
    CUDARTCFuncMap::const_iterator mm = rtc_func_map.find(function_name);
    if (mm == rtc_func_map.end()) {
        THROW InvalidAgentFunc("Function name '%s' is not a runtime compiled agent function in agent '%s', "
            "in CUDAAgent::getRTCInstantiation()\n",
            function_name.c_str(), agent_description.name.c_str());
    }

    return *mm->second;
}
CurveRTCHost& CUDAAgent::getRTCHeader(const std::string& function_name) const {
    CUDARTCHeaderMap::const_iterator mm = rtc_header_map.find(function_name);
    if (mm == rtc_header_map.end()) {
        THROW InvalidAgentFunc("Function name '%s' is not a runtime compiled agent function in agent '%s', "
            "in CUDAAgent::getRTCHeader()\n",
            function_name.c_str(), agent_description.name.c_str());
    }

    return *mm->second;
}

const CUDAAgent::CUDARTCFuncMap& CUDAAgent::getRTCFunctions() const {
    return rtc_func_map;
}

void CUDAAgent::initUnmappedVars(CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    for (auto &s : state_map) {
        s.second->initUnmappedVars(scatter, streamId, stream);
    }
}
void CUDAAgent::initExcludedVars(const std::string &state, const unsigned int&count, const unsigned int&offset, CUDAScatter& scatter, const unsigned int& streamId, const cudaStream_t& stream) {
    // check the cuda agent state map to find the correct state list
    const auto& sm = state_map.find(state);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found, "
            "in CUDAAgent::initUnmappedVars()",
            agent_description.name.c_str(), state.c_str());
    }
    sm->second->initExcludedVars(count, offset, scatter, streamId, stream);
}
void CUDAAgent::cullUnmappedStates() {
    for (auto &s : state_map) {
        if (!s.second->getIsSubStatelist()) {
            s.second->clear();
        }
    }
}
void CUDAAgent::cullAllStates() {
    for (auto &s : state_map) {
        s.second->clear();
    }
}
std::list<std::shared_ptr<VariableBuffer>> CUDAAgent::getUnboundVariableBuffers(const std::string& state) {
    const auto& sm = state_map.find(state);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found, "
            "in CUDAAgent::getUnboundVariableBuffers()",
            agent_description.name.c_str(), state.c_str());
    }
    return sm->second->getUnboundVariableBuffers();
}
