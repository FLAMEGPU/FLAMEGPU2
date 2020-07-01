#include "flamegpu/gpu/CUDAAgent.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/gpu/CUDAFatAgent.h"
#include "flamegpu/gpu/CUDAAgentStateList.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/gpu/CUDAAgentModel.h"

#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/runtime/cuRVE/curve_rtc.h"
#include "flamegpu/gpu/CUDAScatter.h"

CUDAAgent::CUDAAgent(const AgentData& description, const CUDAAgentModel &_cuda_model)
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
    const CUDAAgentModel &_cuda_model,
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

void CUDAAgent::mapRuntimeVariables(const AgentFunctionData& func) const {
    // check the cuda agent state map to find the correct state list for functions starting state
    auto sm = state_map.find(func.initial_state);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::mapRuntimeVariables()",
            agent_description.name.c_str(), func.initial_state.c_str());
    }

    const Curve::VariableHash agent_hash = Curve::variableRuntimeHash(agent_description.name.c_str());
    const Curve::VariableHash func_hash = Curve::variableRuntimeHash(func.name.c_str());
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
        Curve::getInstance().registerVariableByHash(var_hash + agent_hash + func_hash, d_ptr, type_size, agent_count);

        // Map RTC variables to agent function (these must be mapped before each function execution as the runtime pointer may have changed to the swapping)
        if (!func.rtc_func_name.empty()) {
            // get the rtc varibale ptr
            const jitify::KernelInstantiation& instance = getRTCInstantiation(func.rtc_func_name);
            std::stringstream d_var_ptr_name;
            d_var_ptr_name << CurveRTCHost::getVariableSymbolName(mmp.first.c_str(), agent_hash + func_hash);
            CUdeviceptr d_var_ptr = instance.get_global_ptr(d_var_ptr_name.str().c_str());
            // copy runtime ptr (d_ptr) to rtc ptr (d_var_ptr)
            gpuErrchkDriverAPI(cuMemcpyHtoD(d_var_ptr, &d_ptr, sizeof(void*)));
        }

        // Map RTC variables to agent function conditions (these must be mapped before each function execution as the runtime pointer may have changed to the swapping)
        if (!func.rtc_func_condition_name.empty()) {
            // get the rtc varibale ptr
            std::string func_name = func.name + "_condition";
            const jitify::KernelInstantiation& instance = getRTCInstantiation(func_name);
            std::stringstream d_var_ptr_name;
            d_var_ptr_name << CurveRTCHost::getVariableSymbolName(mmp.first.c_str(), agent_hash + func_hash);
            CUdeviceptr d_var_ptr = instance.get_global_ptr(d_var_ptr_name.str().c_str());
            // copy runtime ptr (d_ptr) to rtc ptr (d_var_ptr)
            gpuErrchkDriverAPI(cuMemcpyHtoD(d_var_ptr, &d_ptr, sizeof(void*)));
        }
    }
}

void CUDAAgent::unmapRuntimeVariables(const AgentFunctionData& func) const {
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
        Curve::getInstance().unregisterVariableByHash(var_hash + agent_hash + func_hash);
    }

    // No current need to unmap RTC variables as they are specific to the agent functions and thus do not persist beyond the scope of a single function
}

void CUDAAgent::setPopulationData(const AgentPopulation& population, CUDAScatter &scatter, const unsigned int &streamId) {
    // Make sure population uses same agent description as was used to initialise the agent CUDAAgent
    if (population.getAgentDescription() != agent_description) {
        THROW InvalidCudaAgentDesc("Agent State memory has different description to CUDA Agent ('%s'), "
            "in CUDAAgentStateList::setPopulationData().",
            agent_description.name.c_str());
    }
    // For each possible state
    for (const auto &state : agent_description.states) {
        auto &our_state = state_map.at(state);
        // Copy population data
        our_state->setAgentData(population.getReadOnlyStateMemory(state), scatter, streamId);
    }
}

void CUDAAgent::getPopulationData(AgentPopulation& population) const {
    // Make sure population uses same agent description as was used to initialise the agent CUDAAgent
    if (population.getAgentDescription() != agent_description) {
        THROW InvalidCudaAgentDesc("Agent State memory has different description to CUDA Agent ('%s'), "
            "in CUDAAgentStateList::getPopulationData().",
            agent_description.name.c_str());
    }
    // For each possible state
    for (const auto &state : agent_description.states) {
        auto &our_state = state_map.at(state);
        // All buffers in Agent pop are same size, so resize here
        if (population.getMaximumStateListCapacity() < our_state->getSize())
            population.setStateListCapacity(our_state->getSize());
        // Copy population data
        our_state->getAgentData(population.getStateMemory(state));
    }
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
void CUDAAgent::processDeath(const AgentFunctionData& func, CUDAScatter &scatter, const unsigned int &streamId) {
    // Optionally process agent death
    if (func.has_agent_death) {
        // Agent death operates on all mapped vars, so handled by fat agent
        fat_agent->processDeath(fat_index, func.initial_state, scatter, streamId);
    }
}
/**
 * Transitions all active agents from the source state to the destination state
 * @param _src The source state
 * @param _dest The destination state
 * @param streamId The index of the agent function within the current layer
 * @see CUDAFatAgent::transitionState(const unsigned int &, const std::string &, const std::string &, const unsigned int &)
 */
void CUDAAgent::transitionState(const std::string &_src, const std::string &_dest, CUDAScatter &scatter, const unsigned int &streamId) {
    // All mapped vars need to transition too, so handled by fat agent
    fat_agent->transitionState(fat_index, _src, _dest, scatter, streamId);
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
void CUDAAgent::processFunctionCondition(const AgentFunctionData& func, CUDAScatter &scatter, const unsigned int &streamId) {
    // Optionally process function condition
    if ((func.condition) || (!func.rtc_func_condition_name.empty())) {
        // Agent function condition operates on all mapped vars, so handled by fat agent
        fat_agent->processFunctionCondition(fat_index, func.initial_state, scatter, streamId);
    }
}
/**
 * 
 */
void CUDAAgent::scatterHostCreation(const std::string &state_name, const unsigned int &newSize, char *const d_inBuff, const VarOffsetStruct &offsets, CUDAScatter &scatter, const unsigned int &streamId) {
    auto sm = state_map.find(state_name);
    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::scatterHostCreation()",
            agent_description.name.c_str(), state_name.c_str());
    }
    sm->second->scatterHostCreation(newSize, d_inBuff, offsets, scatter, streamId);
}
void CUDAAgent::scatterSort(const std::string &state_name, CUDAScatter &scatter, const unsigned int &streamId) {
    auto sm = state_map.find(state_name);
    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::scatterHostCreation()",
            agent_description.name.c_str(), state_name.c_str());
    }
    sm->second->scatterSort(scatter, streamId);
}
void CUDAAgent::mapNewRuntimeVariables(const AgentFunctionData& func, const unsigned int &maxLen, CUDAScatter &scatter, const unsigned int &streamId) {
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
            agent_description.variables,
            d_new_buffer,
            maxLen, 0);

        // Map variables to curve
        const Curve::VariableHash _agent_birth_hash = Curve::variableRuntimeHash("_agent_birth");
        const Curve::VariableHash func_hash = Curve::variableRuntimeHash(func.name.c_str());
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
            Curve::getInstance().registerVariableByHash(var_hash + _agent_birth_hash + func_hash, d_ptr, type_size, maxLen);

            // Map RTC variables (these must be mapped before each function execution as the runtime pointer may have changed to the swapping)
            if (!func.rtc_func_name.empty()) {
                // get the rtc variable ptr
                const jitify::KernelInstantiation& instance = getRTCInstantiation(func.rtc_func_name);
                std::stringstream d_var_ptr_name;
                d_var_ptr_name << CurveRTCHost::getVariableSymbolName(mmp.first.c_str(), _agent_birth_hash + func_hash);
                CUdeviceptr d_var_ptr = instance.get_global_ptr(d_var_ptr_name.str().c_str());
                // copy runtime ptr (d_ptr) to rtc ptr (d_var_ptr)
                gpuErrchkDriverAPI(cuMemcpyHtoD(d_var_ptr, &d_ptr, sizeof(void*)));
            }
        }
    }
}
void CUDAAgent::unmapNewRuntimeVariables(const AgentFunctionData& func) {
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
        // Unmap curve
        const Curve::VariableHash _agent_birth_hash = Curve::variableRuntimeHash("_agent_birth");
        const Curve::VariableHash func_hash = Curve::variableRuntimeHash(func.name.c_str());
        // loop through the agents variables to map each variable name using cuRVE
        for (const auto &mmp : agent_description.variables) {
            // get a device pointer for the agent variable name
            // void* d_ptr = sm->second->getAgentListVariablePointer(mmp.first);

            // unmap using curve
            const Curve::VariableHash var_hash = Curve::variableRuntimeHash(mmp.first.c_str());
            Curve::getInstance().unregisterVariableByHash(var_hash + _agent_birth_hash + func_hash);

            // no need to unmap RTC variables
        }
    }
}

void CUDAAgent::scatterNew(const AgentFunctionData& func, const unsigned int &newSize, CUDAScatter &scatter, const unsigned int &streamId) {
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
        sm->second->scatterNew(newBuff, newSize, scatter, streamId);
    }
}
void CUDAAgent::clearFunctionCondition(const std::string &state) {
    fat_agent->setConditionState(fat_index, state, 0);
}

void CUDAAgent::addInstantitateRTCFunction(const AgentFunctionData& func, bool function_condition) {
    // get header location for fgpu
    const char* env_inc_fgp2 = std::getenv("FLAMEGPU2_INC_DIR");
    if (!env_inc_fgp2) {
        THROW InvalidAgentFunc("Error compiling runtime agent function ('%s'): FLAMEGPU2_INC_DIR environment variable does not exist, "
            "in CUDAAgent::addInstantitateRTCFunction().",
            func.name.c_str());
    }

    // get the cuda path
    const char* env_cuda_path = std::getenv("CUDA_PATH");
    if (!env_cuda_path) {
        THROW InvalidAgentFunc("Error compiling runtime agent function ('%s'): CUDA_PATH environment variable does not exist, "
            "in CUDAAgent::addInstantitateRTCFunction().",
            func.name.c_str());
    }

    // vector of compiler options for jitify
    std::vector<std::string> options;
    std::vector<std::string> headers;

    // fpgu incude
    std::string include_fgpu;
    include_fgpu = "-I" + std::string(env_inc_fgp2);
    options.push_back(include_fgpu);

    // cuda path
    std::string include_cuda;
    include_cuda = "-I" + std::string(env_cuda_path) + "/include";
    options.push_back(include_cuda);

    // cuda.h
    std::string include_cuda_h;
    include_cuda_h = "--pre-include=" + std::string(env_cuda_path) + "/include/cuda.h";
    options.push_back(include_cuda_h);

    // curve rtc header
    CurveRTCHost curve_header;
    // agent function hash
    Curve::NamespaceHash agentname_hash = Curve::variableRuntimeHash(this->getAgentDescription().name.c_str());
    Curve::NamespaceHash funcname_hash = Curve::variableRuntimeHash(func.name.c_str());
    Curve::NamespaceHash agent_func_name_hash = agentname_hash + funcname_hash;

    // set agent function variables in rtc curve
    for (const auto& mmp : func.parent.lock()->variables) {
        curve_header.registerVariable(mmp.first.c_str(), agent_func_name_hash, mmp.second.type.name(), mmp.second.elements);
    }

    // for normal agent function (e.g. not an agent function condition) append messages and agent outputs
    if (!function_condition) {
        // Set input message variables in curve
        if (auto im = func.message_input.lock()) {
            // get the message input hash
            Curve::NamespaceHash msg_in_hash = Curve::variableRuntimeHash(im->name.c_str());
            for (auto msg_in_var : im->variables) {
                // register message variables using combined hash
                curve_header.registerVariable(msg_in_var.first.c_str(), msg_in_hash + agent_func_name_hash, msg_in_var.second.type.name(), msg_in_var.second.elements, true, false);
            }
        }
        // Set output message variables in curve
        if (auto om = func.message_output.lock()) {
            // get the message input hash
            Curve::NamespaceHash msg_out_hash = Curve::variableRuntimeHash(om->name.c_str());
            for (auto msg_out_var : om->variables) {
                // register message variables using combined hash
                curve_header.registerVariable(msg_out_var.first.c_str(), msg_out_hash + agent_func_name_hash, msg_out_var.second.type.name(), msg_out_var.second.elements, false, true);
            }
        }
        // Set agent output variables in curve
        if (auto ao = func.agent_output.lock()) {
            // get the message input hash
            Curve::NamespaceHash agent_out_hash = Curve::variableRuntimeHash("_agent_birth");
            for (auto agent_out_var : ao->variables) {
                // register message variables using combined hash
                curve_header.registerVariable(agent_out_var.first.c_str(), agent_out_hash + funcname_hash, agent_out_var.second.type.name(), agent_out_var.second.elements, false, true);
            }
        }
    }

    // Set Environment variables in curve
    Curve::NamespaceHash model_hash = Curve::variableRuntimeHash(cuda_model.getModelDescription().name.c_str());
    for (auto p : EnvironmentManager::getInstance().getPropertiesMap()) {
        if (p.first.first == cuda_model.getModelDescription().name) {
            const char* variableName = p.first.second.c_str();
            const char* type = p.second.type.name();
            unsigned int elements = p.second.elements;
            curve_header.registerEnvVariable(variableName, model_hash, type, elements);
        }
    }

    // get the dynamically generated header from curve rtc
    headers.push_back(curve_header.getDynamicHeader());

    // cassert header (to remove remaining warnings) TODO: Ask Jitify to implement safe version of this
    std::string cassert_h = "cassert\n";
    headers.push_back(cassert_h);

    // jitify to create program (with compilation settings)
    try {
        static jitify::JitCache kernel_cache;
        // switch between normal agent function and agent function condition
        if (!function_condition) {
            auto program = kernel_cache.program(func.rtc_source, headers, options);
            // create jifity instance
            auto kernel = program.kernel("agent_function_wrapper");
            // create string for agent function implementation
            std::string func_impl = std::string(func.rtc_func_name).append("_impl");
            // add kernal instance to map
            rtc_func_map.insert(CUDARTCFuncMap::value_type(func.name, std::unique_ptr<jitify::KernelInstantiation>(new jitify::KernelInstantiation(kernel, { func_impl.c_str(), func.msg_in_type.c_str(), func.msg_out_type.c_str() }))));
        } else {
            auto program = kernel_cache.program(func.rtc_condition_source, headers, options);
            // create jifity instance
            auto kernel = program.kernel("agent_function_condition_wrapper");
            // create string for agent function implementation
            std::string func_impl = std::string(func.rtc_func_condition_name).append("_cdn_impl");
            // add kernal instance to map
            std::string func_name = func.name + "_condition";
            rtc_func_map.insert(CUDARTCFuncMap::value_type(func_name, std::unique_ptr<jitify::KernelInstantiation>(new jitify::KernelInstantiation(kernel, { func_impl.c_str()}))));
        }
    }
    catch (std::runtime_error const&) {
        // jitify does not have a method for getting compile logs so rely on JITIFY_PRINT_LOG defined in cmake
        THROW InvalidAgentFunc("Error compiling runtime agent function (or function condition) ('%s'): function had compilation errors (see std::cout), "
            "in CUDAAgent::addInstantitateRTCFunction().",
            func.name.c_str());
    }
}

const jitify::KernelInstantiation& CUDAAgent::getRTCInstantiation(const std::string &function_name) const {
    CUDARTCFuncMap::const_iterator mm = rtc_func_map.find(function_name);
    if (mm == rtc_func_map.end()) {
        THROW InvalidAgentFunc("Function name '%s' is not a runtime compiled agent function , "
            "in CUDAAgent::getRTCInstantiation()\n",
            function_name.c_str());
    }

    return *mm->second;
}

const CUDAAgent::CUDARTCFuncMap& CUDAAgent::getRTCFunctions() const {
    return rtc_func_map;
}

void CUDAAgent::initUnmappedVars(CUDAScatter &scatter, const unsigned int &streamId) {
    for (auto &s : state_map) {
        s.second->initUnmappedVars(scatter, streamId);
    }
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
