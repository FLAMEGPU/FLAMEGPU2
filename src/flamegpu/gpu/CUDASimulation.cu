#include "flamegpu/gpu/CUDASimulation.h"

#include <curand_kernel.h>

#include <algorithm>
#include <string>

#include "flamegpu/model/AgentFunctionData.h"
#include "flamegpu/model/LayerData.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/model/SubAgentData.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/runtime/flamegpu_host_api.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"
#include "flamegpu/util/nvtx.h"
#include "flamegpu/util/compute_capability.cuh"
#include "flamegpu/util/SignalHandlers.h"
#include "flamegpu/runtime/cuRVE/curve_rtc.h"
#include "flamegpu/runtime/HostFunctionCallback.h"


std::atomic<int> CUDASimulation::active_instances;  // This value should default init to 0, specifying =0 was causing warnings on Windows.
bool CUDASimulation::AUTO_CUDA_DEVICE_RESET = true;

CUDASimulation::CUDASimulation(const ModelDescription& _model, int argc, const char** argv)
    : Simulation(_model)
    , step_count(0)
    , simulation_elapsed_time(0.f)
    , streams(std::vector<cudaStream_t>())
    , singletons(nullptr)
    , singletonsInitialised(false)
    , rtcInitialised(false)
    , rtc_kernel_cache(nullptr) {
    ++active_instances;
    initOffsetsAndMap();

    // Register the signal handler.
    SignalHandlers::registerSignalHandlers();

    // Populate the environment properties
    initEnvironmentMgr();

    // populate the CUDA agent map
    const auto &am = model->agents;
    // create new cuda agent and add to the map
    for (auto it = am.cbegin(); it != am.cend(); ++it) {
        // insert into map using value_type and store a reference to the map pair
        agent_map.emplace(it->first, std::make_unique<CUDAAgent>(*it->second, *this)).first;
    }

    // populate the CUDA message map
    const auto &mm = model->messages;
    // create new cuda message and add to the map
    for (auto it_m = mm.cbegin(); it_m != mm.cend(); ++it_m) {
        message_map.emplace(it_m->first, std::make_unique<CUDAMessage>(*it_m->second, *this));
    }

    // populate the CUDA submodel map
    const auto &smm = model->submodels;
    // create new cuda message and add to the map
    for (auto it_sm = smm.cbegin(); it_sm != smm.cend(); ++it_sm) {
        submodel_map.emplace(it_sm->first, std::unique_ptr<CUDASimulation>(new CUDASimulation(it_sm->second, this)));
    }

    if (argc && argv) {
        initialise(argc, argv);
    }
}
CUDASimulation::CUDASimulation(const std::shared_ptr<SubModelData> &submodel_desc, CUDASimulation *master_model)
    : Simulation(submodel_desc, master_model)
    , step_count(0)
    , streams(std::vector<cudaStream_t>())
    , singletons(nullptr)
    , singletonsInitialised(false)
    , rtcInitialised(false)
    , rtc_kernel_cache(nullptr)  {
    ++active_instances;
    initOffsetsAndMap();

    // Populate the environment properties
    initEnvironmentMgr();

    // populate the CUDA agent map (With SubAgents!)
    const auto &am = model->agents;
    // create new cuda agent and add to the map
    for (auto it = am.cbegin(); it != am.cend(); ++it) {
        // Locate the mapping
        auto _mapping = submodel_desc->subagents.find(it->second->name);
        if (_mapping != submodel_desc->subagents.end()) {
            // Agent is mapped, create subagent
            std::shared_ptr<SubAgentData> &mapping = _mapping->second;
            // Locate the master agent
            std::shared_ptr<AgentData> masterAgentDesc = mapping->masterAgent.lock();
            if (!masterAgentDesc) {
                THROW InvalidParent("Master agent description has expired, in CUDASimulation SubModel constructor.\n");
            }
            std::unique_ptr<CUDAAgent> &masterAgent = master_model->agent_map.at(masterAgentDesc->name);
            agent_map.emplace(it->first, std::make_unique<CUDAAgent>(*it->second, *this, masterAgent, mapping));
        } else {
            // Agent is not mapped, create regular agent
            agent_map.emplace(it->first, std::make_unique<CUDAAgent>(*it->second, *this)).first;
        }
    }  // insert into map using value_type

    // populate the CUDA message map (Sub Messages not currently supported)
    const auto &mm = model->messages;
    // create new cuda message and add to the map
    for (auto it_m = mm.cbegin(); it_m != mm.cend(); ++it_m) {
        message_map.emplace(it_m->first, std::make_unique<CUDAMessage>(*it_m->second, *this));
    }

    // populate the CUDA submodel map
    const auto &smm = model->submodels;
    // create new cuda model and add to the map
    for (auto it_sm = smm.cbegin(); it_sm != smm.cend(); ++it_sm) {
        submodel_map.emplace(it_sm->first, std::unique_ptr<CUDASimulation>(new CUDASimulation(it_sm->second, this)));
    }
    // Submodels all run silent by default
    SimulationConfig().verbose = false;
}

CUDASimulation::~CUDASimulation() {
    submodel_map.clear();  // Test
    // De-initialise, freeing singletons?
    // @todo - this is unsafe in a destructor as it may invoke cuda commands.
    if (singletonsInitialised) {
        // unique pointers cleanup by automatically
        // Drop all constants from the constant cache linked to this model
        singletons->environment.free(instance_id);

        delete singletons;
        singletons = nullptr;
    }

    // We must explicitly delete all cuda members before we cuda device reset
    agent_map.clear();
    message_map.clear();
    submodel_map.clear();
    host_api.reset();
#ifdef VISUALISATION
    visualisation.reset();
#endif
    delete rtc_kernel_cache;
    rtc_kernel_cache = nullptr;

    // If we are the last instance to destruct
    if ((!--active_instances)&& AUTO_CUDA_DEVICE_RESET) {
        gpuErrchk(cudaDeviceReset());
        EnvironmentManager::getInstance().purge();
        Curve::getInstance().purge();
    }
}

bool CUDASimulation::step() {
    NVTX_RANGE(std::string("CUDASimulation::step " + std::to_string(step_count)).c_str());

    // Ensure singletons have been initialised
    initialiseSingletons();
    // Update environment on device
    singletons->environment.updateDevice(getInstanceID());

    // If verbose, print the step number.
    if (getSimulationConfig().verbose) {
        fprintf(stdout, "Processing Simulation Step %u\n", step_count);
    }

    unsigned int nStreams = 1;
    std::string message_name;
    Curve::NamespaceHash message_name_inp_hash = 0;
    Curve::NamespaceHash message_name_outp_hash = 0;
    // hash model name
    const Curve::NamespaceHash instance_id_hash = Curve::variableRuntimeHash(instance_id);

    // TODO: simulation.getMaxFunctionsPerLayer()
    for (auto lyr = model->layers.begin(); lyr != model->layers.end(); ++lyr) {
        unsigned int temp = static_cast<unsigned int>((*lyr)->agent_functions.size());
        nStreams = std::max(nStreams, temp);
    }

    /*!  Stream creations */
    // Ensure there are enough streams to execute the layer.
    while (streams.size() < nStreams) {
        cudaStream_t stream;
        gpuErrchk(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }

    // Reset message list flags
    for (auto m =  message_map.begin(); m != message_map.end(); ++m) {
        m->second->setTruncateMessageListFlag();
    }

    /*! for each each sim layer, launch each agent function in its own stream */
    unsigned int lyr_idx = 0;
    for (auto lyr = model->layers.begin(); lyr != model->layers.end(); ++lyr) {
        NVTX_RANGE(std::string("StepLayer " + std::to_string(lyr_idx)).c_str());

        if ((*lyr)->sub_model) {
            auto &sm = submodel_map.at((*lyr)->sub_model->name);
            sm->resetStepCounter();
            sm->simulate();
            sm->reset(true);
            // Next layer, this layer cannot also contain agent functions
            continue;
        }

        const auto& functions = (*lyr)->agent_functions;

        // Track stream id
        int j = 0;
        // Sum the total number of threads being launched in the layer
        unsigned int totalThreads = 0;
        /*! for each function apply any agent function conditions*/
        {
            // Map agent memory
            for (const std::shared_ptr<AgentFunctionData> &func_des : functions) {
                if ((func_des->condition) || (!func_des->rtc_func_condition_name.empty())) {
                    auto func_agent = func_des->parent.lock();
                    NVTX_RANGE(std::string("condition map " + func_agent->name + "::" + func_des->name).c_str());
                    const CUDAAgent& cuda_agent = getCUDAAgent(func_agent->name);

                    const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
                    if (state_list_size == 0) {
                        ++j;
                        continue;
                    }
                    singletons->scatter.Scan().resize(state_list_size, CUDAScanCompaction::AGENT_DEATH, j);

                    // Configure runtime access of the functions variables within the FLAME_API object
                    cuda_agent.mapRuntimeVariables(*func_des);

                    // Zero the scan flag that will be written to
                    singletons->scatter.Scan().zero(CUDAScanCompaction::AGENT_DEATH, j);

                    totalThreads += state_list_size;
                    ++j;
                }
            }

            // Update curve
            singletons->curve.updateDevice();
            // Ensure RandomManager is the correct size to accommodate all threads to be launched
            curandState *d_rng = singletons->rng.resize(totalThreads);
            // Track stream id
            j = 0;
            // Sum the total number of threads being launched in the layer
            totalThreads = 0;
            // Launch function condition kernels
            for (const std::shared_ptr<AgentFunctionData> &func_des : functions) {
                if ((func_des->condition) || (!func_des->rtc_func_condition_name.empty())) {
                    auto func_agent = func_des->parent.lock();
                    NVTX_RANGE(std::string("condition " + func_agent->name + "::" + func_des->name).c_str());
                    if (!func_agent) {
                        THROW InvalidAgentFunc("Agent function condition refers to expired agent.");
                    }
                    std::string agent_name = func_agent->name;
                    std::string func_name = func_des->name;

                    const CUDAAgent& cuda_agent = getCUDAAgent(agent_name);

                    const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
                    if (state_list_size == 0) {
                        ++j;
                        continue;
                    }

                    int blockSize = 0;  // The launch configurator returned block size
                    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
                    int gridSize = 0;  // The actual grid size needed, based on input size

                    //  Agent function condition kernel wrapper args
                    Curve::NamespaceHash agentname_hash = Curve::variableRuntimeHash(agent_name.c_str());
                    Curve::NamespaceHash funcname_hash = Curve::variableRuntimeHash(func_name.c_str());
                    Curve::NamespaceHash agent_func_name_hash = agentname_hash + funcname_hash;
                    curandState *t_rng = d_rng + totalThreads;
                    unsigned int *scanFlag_agentDeath = this->singletons->scatter.Scan().Config(CUDAScanCompaction::Type::AGENT_DEATH, j).d_ptrs.scan_flag;
                    unsigned int sm_size = 0;
#ifndef NO_SEATBELTS
                    auto *error_buffer = this->singletons->exception.getDevicePtr(j);
                    sm_size = sizeof(error_buffer);
#endif

                    // switch between normal and RTC agent function condition
                    if (func_des->condition) {
                        // calculate the grid block size for agent function condition
                        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func_des->condition, 0, state_list_size);

                        //! Round up according to CUDAAgent state list size
                        gridSize = (state_list_size + blockSize - 1) / blockSize;
                        (func_des->condition) << <gridSize, blockSize, sm_size, streams.at(j) >> > (
#ifndef NO_SEATBELTS
                        error_buffer,
#endif
                        instance_id_hash,
                        agent_func_name_hash,
                        state_list_size,
                        t_rng,
                        scanFlag_agentDeath);
                        gpuErrchkLaunch();
                    } else {  // RTC function
                        std::string func_condition_identifier = func_name + "_condition";
                        // get instantiation
                        const jitify::KernelInstantiation& instance = cuda_agent.getRTCInstantiation(func_condition_identifier);
                        // calculate the grid block size for main agent function
                        CUfunction cu_func = (CUfunction)instance;
                        cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cu_func, 0, 0, state_list_size);
                        //! Round up according to CUDAAgent state list size
                        gridSize = (state_list_size + blockSize - 1) / blockSize;
                        // launch the kernel
                        CUresult a = instance.configure(gridSize, blockSize, sm_size, streams.at(j)).launch({
#ifndef NO_SEATBELTS
                                reinterpret_cast<void*>(&error_buffer),
#endif
                                const_cast<void*>(reinterpret_cast<const void*>(&instance_id_hash)),
                                reinterpret_cast<void*>(&agent_func_name_hash),
                                const_cast<void *>(reinterpret_cast<const void*>(&state_list_size)),
                                reinterpret_cast<void*>(&t_rng),
                                reinterpret_cast<void*>(&scanFlag_agentDeath) });
                        if (a != CUresult::CUDA_SUCCESS) {
                            const char* err_str = nullptr;
                            cuGetErrorString(a, &err_str);
                            THROW InvalidAgentFunc("There was a problem launching the runtime agent function condition '%s': %s", func_des->rtc_func_condition_name.c_str(), err_str);
                        }
                        cudaDeviceSynchronize();
                        gpuErrchkLaunch();
                    }

                    totalThreads += state_list_size;
                    ++j;
                }
            }

            // Track stream id
            j = 0;
            // Unmap agent memory, apply condition
            for (const std::shared_ptr<AgentFunctionData> &func_des : functions) {
                if ((func_des->condition) || (!func_des->rtc_func_condition_name.empty())) {
                    auto func_agent = func_des->parent.lock();
                    if (!func_agent) {
                        THROW InvalidAgentFunc("Agent function condition refers to expired agent.");
                    }
                    NVTX_RANGE(std::string("condition unmap " + func_agent->name + "::" + func_des->name).c_str());
                    CUDAAgent& cuda_agent = getCUDAAgent(func_agent->name);

                    const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
                    if (state_list_size == 0) {
                        ++j;
                        continue;
                    }

                    // unmap the function variables
                    cuda_agent.unmapRuntimeVariables(*func_des);

#ifndef NO_SEATBELTS
                    // Error check after unmap vars
                    // This means that curve is cleaned up before we throw exception (mostly prevents curve being polluted if we catch and handle errors)
                    this->singletons->exception.checkError("condition " + func_des->name, j);
#endif

                    // Process agent function condition
                    cuda_agent.processFunctionCondition(*func_des, this->singletons->scatter, j);

                    ++j;
                }
            }
        }

        j = 0;
        // Sum the total number of threads being launched in the layer
        totalThreads = 0;
        /*! for each func function - Loop through to do all mapping of agent and message variables */
        for (const std::shared_ptr<AgentFunctionData> &func_des : functions) {
            auto func_agent = func_des->parent.lock();
            if (!func_agent) {
                THROW InvalidAgentFunc("Agent function refers to expired agent.");
            }
            NVTX_RANGE(std::string("map" + func_agent->name + "::" + func_des->name).c_str());

            const CUDAAgent& cuda_agent = getCUDAAgent(func_agent->name);
            const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
            if (state_list_size == 0) {
                ++j;
                continue;
            }
            // Resize death flag array if necessary
            singletons->scatter.Scan().resize(state_list_size, CUDAScanCompaction::AGENT_DEATH, j);

            // check if a function has an input message
            if (auto im = func_des->message_input.lock()) {
                std::string inpMessage_name = im->name;
                CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name);
                // Construct PBM here if required!!
                cuda_message.buildIndex(this->singletons->scatter, j);
                // Map variables after, as index building can swap arrays
                cuda_message.mapReadRuntimeVariables(*func_des, cuda_agent);
            }

            // check if a function has an output message
            if (auto om = func_des->message_output.lock()) {
                std::string outpMessage_name = om->name;
                CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);
                // Resize message list if required
                const unsigned int existingMessages = cuda_message.getTruncateMessageListFlag() ? 0 : cuda_message.getMessageCount();
                cuda_message.resize(existingMessages + state_list_size, this->singletons->scatter, j);
                cuda_message.mapWriteRuntimeVariables(*func_des, cuda_agent, state_list_size);
                singletons->scatter.Scan().resize(state_list_size, CUDAScanCompaction::MESSAGE_OUTPUT, j);
                // Zero the scan flag that will be written to
                if (func_des->message_output_optional)
                    singletons->scatter.Scan().zero(CUDAScanCompaction::MESSAGE_OUTPUT, j);
            }

            // check if a function has an output agent
            if (auto oa = func_des->agent_output.lock()) {
                // This will act as a reserve word
                // which is added to variable hashes for agent creation on device
                CUDAAgent& output_agent = getCUDAAgent(oa->name);

                // Map vars with curve (this allocates/requests enough new buffer space if an existing version is not available/suitable)
                output_agent.mapNewRuntimeVariables(cuda_agent, *func_des, state_list_size, this->singletons->scatter, j);
            }


            /**
             * Configure runtime access of the functions variables within the FLAME_API object
             */
            cuda_agent.mapRuntimeVariables(*func_des);

            // Zero the scan flag that will be written to
            if (func_des->has_agent_death)
                singletons->scatter.Scan().CUDAScanCompaction::zero(CUDAScanCompaction::AGENT_DEATH, j);

            // Count total threads being launched
            totalThreads += cuda_agent.getStateSize(func_des->initial_state);
            ++j;
        }

        // Update curve
        singletons->curve.updateDevice();
        // Ensure RandomManager is the correct size to accommodate all threads to be launched
        curandState *d_rng = singletons->rng.resize(totalThreads);
        // Total threads is now used to provide kernel launches an offset to thread-safe thread-index
        totalThreads = 0;
        j = 0;
        //! for each func function - Loop through to launch all agent functions
        for (const std::shared_ptr<AgentFunctionData> &func_des : functions) {
            auto func_agent = func_des->parent.lock();
            if (!func_agent) {
                THROW InvalidAgentFunc("Agent function refers to expired agent.");
            }
            NVTX_RANGE(std::string(func_agent->name + "::" + func_des->name).c_str());
            const void *d_in_messagelist_metadata = nullptr;
            const void *d_out_messagelist_metadata = nullptr;
            std::string agent_name = func_agent->name;
            std::string func_name = func_des->name;

            // check if a function has an input message
            if (auto im = func_des->message_input.lock()) {
                std::string inpMessage_name = im->name;
                const CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name);

                // hash message name
                message_name_inp_hash = Curve::variableRuntimeHash(inpMessage_name.c_str());

                d_in_messagelist_metadata = cuda_message.getMetaDataDevicePtr();
            }

            // check if a function has an output message
            if (auto om = func_des->message_output.lock()) {
                std::string outpMessage_name = om->name;
                const CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);

                // hash message name
                message_name_outp_hash =  Curve::variableRuntimeHash(outpMessage_name.c_str());
                d_out_messagelist_metadata = cuda_message.getMetaDataDevicePtr();
            }

            const CUDAAgent& cuda_agent = getCUDAAgent(agent_name);

            const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
            if (state_list_size == 0) {
                ++j;
                continue;
            }

            int blockSize = 0;  // The launch configurator returned block size
            int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
            int gridSize = 0;  // The actual grid size needed, based on input size

            // Agent function kernel wrapper args
            Curve::NamespaceHash agentname_hash = Curve::variableRuntimeHash(agent_name.c_str());
            Curve::NamespaceHash funcname_hash = Curve::variableRuntimeHash(func_name.c_str());
            Curve::NamespaceHash agent_func_name_hash = agentname_hash + funcname_hash;
            Curve::NamespaceHash agentoutput_hash = func_des->agent_output.lock() ? singletons->curve.variableRuntimeHash("_agent_birth") + funcname_hash : 0;
            curandState * t_rng = d_rng + totalThreads;
            unsigned int *scanFlag_agentDeath = func_des->has_agent_death ? this->singletons->scatter.Scan().Config(CUDAScanCompaction::Type::AGENT_DEATH, j).d_ptrs.scan_flag : nullptr;
            unsigned int *scanFlag_messageOutput = this->singletons->scatter.Scan().Config(CUDAScanCompaction::Type::MESSAGE_OUTPUT, j).d_ptrs.scan_flag;
            unsigned int *scanFlag_agentOutput = this->singletons->scatter.Scan().Config(CUDAScanCompaction::Type::AGENT_OUTPUT, j).d_ptrs.scan_flag;
            unsigned int sm_size = 0;
#ifndef NO_SEATBELTS
            auto *error_buffer = this->singletons->exception.getDevicePtr(j);
            sm_size = sizeof(error_buffer);
#endif

            if (func_des->func) {   // compile time specified agent function launch
                // calculate the grid block size for main agent function
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func_des->func, 0, state_list_size);
                //! Round up according to CUDAAgent state list size
                gridSize = (state_list_size + blockSize - 1) / blockSize;

                (func_des->func) << <gridSize, blockSize, sm_size, streams.at(j) >> > (
#ifndef NO_SEATBELTS
                    error_buffer,
#endif
                    instance_id_hash,
                    agent_func_name_hash,
                    message_name_inp_hash,
                    message_name_outp_hash,
                    agentoutput_hash,
                    state_list_size,
                    d_in_messagelist_metadata,
                    d_out_messagelist_metadata,
                    t_rng,
                    scanFlag_agentDeath,
                    scanFlag_messageOutput,
                    scanFlag_agentOutput);
                gpuErrchkLaunch();
            } else {      // assume this is a runtime specified agent function
                // get instantiation
                const jitify::KernelInstantiation&  instance = cuda_agent.getRTCInstantiation(func_name);
                // calculate the grid block size for main agent function
                CUfunction cu_func = (CUfunction)instance;
                cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cu_func, 0, 0, state_list_size);
                //! Round up according to CUDAAgent state list size
                gridSize = (state_list_size + blockSize - 1) / blockSize;
                // launch the kernel
                CUresult a = instance.configure(gridSize, blockSize, sm_size, streams.at(j)).launch({
#ifndef NO_SEATBELTS
                        reinterpret_cast<void*>(&error_buffer),
#endif
                        const_cast<void*>(reinterpret_cast<const void*>(&instance_id_hash)),
                        reinterpret_cast<void*>(&agent_func_name_hash),
                        reinterpret_cast<void*>(&message_name_inp_hash),
                        reinterpret_cast<void*>(&message_name_outp_hash),
                        reinterpret_cast<void*>(&agentoutput_hash),
                        const_cast<void*>(reinterpret_cast<const void*>(&state_list_size)),
                        const_cast<void*>(reinterpret_cast<const void*>(&d_in_messagelist_metadata)),
                        const_cast<void*>(reinterpret_cast<const void*>(&d_out_messagelist_metadata)),
                        const_cast<void*>(reinterpret_cast<const void*>(&t_rng)),
                        reinterpret_cast<void*>(&scanFlag_agentDeath),
                        reinterpret_cast<void*>(&scanFlag_messageOutput),
                        reinterpret_cast<void*>(&scanFlag_agentOutput)});
                if (a != CUresult::CUDA_SUCCESS) {
                    const char* err_str = nullptr;
                    cuGetErrorString(a, &err_str);
                    THROW InvalidAgentFunc("There was a problem launching the runtime agent function '%s': %s", func_name.c_str(), err_str);
                }
                cudaDeviceSynchronize();
                gpuErrchkLaunch();
            }

            totalThreads += state_list_size;
            ++j;
            ++lyr_idx;
        }

        j = 0;
        // for each func function - Loop through to un-map all agent and message variables
        for (const std::shared_ptr<AgentFunctionData> &func_des : functions) {
            auto func_agent = func_des->parent.lock();
            if (!func_agent) {
                THROW InvalidAgentFunc("Agent function refers to expired agent.");
            }
            NVTX_RANGE(std::string("unmap" + func_agent->name + "::" + func_des->name).c_str());
            CUDAAgent& cuda_agent = getCUDAAgent(func_agent->name);

            const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
            // If agent function wasn't executed, these are redundant
            if (state_list_size > 0) {
                // check if a function has an input message
                if (auto im = func_des->message_input.lock()) {
                    std::string inpMessage_name = im->name;
                    const CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name);
                    cuda_message.unmapRuntimeVariables(*func_des);
                }

                // check if a function has an output message
                if (auto om = func_des->message_output.lock()) {
                    std::string outpMessage_name = om->name;
                    CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);
                    cuda_message.unmapRuntimeVariables(*func_des);
                    cuda_message.swap(func_des->message_output_optional, state_list_size, this->singletons->scatter, j);
                    cuda_message.clearTruncateMessageListFlag();
                    cuda_message.setPBMConstructionRequiredFlag();
                }

                // Process agent death (has agent death check is handled by the method)
                // This MUST occur before agent_output, as if agent_output triggers resize then scan_flag for death will be purged
                cuda_agent.processDeath(*func_des, this->singletons->scatter, j);

                // Process agent state transition (Longer term merge this with process death?)
                cuda_agent.transitionState(func_des->initial_state, func_des->end_state, this->singletons->scatter, j);
            }

            // Process agent function condition
            cuda_agent.clearFunctionCondition(func_des->initial_state);

            // If agent function wasn't executed, these are redundant
            if (state_list_size > 0) {
                // check if a function has an output agent
                if (auto oa = func_des->agent_output.lock()) {
                    // This will act as a reserve word
                    // which is added to variable hashes for agent creation on device
                    CUDAAgent& output_agent = getCUDAAgent(oa->name);
                    // Scatter the agent birth
                    output_agent.scatterNew(*func_des, state_list_size, this->singletons->scatter, j);  // This must be passed the state list size prior to death
                    // unmap vars with curve
                    output_agent.unmapNewRuntimeVariables(*func_des);
                }

                // unmap the function variables
                cuda_agent.unmapRuntimeVariables(*func_des);
#ifndef NO_SEATBELTS
                // Error check after unmap vars
                // This means that curve is cleaned up before we throw exception (mostly prevents curve being polluted if we catch and handle errors)
                this->singletons->exception.checkError(func_des->name, j);
#endif
            }

            ++j;
        }

        NVTX_PUSH("CUDASimulation::step::HostFunctions");
        // Execute all host functions attached to layer
        // TODO: Concurrency?
        assert(host_api);
        for (auto &stepFn : (*lyr)->host_functions) {
            NVTX_RANGE("hostFunc");
            stepFn(this->host_api.get());
        }
        // Execute all host function callbacks attached to layer
        for (auto &stepFn : (*lyr)->host_functions_callbacks) {
            NVTX_RANGE("hostFunc_swig");
            stepFn->run(this->host_api.get());
        }
        // If we have host layer functions, we might have host agent creation
        if ((*lyr)->host_functions.size() || ((*lyr)->host_functions_callbacks.size()))
            processHostAgentCreation(j);
        // Update environment on device
        singletons->environment.updateDevice(getInstanceID());
        NVTX_POP();

        // cudaDeviceSynchronize();
    }

    NVTX_PUSH("CUDASimulation::step::StepFunctions");
    // Execute step functions
    for (auto &stepFn : model->stepFunctions) {
        NVTX_RANGE("stepFunc");
        stepFn(this->host_api.get());
    }
    // Execute step function callbacks
    for (auto &stepFn : model->stepFunctionCallbacks) {
        NVTX_RANGE("stepFunc_swig");
        stepFn->run(this->host_api.get());
    }
    // If we have step functions, we might have host agent creation
    if (model->stepFunctions.size() || model->stepFunctionCallbacks.size())
        processHostAgentCreation(0);
    NVTX_POP();

    // Execute exit conditions
    for (auto &exitCdns : model->exitConditions)
        if (exitCdns(this->host_api.get()) == EXIT) {
#ifdef VISUALISATION
            if (visualisation) {
                visualisation->updateBuffers(step_count+1);
            }
#endif
            // If there were any exit conditions, we also need to update the step count
            incrementStepCounter();
            return false;
        }
    // Execute exit condition callbacks
    for (auto &exitCdns : model->exitConditionCallbacks)
        if (exitCdns->run(this->host_api.get()) == EXIT) {
#ifdef VISUALISATION
            if (visualisation) {
                visualisation->updateBuffers(step_count+1);
            }
#endif
            // If there were any exit conditions, we also need to update the step count
            incrementStepCounter();
            return false;
        }
    // If we have exit conditions functions, we might have host agent creation
    if (model->exitConditions.size() || model->exitConditionCallbacks.size())
        processHostAgentCreation(0);

#ifdef VISUALISATION
        if (visualisation) {
            visualisation->updateBuffers(step_count+1);
        }
#endif
    // Update step count at the end of the step - when it has completed.
    incrementStepCounter();
    return true;
}

void CUDASimulation::simulate() {
    if (agent_map.size() == 0) {
        THROW InvalidCudaAgentMapSize("Simulation has no agents, in CUDASimulation::simulate().");  // recheck if this is really required
    }

    // Ensure singletons have been initialised
    initialiseSingletons();

    // Reinitialise any unmapped agent variables
    if (submodel) {
        int j = 0;
        for (auto &a : agent_map) {
            a.second->initUnmappedVars(this->singletons->scatter, j++);
        }
    }

    // create cude events to record the elapsed time for simulate.
    cudaEvent_t simulateStartEvent = nullptr;
    cudaEvent_t simulateEndEvent = nullptr;
    gpuErrchk(cudaEventCreate(&simulateStartEvent));
    gpuErrchk(cudaEventCreate(&simulateEndEvent));
    // Record the start event.
    gpuErrchk(cudaEventRecord(simulateStartEvent));
    // Reset the elapsed time.
    simulation_elapsed_time = 0.f;

    // CUDAAgentMap::iterator it;

    // check any CUDAAgents with population size == 0
    // if they have executable functions then these can be ignored
    // if they have agent creations then buffer space must be allocated for them

    // Execute init functions
    NVTX_PUSH("CUDASimulation::step::InitFunctions");
    for (auto &initFn : model->initFunctions) {
        NVTX_RANGE("initFunc");
        initFn(this->host_api.get());
    }
    // Execute init function callbacks
    for (auto &initFn : model->initFunctionCallbacks) {
        NVTX_RANGE("initFunc_swig");
        initFn->run(this->host_api.get());
    }
    // Check if host agent creation was used in init functions
    if (model->initFunctions.size() || model->initFunctionCallbacks.size())
        processHostAgentCreation(0);
    // Update environment on device
    singletons->environment.updateDevice(getInstanceID());
    NVTX_POP();

#ifdef VISUALISATION
    if (visualisation) {
        visualisation->updateBuffers();
    }
#endif

    for (unsigned int i = 0; getSimulationConfig().steps == 0 ? true : i < getSimulationConfig().steps; i++) {
        // std::cout <<"step: " << i << std::endl;
        if (!step())
            break;
#ifdef VISUALISATION
        // Special case, if steps == 0 and visualisation has been closed
        if (getSimulationConfig().steps == 0 &&
            visualisation && !visualisation->isRunning()) {
            visualisation->join();  // Vis exists in separate thread, make sure it has actually exited
            break;
        }
#endif
    }

    // Execute exit functions
    for (auto &exitFn : model->exitFunctions)
        exitFn(this->host_api.get());
    for (auto &exitFn : model->exitFunctionCallbacks)
        exitFn->run(this->host_api.get());

#ifdef VISUALISATION
    if (visualisation) {
        visualisation->updateBuffers();
    }
#endif

    // Record and store the elapsed time
    // if --timing was passed, output to stdout.
    gpuErrchk(cudaEventRecord(simulateEndEvent));
    // Syncrhonize the stop event
    gpuErrchk(cudaEventSynchronize(simulateEndEvent));
    gpuErrchk(cudaEventElapsedTime(&simulation_elapsed_time, simulateStartEvent, simulateEndEvent));

    gpuErrchk(cudaEventDestroy(simulateStartEvent));
    gpuErrchk(cudaEventDestroy(simulateEndEvent));
    simulateStartEvent = nullptr;
    simulateEndEvent = nullptr;

    if (getSimulationConfig().timing) {
        // Record the end event.
        // Resolution is 0.5 microseconds, so print to 1 us.
        fprintf(stdout, "Total Processing time: %.3f ms\n", simulation_elapsed_time);
    }

    // Destroy streams.
    for (auto stream : streams) {
        gpuErrchk(cudaStreamDestroy(stream));
    }
    streams.clear();
}

void CUDASimulation::reset(bool submodelReset) {
    // Reset step counter
    resetStepCounter();

    if (singletonsInitialised) {
        // Reset environment properties
        singletons->environment.resetModel(instance_id, *model->environment);

        // Reseed random, unless performing submodel reset
        if (!submodelReset) {
            singletons->rng.reseed(getSimulationConfig().random_seed);
        }
    }

    // Cull agents
    if (submodel) {
        // Submodels only want to reset unmapped states, otherwise they will break parent model
        for (auto &a : agent_map) {
            a.second->cullUnmappedStates();
        }
    } else {
        for (auto &a : agent_map) {
            a.second->cullAllStates();
        }
    }

    // Cull messagelists
    for (auto &a : message_map) {
        a.second->setMessageCount(0);
        a.second->setTruncateMessageListFlag();
    }


    // Trigger reset in all submodels, propagation is not necessary when performing submodel reset
    if (!submodelReset) {
        for (auto &s : submodel_map) {
            s.second->reset(false);
        }
    }
}

void CUDASimulation::setPopulationData(AgentPopulation& population) {
    // Ensure singletons have been initialised
    initialiseSingletons();

    CUDAAgentMap::iterator it;
    it = agent_map.find(population.getAgentName());

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("Error: Agent ('%s') was not found, "
            "in CUDASimulation::setPopulationData()",
            population.getAgentName().c_str());
    }

    /*! create agent state lists */
    it->second->setPopulationData(population, this->singletons->scatter, 0);  // Streamid shouldn't matter here

#ifdef VISUALISATION
    if (visualisation) {
        visualisation->updateBuffers();
    }
#endif
}

void CUDASimulation::getPopulationData(AgentPopulation& population) {
    // Ensure singletons have been initialised
    initialiseSingletons();

    CUDAAgentMap::iterator it;
    it = agent_map.find(population.getAgentName());

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("Error: Agent ('%s') was not found, "
            "in CUDASimulation::getPopulationData()",
            population.getAgentName().c_str());
    }

    /*!create agent state lists */
    it->second->getPopulationData(population);
}

CUDAAgent& CUDASimulation::getCUDAAgent(const std::string& agent_name) const {
    CUDAAgentMap::const_iterator it;
    it = agent_map.find(agent_name);

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("CUDA agent ('%s') not found, in CUDASimulation::getCUDAAgent().",
            agent_name.c_str());
    }

    return *(it->second);
}

AgentInterface& CUDASimulation::getAgent(const std::string& agent_name) {
    // Ensure singletons have been initialised
    initialiseSingletons();

    auto it = agent_map.find(agent_name);

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("CUDA agent ('%s') not found, in CUDASimulation::getAgent().",
            agent_name.c_str());
    }

    return *(it->second);
}

CUDAMessage& CUDASimulation::getCUDAMessage(const std::string& message_name) const {
    CUDAMessageMap::const_iterator it;
    it = message_map.find(message_name);

    if (it == message_map.end()) {
        THROW InvalidCudaMessage("CUDA message ('%s') not found, in CUDASimulation::getCUDAMessage().",
            message_name.c_str());
    }

    return *(it->second);
}

bool CUDASimulation::checkArgs_derived(int argc, const char** argv, int &i) {
    // Get arg as lowercase
    std::string arg(argv[i]);
    std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) { return std::use_facet< std::ctype<char>>(std::locale()).tolower(c); });
    // -device <uint>, Uses the specified cuda device, defaults to 0
    if ((arg.compare("--device") == 0 || arg.compare("-d") == 0) && argc > i+1) {
        config.device_id = static_cast<unsigned int>(strtoul(argv[++i], nullptr, 0));
        return true;
    }
    return false;
}

void CUDASimulation::printHelp_derived() {
    const char *line_fmt = "%-18s %s\n";
    printf("CUDA Model Optional Arguments:\n");
    printf(line_fmt, "-d, --device", "GPU index");
}

void CUDASimulation::applyConfig_derived() {
    NVTX_RANGE("applyConfig_derived");

    // Handle console_mode
#ifdef VISUALISATION
    if (getSimulationConfig().console_mode) {
        if (visualisation) {
            visualisation->deactivate();
        }
    }
#endif


    cudaError_t cudaStatus;
    int device_count;

    // default device
    cudaStatus = cudaGetDeviceCount(&device_count);

    if (cudaStatus != cudaSuccess) {
        THROW InvalidCUDAdevice("Error finding CUDA devices!  Do you have a CUDA-capable GPU installed?");
    }
    if (device_count == 0) {
        THROW InvalidCUDAdevice("Error no CUDA devices found!");
    }

    // Select device
    if (config.device_id >= device_count) {
        THROW InvalidCUDAdevice("Error setting CUDA device to '%d', only %d available!", config.device_id, device_count);
    }

    // Check the compute capability of the device, throw an exception if not valid for the executable.
    if (!util::compute_capability::checkComputeCapability(static_cast<int>(config.device_id))) {
        int min_cc = util::compute_capability::minimumCompiledComputeCapability();
        int cc = util::compute_capability::getComputeCapability(static_cast<int>(config.device_id));
        THROW InvalidCUDAComputeCapability("Error application compiled for CUDA Compute Capability %d and above. Device %u is compute capability %d. Rebuild for SM_%d.", min_cc, config.device_id, cc, cc);
    }

    cudaStatus = cudaSetDevice(static_cast<int>(config.device_id));
    if (cudaStatus != cudaSuccess) {
        THROW InvalidCUDAdevice("Unknown error setting CUDA device to '%d'. (%d available)", config.device_id, device_count);
    }
    // Call cudaFree to initialise the context early
    gpuErrchk(cudaFree(nullptr));

    // Apply changes to submodels
    for (auto &sm : submodel_map) {
        // We're not actually going to use this value, but it might be useful there later
        // Calling apply config a second time would reinit GPU, which might clear existing gpu allocations etc
        sm.second->CUDAConfig().device_id = config.device_id;
    }

    // Initialise singletons once a device has been selected.
    // @todo - if this has already been called, before the device was selected an error should occur.
    initialiseSingletons();

    // We init Random through submodel hierarchy after singletons
    reseed(getSimulationConfig().random_seed);
}

void CUDASimulation::reseed(const unsigned int &seed) {
    SimulationConfig().random_seed = seed;
    singletons->rng.reseed(seed);

    // Propagate to submodels
    int i = 7;
    for (auto &sm : submodel_map) {
        // Pass random seed on to submodels
        sm.second->singletons->rng.reseed(getSimulationConfig().random_seed * i * 23);
        // Mutate seed
        i *= 13;
    }
}
/**
 * These values are ony used by CUDASimulation::initialiseSingletons()
 * Can't put a __device__ symbol method static
 */
namespace {
    __device__ unsigned int DEVICE_HAS_RESET = 0xDEADBEEF;
    const unsigned int DEVICE_HAS_RESET_FLAG = 0xDEADBEEF;
}  // namespace

void CUDASimulation::initialiseSingletons() {
    // Only do this once.
    if (!singletonsInitialised) {
        // If the device has not been specified, also check the compute capability is OK
        // Check the compute capability of the device, throw an exception if not valid for the executable.
        if (!util::compute_capability::checkComputeCapability(static_cast<int>(config.device_id))) {
            int min_cc = util::compute_capability::minimumCompiledComputeCapability();
            int cc = util::compute_capability::getComputeCapability(static_cast<int>(config.device_id));
            THROW InvalidCUDAComputeCapability("Error application compiled for CUDA Compute Capability %d and above. Device %u is compute capability %d. Rebuild for SM_%d.", min_cc, config.device_id, cc, cc);
        }
        // Check if device has been reset
        unsigned int DEVICE_HAS_RESET_CHECK = 0;
        cudaMemcpyFromSymbol(&DEVICE_HAS_RESET_CHECK, DEVICE_HAS_RESET, sizeof(unsigned int));
        if (DEVICE_HAS_RESET_CHECK == DEVICE_HAS_RESET_FLAG) {
            // Device has been reset, purge host mirrors of static objects/singletons
            Curve::getInstance().purge();
            if (singletons) {
                singletons->rng.purge();
                singletons->scatter.purge();
            }
            EnvironmentManager::getInstance().purge();
            // Reset flag
            DEVICE_HAS_RESET_CHECK = 0;  // Any value that doesnt match DEVICE_HAS_RESET_FLAG
            cudaMemcpyToSymbol(DEVICE_HAS_RESET, &DEVICE_HAS_RESET_CHECK, sizeof(unsigned int));
        }
        // Get references to all required singleton and store in the instance.
        singletons = new Singletons(
            Curve::getInstance(),
            EnvironmentManager::getInstance());

        // Reinitialise random for this simulation instance
        singletons->rng.reseed(getSimulationConfig().random_seed);

        // Pass created RandomManager to host api
        host_api = std::make_unique<FLAMEGPU_HOST_API>(*this, singletons->rng, agentOffsets, agentData);

        for (auto &cm : message_map) {
            cm.second->init(singletons->scatter, 0);
        }

        // Propagate singleton init to submodels
        for (auto &sm : submodel_map) {
            sm.second->initialiseSingletons();
        }

        singletonsInitialised = true;
    }

    // Ensure RTC is set up.
    initialiseRTC();

    // Update environment on device
    singletons->environment.updateDevice(getInstanceID());
}

void CUDASimulation::initialiseRTC() {
    // Only do this once.
    if (!rtcInitialised) {
        // Create jitify cache
        if (!rtc_kernel_cache) {
            rtc_kernel_cache = new jitify::JitCache();
        }
        // Build any RTC functions
        const auto& am = model->agents;
        // iterate agents and then agent functions to find any rtc functions or function conditions
        for (auto it = am.cbegin(); it != am.cend(); ++it) {
            auto a_it = agent_map.find(it->first);
            const auto& mf = it->second->functions;
            for (auto it_f = mf.cbegin(); it_f != mf.cend(); ++it_f) {
                // check rtc source to see if this is a RTC function
                if (!it_f->second->rtc_source.empty()) {
                    // create CUDA agent RTC function by calling addInstantitateRTCFunction on CUDAAgent with AgentFunctionData
                    a_it->second->addInstantitateRTCFunction(*rtc_kernel_cache, *it_f->second);
                }
                // check rtc source to see if the function condition is an rtc condition
                if (!it_f->second->rtc_condition_source.empty()) {
                    // create CUDA agent RTC function condition by calling addInstantitateRTCFunction on CUDAAgent with AgentFunctionData
                    a_it->second->addInstantitateRTCFunction(*rtc_kernel_cache, *it_f->second, true);
                }
            }
        }

        // Initialise device environment for RTC
        singletons->environment.initRTC(*this);

        rtcInitialised = true;
    }
}

void CUDASimulation::resetDerivedConfig() {
    this->config = CUDASimulation::Config();
    resetStepCounter();
}


CUDASimulation::Config &CUDASimulation::CUDAConfig() {
    return config;
}
const CUDASimulation::Config &CUDASimulation::getCUDAConfig() const {
    return config;
}
#ifdef VISUALISATION
ModelVis &CUDASimulation::getVisualisation() {
    if (!visualisation)
        visualisation = std::make_unique<ModelVis>(*this);
    return *visualisation.get();
}
#endif

unsigned int CUDASimulation::getStepCounter() {
    return step_count;
}
void CUDASimulation::resetStepCounter() {
    step_count = 0;
}

void CUDASimulation::initOffsetsAndMap() {
    const auto &md = getModelDescription();
    // Build offsets
    agentOffsets.clear();
    for (const auto &agent : md.agents) {
        agentOffsets.emplace(agent.first, VarOffsetStruct(agent.second->variables));
    }
    // Build data
    agentData.clear();
    for (const auto &agent : md.agents) {
        AgentDataBufferStateMap agent_states;
        for (const auto&state : agent.second->states)
            agent_states.emplace(state, AgentDataBuffer());
        agentData.emplace(agent.first, agent_states);
    }
}

void CUDASimulation::processHostAgentCreation(const unsigned int &streamId) {
    size_t t_bufflen = 0;
    char *t_buff = nullptr;
    char *dt_buff = nullptr;
    // For each agent type
    for (auto &agent : agentData) {
        // We need size of agent
        const VarOffsetStruct &offsets = agentOffsets.at(agent.first);
        // For each state within the agent
        for (auto &state : agent.second) {
            // If the buffer has data
            if (state.second.size()) {
                size_t size_req = offsets.totalSize * state.second.size();
                {  // Ensure we have enough temp memory
                    if (size_req > t_bufflen) {
                        if (t_buff) {
                            free(t_buff);
                            gpuErrchk(cudaFree(dt_buff));
                        }
                        t_buff = reinterpret_cast<char*>(malloc(size_req));
                        gpuErrchk(cudaMalloc(&dt_buff, size_req));
                        t_bufflen = size_req;
                    }
                }
                // Copy buffer memory into a single block
                for (unsigned int i = 0; i < state.second.size(); ++i) {
                    memcpy(t_buff + (i*offsets.totalSize), state.second[i].data, offsets.totalSize);
                }
                // Copy t_buff to device
                gpuErrchk(cudaMemcpy(dt_buff, t_buff, size_req, cudaMemcpyHostToDevice));
                // Scatter to device
                auto &cudaagent = agent_map.at(agent.first);
                cudaagent->scatterHostCreation(state.first, static_cast<unsigned int>(state.second.size()), dt_buff, offsets, this->singletons->scatter, streamId);
                // Clear buffer
                state.second.clear();
            }
        }
    }
    // Release temp memory
    if (t_buff) {
        free(t_buff);
        gpuErrchk(cudaFree(dt_buff));
    }
}

void CUDASimulation::RTCSafeCudaMemcpyToSymbol(const void* symbol, const char* rtc_symbol_name, const void* src, size_t count, size_t offset) const {
    // make the mem copy to runtime API symbol
    gpuErrchk(cudaMemcpyToSymbol(symbol, src, count, offset));
    // loop through agents
    for (const auto& agent_pair : agent_map) {
        // loop through any agent functions
        for (const CUDAAgent::CUDARTCFuncMapPair& rtc_func_pair : agent_pair.second->getRTCFunctions()) {
            CUdeviceptr rtc_dev_ptr = 0;
            // get the RTC device symbol
            rtc_dev_ptr = rtc_func_pair.second->get_global_ptr(rtc_symbol_name);
            // make the memcpy to the rtc version of the symbol
            gpuErrchkDriverAPI(cuMemcpyHtoD(rtc_dev_ptr + offset, src, count));
        }
    }
}

void CUDASimulation::RTCSafeCudaMemcpyToSymbolAddress(void* ptr, const char* rtc_symbol_name, const void* src, size_t count, size_t offset) const {
    // offset the device pointer by casting to char
    void* offset_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(ptr) + offset);
    // make the mem copy to runtime API symbol
    gpuErrchk(cudaMemcpy(offset_ptr, src, count, cudaMemcpyHostToDevice));
    // loop through agents
    for (const auto& agent_pair : agent_map) {
        // loop through any agent functions
        for (const CUDAAgent::CUDARTCFuncMapPair& rtc_func_pair : agent_pair.second->getRTCFunctions()) {
            CUdeviceptr rtc_dev_ptr = 0;
            // get the RTC device symbol
            rtc_dev_ptr = rtc_func_pair.second->get_global_ptr(rtc_symbol_name);
            // make the memcpy to the rtc version of the symbol
            gpuErrchkDriverAPI(cuMemcpyHtoD(rtc_dev_ptr + offset, src, count));
        }
    }
}

void CUDASimulation::RTCUpdateEnvironmentVariables(const void* src, size_t count) const {
    // loop through agents
    for (const auto& agent_pair : agent_map) {
        // loop through any agent functions
        for (const CUDAAgent::CUDARTCFuncMapPair& rtc_func_pair : agent_pair.second->getRTCFunctions()) {
            CUdeviceptr rtc_dev_ptr = 0;
            // get the RTC device symbol
            std::string rtc_symbol_name = CurveRTCHost::getEnvVariableSymbolName();
            rtc_dev_ptr = rtc_func_pair.second->get_global_ptr(rtc_symbol_name.c_str());
            // make the memcpy to the rtc version of the symbol
            gpuErrchkDriverAPI(cuMemcpyHtoD(rtc_dev_ptr, src, count));
        }
    }
}

void CUDASimulation::incrementStepCounter() {
    this->step_count++;
    this->singletons->environment.set({instance_id, "_stepCount"}, this->step_count);
}

float CUDASimulation::getSimulationElapsedTime() const {
    // Get the value
    return this->simulation_elapsed_time;
}

void CUDASimulation::initEnvironmentMgr() {
    // Populate the environment properties
    if (!submodel) {
        EnvironmentManager::getInstance().init(instance_id, *model->environment);
    } else {
        EnvironmentManager::getInstance().init(instance_id, *model->environment, mastermodel->getInstanceID(), *submodel->subenvironment);
    }

    // Add the CUDASimulation specific variables(s)
    EnvironmentManager::getInstance().add({instance_id, "_stepCount"}, 0u, false);
}
