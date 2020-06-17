#include "flamegpu/gpu/CUDAAgentModel.h"

#include <algorithm>
#include <string>

#include "flamegpu/model/AgentFunctionData.h"
#include "flamegpu/model/LayerData.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/runtime/flamegpu_host_api.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"
#include "flamegpu/util/nvtx.h"
#include "flamegpu/util/compute_capability.cuh"
#include "flamegpu/util/SignalHandlers.h"
#include "flamegpu/runtime/cuRVE/curve_rtc.h"

CUDAAgentModel::CUDAAgentModel(const ModelDescription& _model)
    : Simulation(_model)
    , step_count(0)
    , simulation_elapsed_time(0.f)
    , agent_map()
    , message_map()
    , streams(std::vector<cudaStream_t>())
    , singletons(nullptr)
    , singletonsInitialised(false)
    , rtcInitialised(false)
    , host_api(std::make_unique<FLAMEGPU_HOST_API>(*this, agentOffsets, agentData)) {
    initOffsetsAndMap();

    // Register the signal handler.
    SignalHandlers::registerSignalHandlers();
    // populate the CUDA agent map
    const auto &am = model->agents;
    // create new cuda agent and add to the map
    for (auto it = am.cbegin(); it != am.cend(); ++it) {
        // insert into map using value_type and store a referecne to the map pair
        agent_map.emplace(it->first, std::make_unique<CUDAAgent>(*it->second, *this)).first;
    }

    // populate the CUDA message map
    const auto &mm = model->messages;
    // create new cuda message and add to the map
    for (auto it_m = mm.cbegin(); it_m != mm.cend(); ++it_m) {
        message_map.emplace(it_m->first, std::make_unique<CUDAMessage>(*it_m->second, *this));
    }
}

CUDAAgentModel::~CUDAAgentModel() {
    // De-initialise, freeing singletons?
    // @todo - this is unsafe in a destrcutor as it may invoke cuda commands.
    if (singletonsInitialised) {
        singletons->rng.decreaseSimCounter();
        singletons->scatter.decreaseSimCounter();
        // unique pointers cleanup by automatically
        // Drop all constants from the constant cache linked to this model
        singletons->environment.free(model->name);

        delete singletons;
        singletons = nullptr;
    }
}

bool CUDAAgentModel::step() {
    NVTX_RANGE(std::string("CUDAAgentModel::step " + std::to_string(step_count)).c_str());

    // Ensure singletons have been initialised
    initialiseSingletons();

    // If verbose, print the step number.
    if (getSimulationConfig().verbose) {
        fprintf(stdout, "Processing Simulation Step %u\n", step_count);
    }

    unsigned int nStreams = 1;
    std::string message_name;
    Curve::NamespaceHash message_name_inp_hash = 0;
    Curve::NamespaceHash message_name_outp_hash = 0;
    // hash model name
    const Curve::NamespaceHash modelname_hash = singletons->curve.variableRuntimeHash(model->name.c_str());

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
                    flamegpu_internal::CUDAScanCompaction::resize(cuda_agent.getStateSize(func_des->initial_state), flamegpu_internal::CUDAScanCompaction::AGENT_DEATH, j, *this);

                    // Configure runtime access of the functions variables within the FLAME_API object
                    cuda_agent.mapRuntimeVariables(*func_des, func_des->initial_state);

                    // Zero the scan flag that will be written to
                    flamegpu_internal::CUDAScanCompaction::zero(flamegpu_internal::CUDAScanCompaction::AGENT_DEATH, j);

                    totalThreads += cuda_agent.getStateSize(func_des->initial_state);
                }
            }

            // Ensure RandomManager is the correct size to accomodate all threads to be launched
            singletons->rng.resize(totalThreads, *this);
            // Track stream id
            j = 0;
            // Sum the total number of threads being launched in the layer
            totalThreads = 0;
            // Launch kernel
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

                    int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
                    if (state_list_size == 0)
                        continue;

                    int blockSize = 0;  // The launch configurator returned block size
                    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
                    int gridSize = 0;  // The actual grid size needed, based on input size

                    // hash agent name
                    Curve::NamespaceHash agentname_hash = singletons->curve.variableRuntimeHash(agent_name.c_str());
                    // hash function name
                    Curve::NamespaceHash funcname_hash = singletons->curve.variableRuntimeHash(func_name.c_str());
                    // agent function name hash
                    Curve::NamespaceHash agent_func_name_hash = agentname_hash + funcname_hash;

                    // switch between normal and RTC agent function condition
                    if (func_des->condition) {
                        // calculate the grid block size for agent function condition
                        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func_des->condition, 0, state_list_size);

                        //! Round up according to CUDAAgent state list size
                        gridSize = (state_list_size + blockSize - 1) / blockSize;

                        (func_des->condition) << <gridSize, blockSize, 0, streams.at(j) >> > (modelname_hash, agentname_hash + funcname_hash, state_list_size, totalThreads, j);
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
                        CUresult a = instance.configure(gridSize, blockSize).launch({
                                const_cast<void*>(reinterpret_cast<const void*>(&modelname_hash)),
                                reinterpret_cast<void*>(&agent_func_name_hash),
                                reinterpret_cast<void*>(&state_list_size),
                                reinterpret_cast<void*>(&totalThreads),
                                reinterpret_cast<void*>(&j) });
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

                    // unmap the function variables
                    cuda_agent.unmapRuntimeVariables(*func_des);

                    // Process agent function condition
                    cuda_agent.processFunctionCondition(*func_des, j);

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
            const unsigned int STATE_SIZE = cuda_agent.getStateSize(func_des->initial_state);
            flamegpu_internal::CUDAScanCompaction::resize(STATE_SIZE, flamegpu_internal::CUDAScanCompaction::AGENT_DEATH, j, *this);

            // check if a function has an input message
            if (auto im = func_des->message_input.lock()) {
                std::string inpMessage_name = im->name;
                CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name);
                // Construct PBM here if required!!
                cuda_message.buildIndex();
                // Map variables after, as index building can swap arrays
                cuda_message.mapReadRuntimeVariables(*func_des, cuda_agent);
            }

            // check if a function has an output message
            if (auto om = func_des->message_output.lock()) {
                std::string outpMessage_name = om->name;
                CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);
                // Resize message list if required
                const unsigned int existingMessages = cuda_message.getTruncateMessageListFlag() ? 0 : cuda_message.getMessageCount();
                cuda_message.resize(existingMessages + STATE_SIZE, j);
                cuda_message.mapWriteRuntimeVariables(*func_des, cuda_agent, STATE_SIZE);
                flamegpu_internal::CUDAScanCompaction::resize(STATE_SIZE, flamegpu_internal::CUDAScanCompaction::MESSAGE_OUTPUT, j, *this);
                // Zero the scan flag that will be written to
                if (func_des->message_output_optional)
                    flamegpu_internal::CUDAScanCompaction::zero(flamegpu_internal::CUDAScanCompaction::MESSAGE_OUTPUT, j);
            }

            // check if a function has an output agent
            if (auto oa = func_des->agent_output.lock()) {
                // This will act as a reserve word
                // which is added to variable hashes for agent creation on device
                CUDAAgent& output_agent = getCUDAAgent(oa->name);
                // Ensure we have enough memory (this resizes the scan flag too)
                output_agent.resizeNew(*func_des, STATE_SIZE, j);
                // Ensure the scan flag is zeroed
                flamegpu_internal::CUDAScanCompaction::zero(flamegpu_internal::CUDAScanCompaction::AGENT_OUTPUT, j);
                // Map vars with curve
                output_agent.mapNewRuntimeVariables(*func_des, STATE_SIZE);
            }


            /**
             * Configure runtime access of the functions variables within the FLAME_API object
             */
            cuda_agent.mapRuntimeVariables(*func_des, func_des->initial_state);

            // Zero the scan flag that will be written to
            if (func_des->has_agent_death)
                flamegpu_internal::CUDAScanCompaction::zero(flamegpu_internal::CUDAScanCompaction::AGENT_DEATH, j);

            // Count total threads being launched
            totalThreads += cuda_agent.getStateSize(func_des->initial_state);
            ++j;
        }

        // Ensure RandomManager is the correct size to accomodate all threads to be launched
        singletons->rng.resize(totalThreads, *this);
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
                message_name_inp_hash = singletons->curve.variableRuntimeHash(inpMessage_name.c_str());

                d_in_messagelist_metadata = cuda_message.getMetaDataDevicePtr();
            }

            // check if a function has an output message
            if (auto om = func_des->message_output.lock()) {
                std::string outpMessage_name = om->name;
                const CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);

                // hash message name
                message_name_outp_hash =  singletons->curve.variableRuntimeHash(outpMessage_name.c_str());
                d_out_messagelist_metadata = cuda_message.getMetaDataDevicePtr();
            }

            const CUDAAgent& cuda_agent = getCUDAAgent(agent_name);

            int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
            if (state_list_size == 0)
                continue;

            // curve hash values
            Curve::NamespaceHash agentname_hash = singletons->curve.variableRuntimeHash(agent_name.c_str());
            Curve::NamespaceHash funcname_hash = singletons->curve.variableRuntimeHash(func_name.c_str());
            Curve::NamespaceHash agentoutput_hash = func_des->agent_output.lock() ? singletons->curve.variableRuntimeHash("_agent_birth") + funcname_hash : 0;
            Curve::NamespaceHash agent_func_name_hash = agentname_hash + funcname_hash;
            int blockSize = 0;  // The launch configurator returned block size
            int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
            int gridSize = 0;  // The actual grid size needed, based on input size

            if (func_des->func) {   // compile time specified agent function launch
                // calculate the grid block size for main agent function
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func_des->func, 0, state_list_size);
                //! Round up according to CUDAAgent state list size
                gridSize = (state_list_size + blockSize - 1) / blockSize;

                (func_des->func) << <gridSize, blockSize, 0, streams.at(j) >> > (
                    modelname_hash,
                    agent_func_name_hash,
                    message_name_inp_hash,
                    message_name_outp_hash,
                    agentoutput_hash,
                    state_list_size,
                    d_in_messagelist_metadata,
                    d_out_messagelist_metadata,
                    totalThreads,
                    j);
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
                CUresult a = instance.configure(gridSize, blockSize).launch({
                        const_cast<void*>(reinterpret_cast<const void*>(&modelname_hash)),
                        reinterpret_cast<void*>(&agent_func_name_hash),
                        reinterpret_cast<void*>(&message_name_inp_hash),
                        reinterpret_cast<void*>(&message_name_outp_hash),
                        reinterpret_cast<void*>(&agentoutput_hash),
                        reinterpret_cast<void*>(&state_list_size),
                        const_cast<void*>(reinterpret_cast<const void*>(&d_in_messagelist_metadata)),
                        const_cast<void*>(reinterpret_cast<const void*>(&d_out_messagelist_metadata)),
                        reinterpret_cast<void*>(&totalThreads),
                        reinterpret_cast<void*>(&j) });
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
                cuda_message.swap(func_des->message_output_optional, cuda_agent.getStateSize(func_des->initial_state), j);
                cuda_message.clearTruncateMessageListFlag();
                cuda_message.setPBMConstructionRequiredFlag();
            }

            // Process agent death (has agent death check is handled by the method)
            // This MUST occur before agent_output, as if agent_output triggers resize then scan_flag for death will be purged
            const unsigned int PRE_DEATH_STATE_SIZE = cuda_agent.getStateSize(func_des->initial_state);
            cuda_agent.processDeath(*func_des, j);

            // Process agent state transition (Longer term merge this with process death?)
            cuda_agent.transitionState(func_des->initial_state, func_des->end_state, j);
            // Process agent function condition
            cuda_agent.clearFunctionConditionState(func_des->initial_state);

            // check if a function has an output agent
            if (auto oa = func_des->agent_output.lock()) {
                // This will act as a reserve word
                // which is added to variable hashes for agent creation on device
                CUDAAgent& output_agent = getCUDAAgent(oa->name);
                // Scatter the agent birth
                output_agent.scatterNew(func_des->agent_output_state, PRE_DEATH_STATE_SIZE, j);
                // unmap vars with curve
                output_agent.unmapNewRuntimeVariables(*func_des);
            }

            // unmap the function variables
            cuda_agent.unmapRuntimeVariables(*func_des);

            ++j;
        }

        // Execute all host functions attached to layer
        // TODO: Concurrency?
        for (auto &stepFn : (*lyr)->host_functions) {
            stepFn(this->host_api.get());
        }
        // If we have host layer functions, we might have host agent creation
        if ((*lyr)->host_functions.size())
            processHostAgentCreation();

        // cudaDeviceSynchronize();
    }

    // Execute step functions
    for (auto &stepFn : model->stepFunctions) {
        NVTX_RANGE(std::string("stepFunc").c_str());
        stepFn(this->host_api.get());
    }
    // If we have step functions, we might have host agent creation
    if (model->stepFunctions.size())
        processHostAgentCreation();

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
    // If we have exit conditions functions, we might have host agent creation
    if (model->exitConditions.size())
        processHostAgentCreation();

#ifdef VISUALISATION
        if (visualisation) {
            visualisation->updateBuffers(step_count+1);
        }
#endif
    // Update step count at the end of the step - when it has completed.
    incrementStepCounter();
    return true;
}

void CUDAAgentModel::simulate() {
    if (agent_map.size() == 0) {
        THROW InvalidCudaAgentMapSize("Simulation has no agents, in CUDAAgentModel::simulate().");  // recheck if this is really required
    }

    // Ensure singletons have been initialised
    initialiseSingletons();

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
    for (auto &initFn : model->initFunctions)
        initFn(this->host_api.get());
    // Check if host agent creation was used in init functions
    if (model->initFunctions.size())
        processHostAgentCreation();

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

void CUDAAgentModel::setPopulationData(AgentPopulation& population) {
    // Ensure singletons have been initialised
    initialiseSingletons();

    CUDAAgentMap::iterator it;
    it = agent_map.find(population.getAgentName());

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("Error: Agent ('%s') was not found, "
            "in CUDAAgentModel::setPopulationData()",
            population.getAgentName().c_str());
    }

    /*! create agent state lists */
    it->second->setPopulationData(population);

#ifdef VISUALISATION
    if (visualisation) {
        visualisation->updateBuffers();
    }
#endif
}

void CUDAAgentModel::getPopulationData(AgentPopulation& population) {
    // Ensure singletons have been initialised
    initialiseSingletons();

    CUDAAgentMap::iterator it;
    it = agent_map.find(population.getAgentName());

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("Error: Agent ('%s') was not found, "
            "in CUDAAgentModel::getPopulationData()",
            population.getAgentName().c_str());
    }

    /*!create agent state lists */
    it->second->getPopulationData(population);
}

CUDAAgent& CUDAAgentModel::getCUDAAgent(const std::string& agent_name) const {
    CUDAAgentMap::const_iterator it;
    it = agent_map.find(agent_name);

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("CUDA agent ('%s') not found, in CUDAAgentModel::getCUDAAgent().",
            agent_name.c_str());
    }

    return *(it->second);
}

AgentInterface& CUDAAgentModel::getAgent(const std::string& agent_name) {
    // Ensure singletons have been initialised
    initialiseSingletons();

    auto it = agent_map.find(agent_name);

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("CUDA agent ('%s') not found, in CUDAAgentModel::getAgent().",
            agent_name.c_str());
    }

    return *(it->second);
}

CUDAMessage& CUDAAgentModel::getCUDAMessage(const std::string& message_name) const {
    CUDAMessageMap::const_iterator it;
    it = message_map.find(message_name);

    if (it == message_map.end()) {
        THROW InvalidCudaMessage("CUDA message ('%s') not found, in CUDAAgentModel::getCUDAMessage().",
            message_name.c_str());
    }

    return *(it->second);
}

bool CUDAAgentModel::checkArgs_derived(int argc, const char** argv, int &i) {
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

void CUDAAgentModel::printHelp_derived() {
    const char *line_fmt = "%-18s %s\n";
    printf("CUDA Model Optional Arguments:\n");
    printf(line_fmt, "-d, --device", "GPU index");
}

void CUDAAgentModel::applyConfig_derived() {
    NVTX_RANGE("applyConfig_derived");
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
    gpuErrchk(cudaFree(0));

    // Initialise singletons once a device has been selected.
    // @todo - if this has already been called, before the device was selected an error should occur.
    initialiseSingletons();
}

/**
 * These values are ony used by CUDAAgentModel::initialiseSingletons()
 * Can't put a __device__ symbol method static
 */
namespace {
    __device__ unsigned int DEVICE_HAS_RESET = 0xDEADBEEF;
    const unsigned int DEVICE_HAS_RESET_FLAG = 0xDEADBEEF;
}  // namespace

void CUDAAgentModel::initialiseSingletons() {
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
            RandomManager::getInstance().purge();
            flamegpu_internal::CUDAScanCompaction::purge();
            CUDAScatter::getInstance(UINT_MAX);
            EnvironmentManager::getInstance().purge();
            // Reset flag
            DEVICE_HAS_RESET_CHECK = 0;  // Any value that doesnt match DEVICE_HAS_RESET_FLAG
            cudaMemcpyToSymbol(DEVICE_HAS_RESET, &DEVICE_HAS_RESET_CHECK, sizeof(unsigned int));
        }
        // Get references to all required singleton and store in the instance.
        singletons = new Singletons(
            Curve::getInstance(),
            RandomManager::getInstance(),
            CUDAScatter::getInstance(0),
            EnvironmentManager::getInstance());

        // Increase counters within some singletons.
        singletons->rng.increaseSimCounter();
        singletons->scatter.increaseSimCounter();

        // Populate the environment properties in constant Cache
        singletons->environment.init(model->name, *model->environment);
        // Add the CUDAAgentModel specific variables(s)
        singletons->environment.add({model->name, "_stepCount"}, 0u, false);

        // Reinitialise random for this simulation instance
        singletons->rng.reseed(getSimulationConfig().random_seed);

        singletonsInitialised = true;
    }

    // Ensure RTC is set up.
    initialiseRTC();
}

void CUDAAgentModel::initialiseRTC() {
    // Only do this once.
    if (!rtcInitialised) {
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
                    a_it->second->addInstantitateRTCFunction(*it_f->second);
                }
                // check rtc source to see if the function condition is an rtc condition
                if (!it_f->second->rtc_condition_source.empty()) {
                    // create CUDA agent RTC function condition by calling addInstantitateRTCFunction on CUDAAgent with AgentFunctionData
                    a_it->second->addInstantitateRTCFunction(*it_f->second, true);
                }
            }
        }

        // Initialise device environment for RTC
        singletons->environment.initRTC(*this);

        rtcInitialised = true;
    }
}

void CUDAAgentModel::resetDerivedConfig() {
    this->config = CUDAAgentModel::Config();
    resetStepCounter();
}


CUDAAgentModel::Config &CUDAAgentModel::CUDAConfig() {
    return config;
}
const CUDAAgentModel::Config &CUDAAgentModel::getCUDAConfig() const {
    return config;
}
#ifdef VISUALISATION
ModelVis &CUDAAgentModel::getVisualisation() {
    if (!visualisation)
        visualisation = std::make_unique<ModelVis>(*this);
    return *visualisation.get();
}
#endif

unsigned int CUDAAgentModel::getStepCounter() {
    return step_count;
}
void CUDAAgentModel::resetStepCounter() {
    step_count = 0;
}

void CUDAAgentModel::initOffsetsAndMap() {
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

void CUDAAgentModel::processHostAgentCreation() {
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

                // Resize agent list if required
                const unsigned int current_state_size = agent_map.at(agent.first)->getStateSize(state.first);
                const unsigned int current_max_state_size = agent_map.at(agent.first)->getMaximumListSize();
                if (current_state_size + state.second.size() > current_max_state_size) {
                    agent_map.at(agent.first)->resize(static_cast<unsigned int>(state.second.size()) + current_state_size, 0);  // StreamId Doesn't matter
                }
                // Scatter to device
                agent_map.at(agent.first)->getAgentStateList(state.first).scatterHostCreation(static_cast<unsigned int>(state.second.size()), dt_buff, offsets);
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

void CUDAAgentModel::RTCSafeCudaMemcpyToSymbol(const void* symbol, const char* rtc_symbol_name, const void* src, size_t count, size_t offset) const {
    // make the mem copy to runtime API symbol
    gpuErrchk(cudaMemcpyToSymbol(symbol, src, count, offset));
    // loop through agents
    for (const auto& agent_pair : agent_map) {
        // loop through any agent functions
        for (const CUDARTCFuncMapPair& rtc_func_pair : agent_pair.second->getRTCFunctions()) {
            CUdeviceptr rtc_dev_ptr = 0;
            // get the RTC device symbol
            rtc_dev_ptr = rtc_func_pair.second->get_global_ptr(rtc_symbol_name);
            // make the memcpy to the rtc version of the symbol
            gpuErrchkDriverAPI(cuMemcpyHtoD(rtc_dev_ptr + offset, src, count));
        }
    }
}

void CUDAAgentModel::RTCSafeCudaMemcpyToSymbolAddress(void* ptr, const char* rtc_symbol_name, const void* src, size_t count, size_t offset) const {
    // offset the device pointer by casting to char
    void* offset_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(ptr) + offset);
    // make the mem copy to runtime API symbol
    gpuErrchk(cudaMemcpy(offset_ptr, src, count, cudaMemcpyHostToDevice));
    // loop through agents
    for (const auto& agent_pair : agent_map) {
        // loop through any agent functions
        for (const CUDARTCFuncMapPair& rtc_func_pair : agent_pair.second->getRTCFunctions()) {
            CUdeviceptr rtc_dev_ptr = 0;
            // get the RTC device symbol
            rtc_dev_ptr = rtc_func_pair.second->get_global_ptr(rtc_symbol_name);
            // make the memcpy to the rtc version of the symbol
            gpuErrchkDriverAPI(cuMemcpyHtoD(rtc_dev_ptr + offset, src, count));
        }
    }
}

void CUDAAgentModel::RTCSetEnvironmentVariable(const char* variable_name, const void* src, size_t count, size_t offset) const {
    // get the model hash
    const Curve::VariableHash model_hash = Curve::getInstance().variableRuntimeHash(getModelDescription().name.c_str());
    // loop through agents
    for (const auto& agent_pair : agent_map) {
        // loop through any agent functions
        for (const CUDARTCFuncMapPair& rtc_func_pair : agent_pair.second->getRTCFunctions()) {
            CUdeviceptr rtc_dev_ptr = 0;
            // get the RTC device symbol
            std::string rtc_symbol_name = CurveRTCHost::getEnvVariableSymbolName(variable_name, model_hash);
            rtc_dev_ptr = rtc_func_pair.second->get_global_ptr(rtc_symbol_name.c_str());
            // make the memcpy to the rtc version of the symbol
            gpuErrchkDriverAPI(cuMemcpyHtoD(rtc_dev_ptr + offset, src, count));
        }
    }
}

void CUDAAgentModel::incrementStepCounter() {
    this->step_count++;
    this->singletons->environment.set({model->name, "_stepCount"}, this->step_count);
}


float CUDAAgentModel::getSimulationElapsedTime() const {
    // Get the value
    return this->simulation_elapsed_time;
}
