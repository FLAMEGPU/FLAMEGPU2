#include "flamegpu/gpu/CUDAEnsemble.h"

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
#include "flamegpu/exception/FGPUException.h"


std::atomic<int> CUDAEnsemble::active_instances;  // This value should default init to 0, specifying =0 was causing warnings on Windows.
bool CUDAEnsemble::AUTO_CUDA_DEVICE_RESET = true;

CUDAEnsemble::CUDAEnsemble(const ModelDescription& _model, int argc, const char** argv)
    : step_count(0)
    , ensemble_elapsed_time(0.f)
    , model(_model.model->clone())
    , streams(std::vector<cudaStream_t>())
    , singletons(nullptr)
    , singletonsInitialised(false)
    , rtcInitialised(false)
    , rtc_kernel_cache(nullptr) {
    ++active_instances;
    initOffsetsAndMap();

    // Register the signal handler.
    SignalHandlers::registerSignalHandlers();

    // We don't actually allocate any CUDA at this point
    // It's not yet clear how many instances will be required

    // Populate the environment properties
    // env doesnt work with ensemble yet
    //initEnvironmentMgr();

    // populate the CUDA submodel map
    const auto &smm = model->submodels;
    // create new cuda message and add to the map
    for (auto it_sm = smm.cbegin(); it_sm != smm.cend(); ++it_sm) {
        submodel_map.emplace(it_sm->first, std::unique_ptr<CUDAEnsemble>(new CUDAEnsemble(it_sm->second, this)));
    }

    if (argc && argv) {
        initialise(argc, argv);
    }
}
CUDAEnsemble::CUDAEnsemble(const std::shared_ptr<SubModelData> &submodel_desc, CUDAEnsemble *master_model)
    : step_count(0)
    , model(submodel_desc->submodel)
    , submodel(submodel_desc)
    , mastermodel(master_model)
    , streams(std::vector<cudaStream_t>())
    , singletons(nullptr)
    , singletonsInitialised(false)
    , rtcInitialised(false)
    , rtc_kernel_cache(nullptr)  {
    ++active_instances;
    initOffsetsAndMap();
    
    // We don't actually allocate any CUDA at this point
    // It's not yet clear how many instances will be required

    // Populate the environment properties
    // env doesnt work with ensemble yet
    //initEnvironmentMgr();

    // populate the CUDA submodel map
    const auto &smm = model->submodels;
    // create new cuda model and add to the map
    for (auto it_sm = smm.cbegin(); it_sm != smm.cend(); ++it_sm) {
        submodel_map.emplace(it_sm->first, std::unique_ptr<CUDAEnsemble>(new CUDAEnsemble(it_sm->second, this)));
    }
    // Submodels all run silent by default
    ensemble_config.verbose = false;
}

CUDAEnsemble::~CUDAEnsemble() {
    submodel_map.clear();  // Test
    // De-initialise, freeing singletons?
    // @todo - this is unsafe in a destructor as it may invoke cuda commands.
    if (singletonsInitialised) {
        // unique pointers cleanup by automatically
        // Drop all constants from the constant cache linked to this model
        //singletons->environment.free(instance_id);

        delete singletons;
        singletons = nullptr;
    }

    // We must explicitly delete all cuda members before we cuda device reset
    submodel_map.clear();
    instance_vector.clear();
    delete rtc_kernel_cache;
    rtc_kernel_cache = nullptr;

    // If we are the last instance to destruct
    if ((!--active_instances)&& AUTO_CUDA_DEVICE_RESET) {
        gpuErrchk(cudaDeviceReset());
        //EnvironmentManager::getInstance().purge();
        Curve::getInstance().purge();
    }
}

bool CUDAEnsemble::step() {
    NVTX_RANGE(std::string("CUDAEnsemble::step " + std::to_string(step_count)).c_str());

    // Ensure singletons have been initialised
    initialiseSingletons();
    // Update environment on device
    //singletons->environment.updateDevice(getInstanceID());

    // If verbose, print the step number.
    if (ensemble_config.verbose) {
        fprintf(stdout, "Processing Simulation Step %u\n", step_count);
    }

    unsigned int nStreams = 1;

    // TODO: simulation.getMaxFunctionsPerLayer()
    for (auto lyr = model->layers.begin(); lyr != model->layers.end(); ++lyr) {
        unsigned int temp = static_cast<unsigned int>((*lyr)->agent_functions.size());
        nStreams = (std::max)(nStreams, temp);
    }

    /*!  Stream creations */
    // Ensure there are enough streams to execute the layer.
    while (streams.size() < nStreams) {
        cudaStream_t stream;
        gpuErrchk(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }
    while (instance_offset_cdn_vector.size() < nStreams) {
        instance_offset_cdn_vector.push_back(InstanceOffsetData_cdn(ensemble_config.max_concurrent_runs));
        instance_offset_vector.push_back(InstanceOffsetData(ensemble_config.max_concurrent_runs));
    }

    // Reset message list flags
    for (auto &instance:instance_vector) {
        for (auto m =  instance.message_map.begin(); m != instance.message_map.end(); ++m) {
            m->second->setTruncateMessageListFlag();
        }
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
        std::vector<unsigned int> instance_offsets;
        std::vector<Curve::NamespaceHash> instance_id_hashes;
        /*! for each function apply any agent function conditions*/
        {
            // Map agent memory
            for (const std::shared_ptr<AgentFunctionData> &func_des : functions) {
                if ((func_des->condition) || (!func_des->rtc_func_condition_name.empty())) {
                    auto func_agent = func_des->parent.lock();
                    NVTX_RANGE(std::string("condition map " + func_agent->name + "::" + func_des->name).c_str());
                    
                    unsigned int flag_array_len = 0;
                    for (auto &instance:instance_vector) {
                        const CUDAAgent& cuda_agent = instance.getCUDAAgent(func_agent->name);

                        const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
                        if (state_list_size == 0) {
                            continue;
                        }

                        // Configure runtime access of the functions variables within the FLAME_API object
                        cuda_agent.mapRuntimeVariables(*func_des);

                        // Store data ready for kernel launch
                        instance_offsets.push_back(state_list_size);
                        instance_id_hashes.push_back(Curve::variableRuntimeHash(instance.getInstanceID()));

                        flag_array_len += state_list_size;
                        totalThreads += state_list_size;
                    }

                    // Resize & zero the scan flag that will be written to
                    singletons->scatter.Scan().resize(flag_array_len, CUDAScanCompaction::AGENT_DEATH, j);
                    singletons->scatter.Scan().zero(CUDAScanCompaction::AGENT_DEATH, j);
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


                    unsigned int state_list_size = 0;
                    
                    for (auto &instance:instance_vector) {
                        const CUDAAgent& cuda_agent = instance.getCUDAAgent(agent_name);
                        state_list_size += cuda_agent.getStateSize(func_des->initial_state);
                    }

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
                    unsigned int total_launching_instances = instance_id_hashes.size();
#ifndef NO_SEATBELTS
                    auto *error_buffer = this->singletons->exception.getDevicePtr(j);
                    sm_size = sizeof(error_buffer);
#endif
                    // Fill the params to send to kernel
                    {
                        instance_offsets.push_back(state_list_size);
                        gpuErrchk(cudaMemcpy(instance_offset_cdn_vector[j].d_instance_offsets, instance_offsets.data(), sizeof(unsigned int) * instance_offsets.size(), cudaMemcpyHostToDevice));
                        gpuErrchk(cudaMemcpy(instance_offset_cdn_vector[j].d_instance_id_hashes, instance_id_hashes.data(), sizeof(Curve::NamespaceHash) * instance_id_hashes.size(), cudaMemcpyHostToDevice));
                    }
                    // switch between normal and RTC agent function condition
                    if (func_des->ensemble_condition) {
                        // calculate the grid block size for agent function condition
                        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func_des->ensemble_condition, 0, state_list_size);

                        //! Round up according to CUDAAgent state list size
                        gridSize = (state_list_size + blockSize - 1) / blockSize;
                        (func_des->ensemble_condition) << <gridSize, blockSize, sm_size, streams.at(j) >> > (
#ifndef NO_SEATBELTS
                        error_buffer,
#endif
                        total_launching_instances,
                        instance_offset_cdn_vector[j].d_instance_offsets,
                        instance_offset_cdn_vector[j].d_instance_id_hashes,
                        agent_func_name_hash,
                        t_rng,
                        scanFlag_agentDeath);
                        gpuErrchkLaunch();
                    } else {  // RTC function
                        std::string func_condition_identifier = func_name + "_condition";
                        // get instantiation
                        const jitify::KernelInstantiation& instance = instance_vector[0].getCUDAAgent(agent_name).getRTCInstantiation(func_condition_identifier);
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
                                reinterpret_cast<void*>(&total_launching_instances),
                                const_cast<void*>(reinterpret_cast<const void*>(&instance_offset_cdn_vector[j].d_instance_offsets)),
                                const_cast<void*>(reinterpret_cast<const void*>(&instance_offset_cdn_vector[j].d_instance_id_hashes)),
                                reinterpret_cast<void*>(&agent_func_name_hash),
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

                    unsigned int flag_array_offset = 0;
                    for (auto &instance:instance_vector) {
                        CUDAAgent& cuda_agent = instance.getCUDAAgent(func_agent->name);

                        const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
                        if (state_list_size == 0) {
                            continue;
                        }

                        // unmap the function variables
                        cuda_agent.unmapRuntimeVariables(*func_des);

#ifndef NO_SEATBELTS
                        // Error check after unmap vars
                        // This means that curve is cleaned up before we throw exception (mostly prevents curve being polluted if we catch and handle errors)
                        this->singletons->exception.checkError("condition " + func_des->name, j);//This needs flag_array_offset passed in
#endif

                        // Process agent function condition
                        cuda_agent.processFunctionCondition(*func_des, this->singletons->scatter, j);//This needs flag_array_offset passed in
                        flag_array_offset += state_list_size;
                    }
                    ++j;
                }
            }
        }

        j = 0;
        instance_offsets.clear();
        instance_id_hashes.clear();
        std::vector<const void*> d_in_messagelist_metadatas;
        std::vector<const void*> d_out_messagelist_metadatas;
        // Sum the total number of threads being launched in the layer
        totalThreads = 0;
        /*! for each func function - Loop through to do all mapping of agent and message variables */
        for (const std::shared_ptr<AgentFunctionData> &func_des : functions) {
            auto func_agent = func_des->parent.lock();
            if (!func_agent) {
                THROW InvalidAgentFunc("Agent function refers to expired agent.");
            }
            NVTX_RANGE(std::string("map" + func_agent->name + "::" + func_des->name).c_str());
            unsigned int scan_flag_len = 0;
            for (auto &instance:instance_vector) {
                const CUDAAgent& cuda_agent = instance.getCUDAAgent(func_agent->name);
                const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
                if (state_list_size == 0) {
                    continue;
                }

                // check if a function has an input message
                if (auto im = func_des->message_input.lock()) {
                    std::string inpMessage_name = im->name;
                    CUDAMessage& cuda_message = instance.getCUDAMessage(inpMessage_name);
                    // Construct PBM here if required!!
                    cuda_message.buildIndex(this->singletons->scatter, j);
                    // Map variables after, as index building can swap arrays
                    cuda_message.mapReadRuntimeVariables(*func_des, cuda_agent);
                    // Store data ready for kernel launch
                    d_in_messagelist_metadatas.push_back(cuda_message.getMetaDataDevicePtr());
                }

                // check if a function has an output message
                if (auto om = func_des->message_output.lock()) {
                    std::string outpMessage_name = om->name;
                    CUDAMessage& cuda_message = instance.getCUDAMessage(outpMessage_name);
                    // Resize message list if required
                    const unsigned int existingMessages = cuda_message.getTruncateMessageListFlag() ? 0 : cuda_message.getMessageCount();
                    cuda_message.resize(existingMessages + state_list_size, this->singletons->scatter, j);
                    cuda_message.mapWriteRuntimeVariables(*func_des, cuda_agent, state_list_size);
                    // Store data ready for kernel launch
                    d_out_messagelist_metadatas.push_back(cuda_message.getMetaDataDevicePtr());
                }

                // check if a function has an output agent
                if (auto oa = func_des->agent_output.lock()) {
                    // This will act as a reserve word
                    // which is added to variable hashes for agent creation on device
                    CUDAAgent& output_agent = instance.getCUDAAgent(oa->name);

                    // Map vars with curve (this allocates/requests enough new buffer space if an existing version is not available/suitable)
                    output_agent.mapNewRuntimeVariables(cuda_agent, *func_des, state_list_size, this->singletons->scatter, j);//This needs something doing, to ensure scan array is sized for all instances properly
                }


                /**
                 * Configure runtime access of the functions variables within the FLAME_API object
                 */
                cuda_agent.mapRuntimeVariables(*func_des);
                
                // Store data ready for kernel launch
                instance_offsets.push_back(state_list_size);
                instance_id_hashes.push_back(Curve::variableRuntimeHash(instance.getInstanceID()));

                // Count total threads being launched
                totalThreads += state_list_size;
                scan_flag_len += state_list_size;
            }
            // Resize death flag array if necessary
            singletons->scatter.Scan().resize(scan_flag_len, CUDAScanCompaction::AGENT_DEATH, j);
            // Zero the scan flag that will be written to
            if (func_des->has_agent_death) {
                singletons->scatter.Scan().zero(CUDAScanCompaction::AGENT_DEATH, j);
            }
            // Resize message output flag if necessary
            if (auto om = func_des->message_output.lock()) {
                singletons->scatter.Scan().resize(scan_flag_len, CUDAScanCompaction::MESSAGE_OUTPUT, j);
                // Zero the scan flag that will be written to
                if (func_des->message_output_optional)
                    singletons->scatter.Scan().zero(CUDAScanCompaction::MESSAGE_OUTPUT, j);
            }
            // Resize the agent output flag if necessary
            if (auto oa = func_des->agent_output.lock()) {
                // Notify scan flag that it might need resizing
                // We need a 3rd array, because a function might combine agent birth, agent death and message output
                singletons->scatter.Scan().resize(scan_flag_len, CUDAScanCompaction::AGENT_OUTPUT, j);
                // Ensure the scan flag is zeroed
                singletons->scatter.Scan().zero(CUDAScanCompaction::AGENT_OUTPUT, j);
            }
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
            unsigned int state_list_size = 0;
            for (auto &instance:instance_vector) {
                const CUDAAgent& cuda_agent = instance.getCUDAAgent(func_agent->name);
                state_list_size += cuda_agent.getStateSize(func_des->initial_state);
            }

            if (state_list_size == 0) {
                ++j;
                continue;
            }

            int blockSize = 0;  // The launch configurator returned block size
            int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
            int gridSize = 0;  // The actual grid size needed, based on input size

            // Agent function kernel wrapper args
            Curve::NamespaceHash agentname_hash = Curve::variableRuntimeHash(func_agent->name.c_str());
            Curve::NamespaceHash funcname_hash = Curve::variableRuntimeHash(func_des->name.c_str());
            Curve::NamespaceHash agent_func_name_hash = agentname_hash + funcname_hash;
            Curve::NamespaceHash d_message_name_inp_hash = func_des->message_input.lock() ? singletons->curve.variableRuntimeHash(func_des->message_input.lock()->name.c_str()) : 0;
            Curve::NamespaceHash d_message_name_outp_hash = func_des->message_output.lock() ? singletons->curve.variableRuntimeHash(func_des->message_output.lock()->name.c_str()) : 0;
            Curve::NamespaceHash agentoutput_hash = func_des->agent_output.lock() ? singletons->curve.variableRuntimeHash("_agent_birth") + funcname_hash : 0;
            curandState * t_rng = d_rng + totalThreads;
            unsigned int *scanFlag_agentDeath = func_des->has_agent_death ? this->singletons->scatter.Scan().Config(CUDAScanCompaction::Type::AGENT_DEATH, j).d_ptrs.scan_flag : nullptr;
            unsigned int *scanFlag_messageOutput = this->singletons->scatter.Scan().Config(CUDAScanCompaction::Type::MESSAGE_OUTPUT, j).d_ptrs.scan_flag;
            unsigned int *scanFlag_agentOutput = this->singletons->scatter.Scan().Config(CUDAScanCompaction::Type::AGENT_OUTPUT, j).d_ptrs.scan_flag;
            unsigned int sm_size = 0;
            unsigned int total_launching_instances = instance_id_hashes.size();
#ifndef NO_SEATBELTS
            auto *error_buffer = this->singletons->exception.getDevicePtr(j);
            sm_size = sizeof(error_buffer);
#endif
            // Fill the params to send to kernel
            {
                instance_offsets.push_back(state_list_size);
                gpuErrchk(cudaMemcpy(instance_offset_vector[j].d_instance_offsets, instance_offsets.data(), sizeof(unsigned int) * instance_offsets.size(), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(instance_offset_vector[j].d_instance_id_hashes, instance_id_hashes.data(), sizeof(Curve::NamespaceHash) * instance_id_hashes.size(), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(instance_offset_vector[j].d_in_messagelist_metadata, d_in_messagelist_metadatas.data(), sizeof(void*) * instance_id_hashes.size(), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(instance_offset_vector[j].d_out_messagelist_metadata, d_out_messagelist_metadatas.data(), sizeof(void*) * instance_id_hashes.size(), cudaMemcpyHostToDevice));
            }

            if (func_des->ensemble_func) {   // compile time specified agent function launch
                // calculate the grid block size for main agent function
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func_des->func, 0, state_list_size);
                //! Round up according to CUDAAgent state list size
                gridSize = (state_list_size + blockSize - 1) / blockSize;

                (func_des->ensemble_func) << <gridSize, blockSize, sm_size, streams.at(j) >> > (
#ifndef NO_SEATBELTS
                    error_buffer,
#endif
                    total_launching_instances,
                    instance_offset_cdn_vector[j].d_instance_offsets,
                    instance_offset_cdn_vector[j].d_instance_id_hashes,
                    agent_func_name_hash,
                    d_message_name_inp_hash,
                    d_message_name_outp_hash,
                    agentoutput_hash,
                    instance_offset_vector[j].d_in_messagelist_metadata,
                    instance_offset_vector[j].d_out_messagelist_metadata,
                    t_rng,
                    scanFlag_agentDeath,
                    scanFlag_messageOutput,
                    scanFlag_agentOutput);
                gpuErrchkLaunch();
            } else {      // assume this is a runtime specified agent function
//              // RTC disabled, dynamic header will need some big changes
//                // get instantiation
//                const jitify::KernelInstantiation&  instance = cuda_agent.getRTCInstantiation(func_des->name);
//                // calculate the grid block size for main agent function
//                CUfunction cu_func = (CUfunction)instance;
//                cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cu_func, 0, 0, state_list_size);
//                //! Round up according to CUDAAgent state list size
//                gridSize = (state_list_size + blockSize - 1) / blockSize;
//                // launch the kernel
//                CUresult a = instance.configure(gridSize, blockSize, sm_size, streams.at(j)).launch({
//#ifndef NO_SEATBELTS
//                        reinterpret_cast<void*>(&error_buffer),
//#endif
//                        reinterpret_cast<void*>(&total_launching_instances),
//                        const_cast<void*>(reinterpret_cast<const void*>(&instance_offset_cdn_vector[j].d_instance_offsets)),
//                        const_cast<void*>(reinterpret_cast<const void*>(&instance_offset_cdn_vector[j].d_instance_id_hashes)),
//                        reinterpret_cast<void*>(&agent_func_name_hash),
//                        reinterpret_cast<void*>(&message_name_inp_hash),
//                        reinterpret_cast<void*>(&message_name_outp_hash),
//                        reinterpret_cast<void*>(&agentoutput_hash),
//                        const_cast<void*>(reinterpret_cast<const void*>(&instance_offset_vector[j].d_in_messagelist_metadata)),
//                        const_cast<void*>(reinterpret_cast<const void*>(&instance_offset_vector[j].d_out_messagelist_metadata)),
//                        const_cast<void*>(reinterpret_cast<const void*>(&t_rng)),
//                        reinterpret_cast<void*>(&scanFlag_agentDeath),
//                        reinterpret_cast<void*>(&scanFlag_messageOutput),
//                        reinterpret_cast<void*>(&scanFlag_agentOutput)});
//                if (a != CUresult::CUDA_SUCCESS) {
//                    const char* err_str = nullptr;
//                    cuGetErrorString(a, &err_str);
//                    THROW InvalidAgentFunc("There was a problem launching the runtime agent function '%s': %s", func_name.c_str(), err_str);
//                }
//                cudaDeviceSynchronize();
//                gpuErrchkLaunch();
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
            unsigned int scan_flag_offset = 0;
            for (auto &instance:instance_vector) {
                CUDAAgent& cuda_agent = instance.getCUDAAgent(func_agent->name);

                const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
                // If agent function wasn't executed, these are redundant
                if (state_list_size > 0) {
                    // check if a function has an input message
                    if (auto im = func_des->message_input.lock()) {
                        std::string inpMessage_name = im->name;
                        const CUDAMessage& cuda_message = instance.getCUDAMessage(inpMessage_name);
                        cuda_message.unmapRuntimeVariables(*func_des);
                    }

                    // check if a function has an output message
                    if (auto om = func_des->message_output.lock()) {
                        std::string outpMessage_name = om->name;
                        CUDAMessage& cuda_message = instance.getCUDAMessage(outpMessage_name);
                        cuda_message.unmapRuntimeVariables(*func_des);
                        cuda_message.swap(func_des->message_output_optional, state_list_size, this->singletons->scatter, j);//Requires passing scan_flag_offset for optional output?
                        cuda_message.clearTruncateMessageListFlag();
                        cuda_message.setPBMConstructionRequiredFlag();
                    }

                    // Process agent death (has agent death check is handled by the method)
                    // This MUST occur before agent_output, as if agent_output triggers resize then scan_flag for death will be purged
                    cuda_agent.processDeath(*func_des, this->singletons->scatter, j);//Requires passing scan_flag_offset

                    // Process agent state transition (Longer term merge this with process death?)
                    cuda_agent.transitionState(func_des->initial_state, func_des->end_state, this->singletons->scatter, j);
                }

                // Process agent function condition
                cuda_agent.clearFunctionCondition(func_des->initial_state);

                // If agent function wasn't executed, these are redundant
                if (state_list_size > 0) {
                    // check if a function has an output agent
                    if (auto oa = func_des->agent_output.lock()) {//This will need some changing to support shared flag arrays, not sure what yet
                        // This will act as a reserve word
                        // which is added to variable hashes for agent creation on device
                        CUDAAgent& output_agent = instance.getCUDAAgent(oa->name);
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
                    this->singletons->exception.checkError(func_des->name, j);//Requires passing scan_flag_offset
#endif
                }
            }
            ++j;
        }

        NVTX_PUSH("CUDAEnsemble::step::HostFunctions");
        // Execute all host functions attached to layer
        // TODO: Concurrency?
        assert(host_api.size());
        for (auto &stepFn : (*lyr)->host_functions) {
            for (auto &instance:instance_vector) {
                NVTX_RANGE("hostFunc");
                stepFn(instance.host_api.get());
            }
        }
        // Execute all host function callbacks attached to layer
        for (auto &stepFn : (*lyr)->host_functions_callbacks) {
            for (auto &instance:instance_vector) {
                NVTX_RANGE("hostFunc_swig");
                stepFn->run(instance.host_api.get());
            }
        }
        // If we have host layer functions, we might have host agent creation
        if ((*lyr)->host_functions.size() || ((*lyr)->host_functions_callbacks.size()))
            processHostAgentCreation(j);
        // Update environment on device
        //singletons->environment.updateDevice(getInstanceID());
        NVTX_POP();

        // cudaDeviceSynchronize();
    }

    NVTX_PUSH("CUDAEnsemble::step::StepFunctions");
    // Execute step functions
    for (auto &stepFn : model->stepFunctions) {
        for (auto &instance:instance_vector) {
            NVTX_RANGE("stepFunc");
            stepFn(instance.host_api.get());
        }
    }
    // Execute step function callbacks
    for (auto &stepFn : model->stepFunctionCallbacks) {
        for (auto &instance:instance_vector) {
            NVTX_RANGE("stepFunc_swig");
            stepFn->run(instance.host_api.get());
        }
    }
    // If we have step functions, we might have host agent creation
    if (model->stepFunctions.size() || model->stepFunctionCallbacks.size())
        processHostAgentCreation(0);
    NVTX_POP();

    // Execute exit conditions
    for (auto &exitCdns : model->exitConditions)
        if (exitCdns(this->host_api.get()) == EXIT) {

            // If there were any exit conditions, we also need to update the step count
            incrementStepCounter();
            return false;
        }
    // Execute exit condition callbacks
    for (auto &exitCdns : model->exitConditionCallbacks)
        if (exitCdns->run(this->host_api.get()) == EXIT) {

            // If there were any exit conditions, we also need to update the step count
            incrementStepCounter();
            return false;
        }
    // If we have exit conditions functions, we might have host agent creation
    if (model->exitConditions.size() || model->exitConditionCallbacks.size())
        processHostAgentCreation(0);


    // Update step count at the end of the step - when it has completed.
    incrementStepCounter();
    return true;
}

void CUDAEnsemble::simulate() {
    //if (agent_map.size() == 0) {
    //    THROW InvalidCudaAgentMapSize("Simulation has no agents, in CUDAEnsemble::simulate().");  // recheck if this is really required
    //}

    // Ensure singletons have been initialised
    initialiseSingletons();

    // Ensemble execution state tracking (maybe move this to class level?)
    assert(ensemble_config.max_concurrent_runs > 2);
    const unsigned int instances_required = ensemble_config.total_runs;
    unsigned int instances_completed = 0;
    unsigned int active_instances = ensemble_config.max_concurrent_runs;

    // Reinitialise any unmapped agent variables
    if (submodel) {
        int j = 0;
        for (auto &i:instance_vector) {
            for (auto &a : i.agent_map) {
                a.second->initUnmappedVars(this->singletons->scatter, j++);
            }
        }
    }

    // create cude events to record the elapsed time for simulate.
    cudaEvent_t ensembleStartEvent = nullptr;
    cudaEvent_t ensembleEndEvent = nullptr;
    gpuErrchk(cudaEventCreate(&ensembleStartEvent));
    gpuErrchk(cudaEventCreate(&ensembleEndEvent));
    // Record the start event.
    gpuErrchk(cudaEventRecord(ensembleStartEvent));
    // Reset the elapsed time.
    ensemble_elapsed_time = 0.f;

    // CUDAAgentMap::iterator it;

    // check any CUDAAgents with population size == 0
    // if they have executable functions then these can be ignored
    // if they have agent creations then buffer space must be allocated for them

    // Execute init functions
    NVTX_PUSH("CUDAEnsemble::step::InitFunctions");
    for (auto &initFn : model->initFunctions) {
        for (auto &instance:instance_vector) {
            NVTX_RANGE("initFunc");
            initFn(instance.host_api.get());
        }
    }
    // Execute init function callbacks
    for (auto &initFn : model->initFunctionCallbacks) {
        for (auto &instance:instance_vector) {
            NVTX_RANGE("initFunc_swig");
            initFn->run(instance.host_api.get());
        }
    }
    // Check if host agent creation was used in init functions
    if (model->initFunctions.size() || model->initFunctionCallbacks.size())
        processHostAgentCreation(0);
    // Update environment on device
    //singletons->environment.updateDevice(getInstanceID());
    NVTX_POP();

    if (ensemble_config.steps == 0 && model->exitConditions.size() == 0 && model->exitConditionCallbacks.size() == 0) {
        THROW EnsembleException("Models executed as ensembles must either be executed with a fixed step count or have atleast 1 exit condition.");
    }

    for (unsigned int i = 0; ensemble_config.steps == 0 ? true : i < ensemble_config.steps; i++) {
        // std::cout <<"step: " << i << std::endl;
        if (!step())
            break;
    }
    
    NVTX_PUSH("CUDAEnsemble::step::ExitFunctions");
    // Execute exit functions
    for (auto &exitFn : model->exitFunctions) {
        for (auto &instance:instance_vector) {
            NVTX_RANGE("exitFunc");
            exitFn(instance.host_api.get());
        }
    }
    for (auto &exitFn : model->exitFunctionCallbacks) {
        for (auto &instance:instance_vector) {
            NVTX_RANGE("exitFunc_swig");
            exitFn->run(instance.host_api.get());
        }
    }
    NVTX_POP();


    // Record and store the elapsed time
    // if --timing was passed, output to stdout.
    gpuErrchk(cudaEventRecord(ensembleEndEvent));
    // Syncrhonize the stop event
    gpuErrchk(cudaEventSynchronize(ensembleEndEvent));
    gpuErrchk(cudaEventElapsedTime(&ensemble_elapsed_time, ensembleStartEvent, ensembleEndEvent));

    gpuErrchk(cudaEventDestroy(ensembleStartEvent));
    gpuErrchk(cudaEventDestroy(ensembleEndEvent));
    ensembleStartEvent = nullptr;
    ensembleEndEvent = nullptr;

    if (ensemble_config.timing) {
        // Record the end event.
        // Resolution is 0.5 microseconds, so print to 1 us.
        fprintf(stdout, "Total Processing time: %.3f ms\n", ensemble_elapsed_time);
    }

    // Destroy streams.
    for (auto stream : streams) {
        gpuErrchk(cudaStreamDestroy(stream));
    }
    streams.clear();
}

void CUDAEnsemble::reset(bool submodelReset) {
    // Reset step counter
    //resetStepCounter();

    if (singletonsInitialised) {
        //// Reset environment properties
        //singletons->environment.resetModel(instance_id, *model->environment);

        //// Reseed random, unless performing submodel reset
        //if (!submodelReset) {
        //    singletons->rng.reseed(ensemble_config.random_seed);
        //}
    }

    // Cull agents
    if (submodel) {
        // Submodels only want to reset unmapped states, otherwise they will break parent model
        for (auto &instance:instance_vector) {
            for (auto &a : instance.agent_map) {
                a.second->cullUnmappedStates();
            }
        }
    } else {
        for (auto &instance:instance_vector) {
            for (auto &a : instance.agent_map) {
                a.second->cullAllStates();
            }
        }
    }

    // Cull messagelists
    for (auto &instance:instance_vector) {
        for (auto &a : instance.message_map) {
            a.second->setMessageCount(0);
            a.second->setTruncateMessageListFlag();
        }
    }


    // Trigger reset in all submodels, propagation is not necessary when performing submodel reset
    if (!submodelReset) {
        for (auto &s : submodel_map) {
            s.second->reset(false);
        }
    }
}

void CUDAEnsemble::setPopulationData(const unsigned int &index, AgentPopulation& population) {
    // Ensure singletons have been initialised
    initialiseSingletons();

    auto it = agent_map.find(population.getAgentName());

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("Error: Agent ('%s') was not found, "
            "in CUDAEnsemble::setPopulationData()",
            population.getAgentName().c_str());
    }

    /*! create agent state lists */
    it->second->setPopulationData(population, this->singletons->scatter, 0);  // Streamid shouldn't matter here

}

void CUDAEnsemble::getPopulationData(const unsigned int &index, AgentPopulation& population) {
    // Ensure singletons have been initialised
    initialiseSingletons();

    auto it = agent_map.find(population.getAgentName());

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("Error: Agent ('%s') was not found, "
            "in CUDAEnsemble::getPopulationData()",
            population.getAgentName().c_str());
    }

    /*!create agent state lists */
    it->second->getPopulationData(population);
}

void CUDAEnsemble::reseed(const unsigned int &seed) {
    //SimulationConfig().random_seed = seed;
    //singletons->rng.reseed(seed);

    //// Propagate to submodels
    //int i = 7;
    //for (auto &sm : submodel_map) {
    //    // Pass random seed on to submodels
    //    sm.second->singletons->rng.reseed(getSimulationConfig().random_seed * i * 23);
    //    // Mutate seed
    //    i *= 13;
    //}
}
/**
 * These values are ony used by CUDAEnsemble::initialiseSingletons()
 * Can't put a __device__ symbol method static
 */
namespace {
    __device__ unsigned int DEVICE_HAS_RESET = 0xDEADBEEF;
    const unsigned int DEVICE_HAS_RESET_FLAG = 0xDEADBEEF;
}  // namespace

void CUDAEnsemble::initialiseSingletons() {
    // Only do this once.
    if (!singletonsInitialised) {
        // If the device has not been specified, also check the compute capability is OK
        // Check the compute capability of the device, throw an exception if not valid for the executable.
        if (!util::compute_capability::checkComputeCapability(static_cast<int>(cuda_config.device_id))) {
            int min_cc = util::compute_capability::minimumCompiledComputeCapability();
            int cc = util::compute_capability::getComputeCapability(static_cast<int>(cuda_config.device_id));
            THROW InvalidCUDAComputeCapability("Error application compiled for CUDA Compute Capability %d and above. Device %u is compute capability %d. Rebuild for SM_%d.", min_cc, cuda_config.device_id, cc, cc);
        }
        // Check if device has been reset
        unsigned int DEVICE_HAS_RESET_CHECK = 0;
        cudaMemcpyFromSymbol(&DEVICE_HAS_RESET_CHECK, DEVICE_HAS_RESET, sizeof(unsigned int));
        if (DEVICE_HAS_RESET_CHECK == DEVICE_HAS_RESET_FLAG) {
            // Device has been reset, purge host mirrors of static objects/singletons
            Curve::getInstance().purge();
            if (singletons) {
                //singletons->rng.purge();
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
        //singletons->rng.reseed(getSimulationConfig().random_seed);

        // Propagate singleton init to submodels
        for (auto &sm : submodel_map) {
            sm.second->initialiseSingletons();
        }

        singletonsInitialised = true;
    }

    // Ensure RTC is set up.
    initialiseRTC();

    // Update environment on device
    //singletons->environment.updateDevice(getInstanceID());
}

void CUDAEnsemble::initialiseRTC() {
    //// Only do this once.
    //if (!rtcInitialised) {
    //    // Create jitify cache
    //    if (!rtc_kernel_cache) {
    //        rtc_kernel_cache = new jitify::JitCache();
    //    }
    //    // Build any RTC functions
    //    const auto& am = model->agents;
    //    // iterate agents and then agent functions to find any rtc functions or function conditions
    //    for (auto it = am.cbegin(); it != am.cend(); ++it) {
    //        auto a_it = agent_map.find(it->first);
    //        const auto& mf = it->second->functions;
    //        for (auto it_f = mf.cbegin(); it_f != mf.cend(); ++it_f) {
    //            // check rtc source to see if this is a RTC function
    //            if (!it_f->second->rtc_source.empty()) {
    //                // create CUDA agent RTC function by calling addInstantitateRTCFunction on CUDAAgent with AgentFunctionData
    //                a_it->second->addInstantitateRTCFunction(*rtc_kernel_cache, *it_f->second);
    //            }
    //            // check rtc source to see if the function condition is an rtc condition
    //            if (!it_f->second->rtc_condition_source.empty()) {
    //                // create CUDA agent RTC function condition by calling addInstantitateRTCFunction on CUDAAgent with AgentFunctionData
    //                a_it->second->addInstantitateRTCFunction(*rtc_kernel_cache, *it_f->second, true);
    //            }
    //        }
    //    }

    //    // Initialise device environment for RTC
    //    singletons->environment.initRTC(*this);

    //    rtcInitialised = true;
    //}
}



CUDAEnsemble::CConfig &CUDAEnsemble::CUDAConfig() {
    return cuda_config;
}
const CUDAEnsemble::CConfig &CUDAEnsemble::getCUDAConfig() const {
    return cuda_config;
}
CUDAEnsemble::EConfig &CUDAEnsemble::EnsembleConfig() {
    return ensemble_config;
}
const CUDAEnsemble::EConfig &CUDAEnsemble::getEnsembleConfig() const {
    return ensemble_config;
}

void CUDAEnsemble::initOffsetsAndMap() {
    const auto &md = *model.get();
    // Build offsets
    agentOffsets.clear();
    for (const auto &agent : md.agents) {
        agentOffsets.emplace(agent.first, VarOffsetStruct(agent.second->variables));
    }
}

void CUDAEnsemble::processHostAgentCreation(const unsigned int &streamId) {
    for (auto &instance:instance_vector) {
        size_t t_bufflen = 0;
        char *t_buff = nullptr;
        char *dt_buff = nullptr;
        // For each agent type
        for (auto &agent : instance.agentData) {
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
                    auto &cudaagent = instance.agent_map.at(agent.first);
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
}

void CUDAEnsemble::RTCSafeCudaMemcpyToSymbol(const void* symbol, const char* rtc_symbol_name, const void* src, size_t count, size_t offset) const {
    //// make the mem copy to runtime API symbol
    //gpuErrchk(cudaMemcpyToSymbol(symbol, src, count, offset));
    //// loop through agents
    //for (const auto& agent_pair : agent_map) {
    //    // loop through any agent functions
    //    for (const CUDAAgent::CUDARTCFuncMapPair& rtc_func_pair : agent_pair.second->getRTCFunctions()) {
    //        CUdeviceptr rtc_dev_ptr = 0;
    //        // get the RTC device symbol
    //        rtc_dev_ptr = rtc_func_pair.second->get_global_ptr(rtc_symbol_name);
    //        // make the memcpy to the rtc version of the symbol
    //        gpuErrchkDriverAPI(cuMemcpyHtoD(rtc_dev_ptr + offset, src, count));
    //    }
    //}
}

void CUDAEnsemble::RTCSafeCudaMemcpyToSymbolAddress(void* ptr, const char* rtc_symbol_name, const void* src, size_t count, size_t offset) const {
    //// offset the device pointer by casting to char
    //void* offset_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(ptr) + offset);
    //// make the mem copy to runtime API symbol
    //gpuErrchk(cudaMemcpy(offset_ptr, src, count, cudaMemcpyHostToDevice));
    //// loop through agents
    //for (const auto& agent_pair : agent_map) {
    //    // loop through any agent functions
    //    for (const CUDAAgent::CUDARTCFuncMapPair& rtc_func_pair : agent_pair.second->getRTCFunctions()) {
    //        CUdeviceptr rtc_dev_ptr = 0;
    //        // get the RTC device symbol
    //        rtc_dev_ptr = rtc_func_pair.second->get_global_ptr(rtc_symbol_name);
    //        // make the memcpy to the rtc version of the symbol
    //        gpuErrchkDriverAPI(cuMemcpyHtoD(rtc_dev_ptr + offset, src, count));
    //    }
    //}
}

void CUDAEnsemble::RTCUpdateEnvironmentVariables(const void* src, size_t count) const {
    //// loop through agents
    //for (const auto& agent_pair : agent_map) {
    //    // loop through any agent functions
    //    for (const CUDAAgent::CUDARTCFuncMapPair& rtc_func_pair : agent_pair.second->getRTCFunctions()) {
    //        CUdeviceptr rtc_dev_ptr = 0;
    //        // get the RTC device symbol
    //        std::string rtc_symbol_name = CurveRTCHost::getEnvVariableSymbolName();
    //        rtc_dev_ptr = rtc_func_pair.second->get_global_ptr(rtc_symbol_name.c_str());
    //        // make the memcpy to the rtc version of the symbol
    //        gpuErrchkDriverAPI(cuMemcpyHtoD(rtc_dev_ptr, src, count));
    //    }
    //}
}

void CUDAEnsemble::incrementStepCounter() {
    for (auto&instance:instance_vector) {
        instance.incrementStepCounter();
    }
    //this->singletons->environment.setProperty({instance_id, "_stepCount"}, this->step_count);
}

void CUDAEnsemble::initEnvironmentMgr() {
    // Populate the environment properties
    //if (!submodel) {
    //    EnvironmentManager::getInstance().init(instance_id, *model->environment);
    //} else {
    //    EnvironmentManager::getInstance().init(instance_id, *model->environment, mastermodel->getInstanceID(), *submodel->subenvironment);
    //}

    //// Add the CUDAEnsemble specific variables(s)
    //EnvironmentManager::getInstance().newProperty({instance_id, "_stepCount"}, 0u, false);
}


int CUDAEnsemble::checkArgs(int argc, const char** argv) {
    // Required args
    if (argc < 1) {
        printHelp(argv[0]);
        return false;
    }

    // First pass only looks for and handles input files
    // Remaining arguments can override args passed via input file
    int i = 1;
    for (; i < argc; i++) {
    //    // -in <string>, Specifies the input state file
    //    if (arg.compare("--in") == 0 || arg.compare("-i") == 0) {
    //        if (i + 1 >= argc) {
    //            fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
    //            return false;
    //        }
    //        const std::string new_input_file = std::string(argv[++i]);
    //        config.input_file = new_input_file;
    //        // Load the input file
    //        {
    //            // Build population vector
    //            std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> pops;
    //            for (auto &agent : model->agents) {
    //                auto a = std::make_shared<AgentPopulation>(*agent.second->description);
    //                pops.emplace(agent.first, a);
    //            }
    //            StateReader *read__ = ReaderFactory::createReader(model->name, getInstanceID(), pops, config.input_file.c_str(), this);
    //            if (read__) {
    //                read__->parse();
    //                for (auto &agent : pops) {
    //                    setPopulationData(*agent.second);
    //                }
    //            }
    //        }
    //        // Reset input file (we don't support input file recursion)
    //        config.input_file = new_input_file;
    //        // Set flag so input file isn't reloaded via apply_config
    //        loaded_input_file = new_input_file;
    //        // Break, we have loaded an input file
    //        break;
    //    }
    }

    // Parse optional args
    i = 1;
    for (; i < argc; i++) {
        // Get arg as lowercase
        std::string arg(argv[i]);
        std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) { return std::use_facet< std::ctype<char>>(std::locale()).tolower(c); });
        // -in <string>, Specifies the input state file
        if (arg.compare("--in") == 0 || arg.compare("-i") == 0) {
            // We already processed input file above, skip here
            ++i;
            continue;
        }
        // -out <string>, Specifies the output directory
        if (arg.compare("--out") == 0 || arg.compare("-o") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
        }
        // -steps <uint>, Number of simulation iterations per model instance
        if (arg.compare("--steps") == 0 || arg.compare("-s") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            ensemble_config.steps = static_cast<int>(strtoul(argv[++i], nullptr, 0));
            continue;
        }
        // -instances <uint>, Number of model instances to execute
        if (arg.compare("--instances") == 0 || arg.compare("-i") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            ensemble_config.total_runs = static_cast<int>(strtoul(argv[++i], nullptr, 0));
            continue;
        }
        // -concurrent <uint>, Number of instances to execute concurrently
        if (arg.compare("--concurrent") == 0 || arg.compare("-c") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
                return false;
            }
            ensemble_config.max_concurrent_runs = static_cast<int>(strtoul(argv[++i], nullptr, 0));
            continue;
        }
        //// -random <uint>, Uses the specified random seed, defaults to clock
        //if (arg.compare("--random") == 0 || arg.compare("-r") == 0) {
        //    if (i + 1 >= argc) {
        //        fprintf(stderr, "%s requires a trailing argument\n", arg.c_str());
        //        return false;
        //    }
        //    // Reinitialise RandomManager state
        //    config.random_seed = static_cast<unsigned int>(strtoul(argv[++i], nullptr, 0));
        //    continue;
        //}
        // -v/--verbose, Verbose FLAME GPU output.
        if (arg.compare("--verbose") == 0 || arg.compare("-v") == 0) {
            ensemble_config.verbose = true;
            continue;
        }
        // -t/--timing, Output timing information to stdout
        if (arg.compare("--timing") == 0 || arg.compare("-t") == 0) {
            ensemble_config.timing = true;
            continue;
        }
        fprintf(stderr, "Unexpected argument: %s\n", arg.c_str());
        printHelp(argv[0]);
        return false;
    }
    return true;
}

void CUDAEnsemble::initialise(int argc, const char** argv) {
    NVTX_RANGE("CUDAEnsemble::initialise");
    // Reset to defaults
    cuda_config = CConfig();
    ensemble_config = EConfig();
    // check input args
    if (argc)
        if (!checkArgs(argc, argv))
            exit(EXIT_FAILURE);
    applyConfig();
}

void CUDAEnsemble::printHelp(const char* executable) {
    printf("Usage: %s [-s steps] [-d device_id] [-r random_seed]\n", executable);
    printf("Optional Arguments:\n");
    const char *line_fmt = "%-18s %s\n";
    // printf(line_fmt, "-i, --in <file.xml/file.json>", "Initial state file (XML or JSON)");
    printf(line_fmt, "-o, --out <directory>", "Directory to store outputs");
    printf(line_fmt, "-s, --steps <steps>", "Number of simulation iterations per model instance");
    printf(line_fmt, "-i, --instances <instances>", "Total number of model instances to execute");
    printf(line_fmt, "-c, --concurrent <concurrent instances>", "Number of instances to execute concurrently");
    // printf(line_fmt, "-r, --random <seed>", "RandomManager seed");
    printf(line_fmt, "-v, --verbose", "Verbose FLAME GPU output");
    printf(line_fmt, "-t, --timing", "Output timing information to stdout");

    printf("CUDA Model Optional Arguments:\n");
    printf(line_fmt, "-d, --device", "GPU index");
}

void CUDAEnsemble::applyConfig() {
    //if (!config.input_file.empty() && config.input_file != loaded_input_file) {
    //    const std::string current_input_file = config.input_file;
    //    // Build population vector
    //    std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> pops;
    //    for (auto &agent : model->agents) {
    //        auto a = std::make_shared<AgentPopulation>(*agent.second->description);
    //        pops.emplace(agent.first, a);
    //    }
    //    StateReader *read__ = ReaderFactory::createReader(model->name, getInstanceID(), pops, config.input_file.c_str(), this);
    //    if (read__) {
    //        read__->parse();
    //        for (auto &agent : pops) {
    //            setPopulationData(*agent.second);
    //        }
    //    }
    //    // Reset input file (we don't support input file recursion)
    //    config.input_file = current_input_file;
    //    // Set flag so we don't reload this in future
    //    loaded_input_file = current_input_file;
    //}

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
    if (cuda_config.device_id >= device_count) {
        THROW InvalidCUDAdevice("Error setting CUDA device to '%d', only %d available!", cuda_config.device_id, device_count);
    }

    // Check the compute capability of the device, throw an exception if not valid for the executable.
    if (!util::compute_capability::checkComputeCapability(static_cast<int>(cuda_config.device_id))) {
        int min_cc = util::compute_capability::minimumCompiledComputeCapability();
        int cc = util::compute_capability::getComputeCapability(static_cast<int>(cuda_config.device_id));
        THROW InvalidCUDAComputeCapability("Error application compiled for CUDA Compute Capability %d and above. Device %u is compute capability %d. Rebuild for SM_%d.", min_cc, cuda_config.device_id, cc, cc);
    }

    cudaStatus = cudaSetDevice(static_cast<int>(cuda_config.device_id));
    if (cudaStatus != cudaSuccess) {
        THROW InvalidCUDAdevice("Unknown error setting CUDA device to '%d'. (%d available)", cuda_config.device_id, device_count);
    }
    // Call cudaFree to initialise the context early
    gpuErrchk(cudaFree(nullptr));

    // Apply changes to submodels
    for (auto &sm : submodel_map) {
        // We're not actually going to use this value, but it might be useful there later
        // Calling apply config a second time would reinit GPU, which might clear existing gpu allocations etc
        sm.second->CUDAConfig().device_id = cuda_config.device_id;
    }

    // Initialise singletons once a device has been selected.
    // @todo - if this has already been called, before the device was selected an error should occur.
    initialiseSingletons();

    // We init Random through submodel hierarchy after singletons
    // Currently random is fixed seed
    reseed(12);

    releaseCUDAInstances();
    initCUDAInstances(ensemble_config.max_concurrent_runs);
}


CUDAEnsemble::CUDAEnsembleInstance::CUDAEnsembleInstance(
    std::shared_ptr<const ModelData> modelddata,
    std::map<std::string, std::unique_ptr<CUDAEnsemble>> &submodels,
    RandomManager &rng,
    CUDAScatter &scatter,
    const AgentOffsetMap &agentOffsets,
    const unsigned int &_instance_id)
        : model(modelddata)
        , submodel_map(submodels)
        , instance_id(_instance_id)
{
    // populate the CUDA agent map
    const auto &am = model->agents;
    // create new cuda agent and add to the map
    for (auto it = am.cbegin(); it != am.cend(); ++it) {
        // Find matching vector and fill it with CUDAAgents
        for (int i = 0; i < active_instances; ++i) {
            agent_map.emplace(it->first, std::make_unique<CUDAAgent>(*it->second, i));//For the time being we use instance_id 0+, this is obviously unsafe
        }
    }
    // populate the CUDA message map
    const auto &mm = model->messages;
    // create new cuda message and add to the map
    for (auto it_m = mm.cbegin(); it_m != mm.cend(); ++it_m) {
        // Find matching vector and fill it with CUDAMessages
        auto _mm = message_map.find(it_m->first);
        assert(_mm != message_map.end());
        assert(_mm->second.size() == 0);
        for (int i = 0; i < active_instances; ++i) {
            message_map.emplace(it_m->first, std::make_unique<CUDAMessage>(*it_m->second));
        }
    }    
    for (auto &cm : message_map) {
        cm.second->init(scatter, 0);
    }
    // Build data
    for (const auto &agent : model->agents) {
        AgentDataBufferStateMap agent_states;
        for (const auto&state : agent.second->states)
            agent_states.emplace(state, AgentDataBuffer());
        agentData.emplace(agent.first, agent_states);
    }
    host_api = std::make_unique<FLAMEGPU_HOST_API>(*this, rng, agentOffsets, agentData);
}

void CUDAEnsemble::initCUDAInstances(const unsigned int &active_instances) {    
    // recursively forward init to submodels
    const auto &smm = model->submodels;
    // create new cuda message and add to the map
    for (auto it_sm = smm.cbegin(); it_sm != smm.cend(); ++it_sm) {
        // Find matching submodel and initialise it's submodel
        auto _sm = submodel_map.find(it_sm->first);
        assert(_sm != submodel_map.end());
        _sm->second->initCUDAInstances(active_instances);
    }

    // Build Simulation_vector (We need an AoS view to sim data too)
    for (int i = 0; i <active_instances; ++i) {
        instance_vector.push_back(CUDAEnsembleInstance(model, submodel_map, singletons->rng, singletons->scatter, agentOffsets, i));
    }
}

void CUDAEnsemble::releaseCUDAInstances() {   
    // recursively forward init to submodels
    const auto &smm = model->submodels;
    // create new cuda message and add to the map
    for (auto it_sm = smm.cbegin(); it_sm != smm.cend(); ++it_sm) {
        // Find matching submodel and initialise it's submodel
        auto _sm = submodel_map.find(it_sm->first);
        assert(_sm != submodel_map.end());
        _sm->second->releaseCUDAInstances();
    }

    instance_vector.clear();
    instance_offset_cdn_vector.clear();
}
CUDAEnsemble::InstanceOffsetData_cdn::InstanceOffsetData_cdn(const unsigned int& activeInstances)
    : d_instance_offsets(static_cast<unsigned int * const>(cmalloc((activeInstances + 1) * sizeof(unsigned int))))
    , d_instance_id_hashes(static_cast<Curve::NamespaceHash * const>(cmalloc(activeInstances * sizeof(Curve::NamespaceHash)))) {
}
CUDAEnsemble::InstanceOffsetData_cdn::~InstanceOffsetData_cdn() {
    gpuErrchk(cudaFree(d_instance_offsets)); 
    gpuErrchk(cudaFree(d_instance_id_hashes));
}
void * const CUDAEnsemble::InstanceOffsetData_cdn::cmalloc(const size_t &len) {
    void *d_t;
    gpuErrchk(cudaMalloc(&d_t, len));
    return d_t;
}
CUDAEnsemble::InstanceOffsetData::InstanceOffsetData(const unsigned int& activeInstances)
    : InstanceOffsetData_cdn(activeInstances)
    , d_in_messagelist_metadata(static_cast<const void **>(cmalloc((activeInstances) * sizeof(void*))))
    , d_out_messagelist_metadata(static_cast<const void **>(cmalloc((activeInstances) * sizeof(void*)))) {
}
CUDAEnsemble::InstanceOffsetData::~InstanceOffsetData() {
    gpuErrchk(cudaFree(d_in_messagelist_metadata));
    gpuErrchk(cudaFree(d_out_messagelist_metadata));
}
void CUDAEnsemble::resetStepCounter() {
    for (auto &instance: instance_vector) {
        instance.resetStepCounter();
    }
}
