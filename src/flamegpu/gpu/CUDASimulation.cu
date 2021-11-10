#include "flamegpu/gpu/CUDASimulation.h"

#include <curand_kernel.h>

#include <algorithm>
#include <string>

#include "flamegpu/model/AgentFunctionData.cuh"
#include "flamegpu/model/LayerData.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/model/SubAgentData.h"
#include "flamegpu/runtime/HostAPI.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"
#include "flamegpu/util/nvtx.h"
#include "flamegpu/util/detail/compute_capability.cuh"
#include "flamegpu/util/detail/SignalHandlers.h"
#include "flamegpu/util/detail/wddm.cuh"
#include "flamegpu/util/detail/SteadyClockTimer.h"
#include "flamegpu/util/detail/CUDAEventTimer.cuh"
#include "flamegpu/runtime/detail/curve/curve_rtc.cuh"
#include "flamegpu/runtime/HostFunctionCallback.h"
#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/sim/LoggingConfig.h"
#include "flamegpu/sim/LogFrame.h"
#ifdef VISUALISATION
#include "flamegpu/visualiser/FLAMEGPU_Visualisation.h"
#endif

namespace flamegpu {

namespace {
    // file-scope only variable used to cache the driver mode
    bool deviceUsingWDDM = false;
    // Inlined method in the anonymous namespace to create a new timer, subject to the driver model.
    std::unique_ptr<util::detail::Timer> getDriverAppropriateTimer() {
        if (!deviceUsingWDDM) {
            return std::unique_ptr<util::detail::Timer>(new util::detail::CUDAEventTimer());
        } else {
            return std::unique_ptr<util::detail::Timer>(new util::detail::SteadyClockTimer());
        }
    }
}  // anonymous namespace

std::map<int, std::atomic<int>> CUDASimulation::active_device_instances;
std::map<int, std::shared_timed_mutex> CUDASimulation::active_device_mutex;
std::shared_timed_mutex CUDASimulation::active_device_maps_mutex;
std::atomic<int> CUDASimulation::active_instances = {0};
bool CUDASimulation::AUTO_CUDA_DEVICE_RESET = true;

CUDASimulation::CUDASimulation(const ModelDescription& _model, int argc, const char** argv)
    : CUDASimulation(_model.model) {
    if (argc && argv) {
        initialise(argc, argv);
    }
}
CUDASimulation::CUDASimulation(const std::shared_ptr<const ModelData> &_model)
    : Simulation(_model)
    , step_count(0)
    , elapsedSecondsSimulation(0.)
    , elapsedSecondsInitFunctions(0.)
    , elapsedSecondsExitFunctions(0.)
    , elapsedSecondsRTCInitialisation(0.)
    , macro_env(*_model->environment, *this)
    , run_log(std::make_unique<RunLog>())
    , streams(std::vector<cudaStream_t>())
    , sortAgentsEveryNSteps(10)
    , singletons(nullptr)
    , singletonsInitialised(false)
    , rtcInitialised(false)
    , isPureRTC(detectPureRTC(model)) {
    ++active_instances;
    initOffsetsAndMap();
    // Register the signal handler.
    util::detail::SignalHandlers::registerSignalHandlers();

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
}
bool CUDASimulation::detectPureRTC(const std::shared_ptr<const ModelData>& _model) {
    const auto& am = _model->agents;
    for (auto it = am.cbegin(); it != am.cend(); ++it) {
        for (const auto& af : it->second->functions) {
            if (af.second->func || af.second->condition)
                return false;
        }
    }
    // TODO: In future this will need to be extended for new forms of device function, e.g. device init
    const auto& as = _model->submodels;
    for (auto it = as.cbegin(); it != as.cend(); ++it) {
        if (!detectPureRTC(it->second->submodel))
          return false;
    }
    return true;
}
CUDASimulation::CUDASimulation(const std::shared_ptr<SubModelData> &submodel_desc, CUDASimulation *master_model)
    : Simulation(submodel_desc, master_model)
    , step_count(0)
    , macro_env(*submodel_desc->submodel->environment, *this)
    , run_log(std::make_unique<RunLog>())
    , streams(std::vector<cudaStream_t>())
    , singletons(nullptr)
    , singletonsInitialised(false)
    , rtcInitialised(false)
    , isPureRTC(master_model->isPureRTC) {
    ++active_instances;
    initOffsetsAndMap();
    // Ensure submodel is valid
    if (submodel_desc->submodel->exitConditions.empty() && submodel_desc->submodel->exitConditionCallbacks.empty() && submodel_desc->max_steps == 0) {
        THROW exception::InvalidSubModel("Model '%s' does not contain any exit conditions or exit condition callbacks and submodel '%s' max steps is set to 0, SubModels must exit of their own accord, "
            "in CUDASimulation::CUDASimulation().",
            submodel_desc->submodel->name.c_str(), submodel_desc->name.c_str());
    }

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
                THROW exception::InvalidParent("Master agent description has expired, in CUDASimulation SubModel constructor.\n");
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
    // Submodels all run quiet/not verbose by default
    SimulationConfig().verbose = false;
    SimulationConfig().steps = submodel_desc->max_steps;
}

CUDASimulation::~CUDASimulation() {
    // Ensure we destruct with the right device, otherwise we could dealloc pointers on the wrong device
    int t_device_id = -1;
    gpuErrchk(cudaGetDevice(&t_device_id));
    if (t_device_id != deviceInitialised && deviceInitialised != -1) {
        gpuErrchk(cudaSetDevice(deviceInitialised));
    }

    submodel_map.clear();  // Test
    // De-initialise, freeing singletons?
    // @todo - this is unsafe in a destructor as it may invoke cuda commands.
    if (singletonsInitialised) {
        // unique pointers cleanup by automatically
        // Drop all constants from the constant cache linked to this model
        singletons->environment.free(singletons->curve, instance_id);
        // if (active_instances == 1) {
        //   assert(singletons->curve.size() == 0);
        // }

        delete singletons;
        singletons = nullptr;
    }

    // Destroy streams, potentially unsafe in a destructor as it will invoke cuda commands.
    // Do this once to re-use existing streams rather than per-step.
    this->destroyStreams();

    // We must explicitly delete all cuda members before we cuda device reset
    agent_map.clear();
    message_map.clear();
    submodel_map.clear();
    host_api.reset();
    macro_env.free();
#ifdef VISUALISATION
    visualisation.reset();
#endif
    // If we are the last instance to destruct
    // This doesn't really play nicely if we are passing multi-device CUDASimulations between threads!
    // I think this exists to prevent curve getting left with dead items when exceptions are thrown during the test suite.
    if (deviceInitialised >= 0 && AUTO_CUDA_DEVICE_RESET) {
        std::shared_lock<std::shared_timed_mutex> maps_lock(active_device_maps_mutex);
        std::unique_lock<std::shared_timed_mutex> lock(active_device_mutex.at(deviceInitialised));
        if (!--active_device_instances.at(deviceInitialised)) {
            // Small chance that time between the atomic and body of this fn will cause a problem
            // Could mutex it with init simulation cuda stuff, but really seems unlikely
            gpuErrchk(cudaDeviceReset());
            EnvironmentManager::getInstance().purge();
            detail::curve::Curve::getInstance().purge();
        }
    }
    if (t_device_id != deviceInitialised) {
        gpuErrchk(cudaSetDevice(t_device_id));
    }
    --active_instances;
}



void CUDASimulation::initFunctions() {
    NVTX_RANGE("CUDASimulation::initFunctions");
    std::unique_ptr<util::detail::Timer> initFunctionsTimer(new util::detail::SteadyClockTimer());
    initFunctionsTimer->start();

    // Execute normal init functions
    for (auto &initFn : model->initFunctions) {
        initFn(this->host_api.get());
    }
    // Execute init function callbacks (python)
    for (auto &initFn : model->initFunctionCallbacks) {
        initFn->run(this->host_api.get());
    }
    // Check if host agent creation was used in init functions
    if (model->initFunctions.size() || model->initFunctionCallbacks.size()) {
        processHostAgentCreation(0);
    }

    // Record, store and output the elapsed time of the step.
    initFunctionsTimer->stop();
    this->elapsedSecondsInitFunctions = initFunctionsTimer->getElapsedSeconds();
    if (getSimulationConfig().timing) {
        fprintf(stdout, "Init Function Processing time: %.6f s\n", this->elapsedSecondsInitFunctions);
    }
}

void CUDASimulation::exitFunctions() {
    NVTX_RANGE("CUDASimulation::exitFunctions");
    std::unique_ptr<util::detail::Timer> exitFunctionsTimer(new util::detail::SteadyClockTimer());
    exitFunctionsTimer->start();

    // Execute exit functions
    for (auto &exitFn : model->exitFunctions) {
        exitFn(this->host_api.get());
    }
    // Execute any exit functions from swig/python
    for (auto &exitFn : model->exitFunctionCallbacks) {
        exitFn->run(this->host_api.get());
    }

    // Record, store and output the elapsed time of the step.
    exitFunctionsTimer->stop();
    this->elapsedSecondsExitFunctions = exitFunctionsTimer->getElapsedSeconds();
    if (getSimulationConfig().timing) {
        fprintf(stdout, "Exit Function Processing time: %.6f s\n", this->elapsedSecondsExitFunctions);
    }
}

void CUDASimulation::setSortAgentsEveryNSteps(const unsigned int n) {
    sortAgentsEveryNSteps = n;
}

__global__ void spatialSortInitToThreadIndex(unsigned int *output, unsigned int threadCount) {
    const unsigned int TID = blockIdx.x * blockDim.x + threadIdx.x;
    if (TID < threadCount) {
        output[TID] = TID;
    }
}

namespace detail {
template <typename T> struct Dims {
    T x;
    T y;
    T z;
};
}

__global__ void calculateSpatialHash(float* x, float* y, float* z, unsigned int* binIndex, detail::Dims<float> envMin, detail::Dims<float> envWidth, detail::Dims<unsigned int> gridDim, unsigned int threadCount) {
    const unsigned int TID = blockIdx.x * blockDim.x + threadIdx.x;
    if (TID < threadCount) {
        // Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
        int gridPos[3] = {
            static_cast<int>(floorf(((x[TID]-envMin.x) / envWidth.x)*gridDim.x)),
            static_cast<int>(floorf(((y[TID]-envMin.y) / envWidth.y)*gridDim.y)),
            0
        };

        // If 3D, set 3rd component
        if (z) {
            gridPos[2] = static_cast<int>(floorf(((z[TID]-envMin.z) / envWidth.z)*gridDim.z));
        }

        // Compute and set the bin index
        unsigned int bindex;

        if (z) {
            bindex = (unsigned int)(
            (gridPos[2] * gridDim.x * gridDim.y +   // z
            (gridPos[1] * gridDim.x) +              // y
            gridPos[0]));                           // x

        } else {
            bindex = (unsigned int)(
            (gridPos[1] * gridDim.x) +              // y
            gridPos[0]);                            // x
        }

        binIndex[TID] = bindex;
    }
}

void CUDASimulation::determineAgentsToSort() {
    const auto& am = model->agents;

    // Iterate agents and then agent functions to find any functions which use spatial messaging
    for (auto it = am.cbegin(); it != am.cend(); ++it) {
        const auto& mf = it->second->functions;
        for (auto it_f = mf.cbegin(); it_f != mf.cend(); ++it_f) {
            if (auto ptr = it_f->second->message_input.lock()) {
                // Check if this agent function uses 3D spatial messages
                if (ptr->getSortingType() == flamegpu::MessageSortingType::spatial3D) {
                    // Agent uses spatial, check it has correct variables
                    const auto& ad = *(it->second->description);
                    if (ad.hasVariable("x") && ad.hasVariable("y") && ad.hasVariable("z")) {
                        sortTriggers3D.insert(it_f->first);
                    }
                }

                // Check if this agent function uses 2D spatial messages
                if (ptr->getSortingType() == flamegpu::MessageSortingType::spatial2D) {
                    // Agent uses spatial, check it has correct variables
                    const auto& ad = *(it->second->description);
                    if (ad.hasVariable("x") && ad.hasVariable("y")) {
                        sortTriggers2D.insert(it_f->first);
                    }
                }
            }
        }
    }
}


void CUDASimulation::spatialSortAgent(const std::string& agentName, const std::string& state, const int mode) {
    float radius;
    detail::Dims<float> envMin;
    detail::Dims<float> envMax;
    detail::Dims<float> envWidth;
    detail::Dims<unsigned int> gridDim;

    try {
        radius = host_api->environment.getProperty<float>("INTERACTION_RADIUS");
        envMin = {host_api->environment.getProperty<float>("MIN_POSITION"), host_api->environment.getProperty<float>("MIN_POSITION"), host_api->environment.getProperty<float>("MIN_POSITION")};
        envMax = {host_api->environment.getProperty<float>("MAX_POSITION"), host_api->environment.getProperty<float>("MAX_POSITION"), host_api->environment.getProperty<float>("MAX_POSITION")};
        envWidth = {(envMax.x-envMin.x), (envMax.y-envMin.y), (envMax.z-envMin.z)};
        gridDim = {static_cast<unsigned int>(ceilf(envWidth.x / radius)), static_cast<unsigned int>(ceilf(envWidth.y / radius)), static_cast<unsigned int>(ceilf(envWidth.z / radius))};
    } catch (exception::InvalidEnvProperty& e) {
        std::cout << "WARNING: Please set the INTERACTION_RADIUS, MIN_POSITION and MAX_POSITION environment properties to enable spatial sorting\n";
        this->setSortAgentsEveryNSteps(0);
        return;
    }

    CUDAAgent& cuda_agent = getCUDAAgent(agentName);

    // Any agent in this list is guaranteed to have x, y, z and fgpu2_reserved_bin_index vars - used in the computation of spatial hash
    // TODO: User could supply alternatives to "x", "y", "z" to use alternative variables?
    void* xPtr = cuda_agent.getStateVariablePtr(state, "x");
    void* yPtr = cuda_agent.getStateVariablePtr(state, "y");
    void* zPtr = mode == Agent3D ? cuda_agent.getStateVariablePtr(state, "z") : 0;
    void* binIndexPtr = cuda_agent.getStateVariablePtr(state, "fgpu2_reserved_bin_index");

    // Compute occupancy
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size
    const unsigned int state_list_size = cuda_agent.getStateSize(state);
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateSpatialHash, 0, state_list_size);

    //! Round up according to CUDAAgent state list size
    gridSize = (state_list_size + blockSize - 1) / blockSize;

    unsigned int sm_size = 0;
    unsigned int streamIdx = 0;
#if !defined(SEATBELTS) || SEATBELTS
    auto *error_buffer = this->singletons->exception.getDevicePtr(streamIdx, this->getStream(streamIdx));
    sm_size = sizeof(error_buffer);
#endif

    // Launch kernel
    calculateSpatialHash<<<gridSize, blockSize, sm_size, this->getStream(streamIdx) >>> (reinterpret_cast<float*>(xPtr),
    reinterpret_cast<float*>(yPtr),
    reinterpret_cast<float*>(zPtr),
    reinterpret_cast<unsigned int*>(binIndexPtr),
    envMin,
    envWidth,
    gridDim,
    state_list_size);

    assert(host_api);
    host_api->agent(agentName).sort<unsigned int>("fgpu2_reserved_bin_index", HostAgentAPI::Asc);
}

bool CUDASimulation::step() {
    NVTX_RANGE(std::string("CUDASimulation::step " + std::to_string(step_count)).c_str());
    // Ensure singletons have been initialised
    initialiseSingletons();

    // Time the individual step, using a CUDAEventTimer if possible, else a steadyClockTimer.
    std::unique_ptr<util::detail::Timer> stepTimer = getDriverAppropriateTimer();
    stepTimer->start();

    // Init any unset agent IDs
    this->assignAgentIDs();

    // If verbose, print the step number.
    if (getSimulationConfig().verbose) {
        fprintf(stdout, "Processing Simulation Step %u\n", step_count);
    }


    // Ensure there are enough streams to execute the layer.
    // Taking into consideration if in-layer concurrency is disabled or not.
    unsigned int nStreams = getMaximumLayerWidth();
    this->createStreams(nStreams);

    // Reset message list flags
    for (auto m =  message_map.begin(); m != message_map.end(); ++m) {
        m->second->setTruncateMessageListFlag();
    }

    // Execute each layer of the simulation.
    unsigned int layerIndex = 0;
    for (auto& layer : model->layers) {
        // Execute the individual layer
        stepLayer(layer, layerIndex);
        // Increment counter
        ++layerIndex;
    }

    // Run the step functions (including pyhton.)
    stepStepFunctions();

    // Run the exit conditons, detecting wheter or not any we
    bool exitRequired = this->stepExitConditions();

    // Record, store and output the elapsed time of the step.
    stepTimer->stop();
    float stepMilliseconds = stepTimer->getElapsedSeconds();
    this->elapsedSecondsPerStep.push_back(stepMilliseconds);
    if (getSimulationConfig().timing) {
        // Resolution is 0.5 microseconds, so print to 1 us.
        fprintf(stdout, "Step %d Processing time: %.6f s\n", this->step_count, stepMilliseconds);
    }

    // Update step count at the end of the step - when it has completed.
    incrementStepCounter();
    // Update the log for the step.
    processStepLog();
    // Return false if any exit condition's passed.
    return !exitRequired;
}

void CUDASimulation::stepLayer(const std::shared_ptr<LayerData>& layer, const unsigned int layerIndex) {
    NVTX_RANGE(std::string("stepLayer " + std::to_string(layerIndex)).c_str());

    std::string message_name;

    // If the layer contains a sub model, it can only execute the sub model.
    if (layer->sub_model) {
        auto &sm = submodel_map.at(layer->sub_model->name);
        sm->resetStepCounter();
        sm->simulate();
        sm->reset(true);
        // Next layer, this layer cannot also contain agent functions
        // Ensure syncrhonisation has occured.
        this->synchronizeAllStreams();
        return;
    }

    // Track stream index
    int streamIdx = 0;
    // Sum the total number of threads being launched in the layer
    unsigned int totalThreads = 0;

    // Spatially sort the agents
    if ((sortAgentsEveryNSteps != 0) && (step_count % sortAgentsEveryNSteps == 0)) {
        for (const auto &func_des : layer->agent_functions) {
            auto func_agent = func_des->parent.lock();
            if (sortTriggers3D.find(func_des->name) != sortTriggers3D.end()) {
                this->spatialSortAgent(func_agent->name, func_des->initial_state, Agent3D);
            }
            if (sortTriggers2D.find(func_des->name) != sortTriggers2D.end()) {
                this->spatialSortAgent(func_agent->name, func_des->initial_state, Agent2D);
            }
        }
    }

    // Map agent memory
    bool has_rtc_func_cond = false;
    for (const auto &func_des : layer->agent_functions) {
        if ((func_des->condition) || (!func_des->rtc_func_condition_name.empty())) {
            auto func_agent = func_des->parent.lock();
            NVTX_RANGE(std::string("condition map " + func_agent->name + "::" + func_des->name).c_str());
            const CUDAAgent& cuda_agent = getCUDAAgent(func_agent->name);

            const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
            if (state_list_size == 0) {
                ++streamIdx;
                continue;
            }
            singletons->scatter.Scan().resize(state_list_size, CUDAScanCompaction::AGENT_DEATH, streamIdx);

            // Configure runtime access of the functions variables within the FLAME_API object
            cuda_agent.mapRuntimeVariables(*func_des, instance_id);

            // Zero the scan flag that will be written to
            singletons->scatter.Scan().zero(CUDAScanCompaction::AGENT_DEATH, streamIdx);  // @todo - stream

            // Push function's RTC cache to device if using RTC
            if (!func_des->rtc_func_condition_name.empty()) {
                has_rtc_func_cond = true;
                std::string func_name = func_des->name + "_condition";
                auto &rtc_header = cuda_agent.getRTCHeader(func_name);
                // Sync EnvManager's RTC cache with RTC header's cache
                rtc_header.updateEnvCache(singletons->environment.getRTCCache(instance_id));
                // Push RTC header's cache to device
                rtc_header.updateDevice(cuda_agent.getRTCInstantiation(func_name));
            }

            totalThreads += state_list_size;
            ++streamIdx;
        }
    }

    // If any condition kernel needs to be executed, do so, by checking the number of threads from before.
    if (totalThreads > 0) {
        auto env_shared_lock = this->singletons->environment.getSharedLock();
        auto env_device_lock = this->singletons->environment.getDeviceSharedLock();
        if (!has_rtc_func_cond) {
            this->singletons->environment.updateDevice(instance_id);
            this->singletons->curve.updateDevice();

            // this->synchronizeAllStreams();  // Not required, the above is snchronizing.
        }

        // Ensure RandomManager is the correct size to accommodate all threads to be launched
        curandState *d_rng = singletons->rng.resize(totalThreads);  // @todo - stream + sync.
        // Track which stream to use for concurrency
        streamIdx = 0;
        // Sum the total number of threads being launched in the layer, for rng offsetting.
        totalThreads = 0;
        // Launch function condition kernels
        for (const auto &func_des : layer->agent_functions) {
            if ((func_des->condition) || (!func_des->rtc_func_condition_name.empty())) {
                auto func_agent = func_des->parent.lock();
                NVTX_RANGE(std::string("condition " + func_agent->name + "::" + func_des->name).c_str());
                if (!func_agent) {
                    THROW exception::InvalidAgentFunc("Agent function condition refers to expired agent.");
                }
                std::string agent_name = func_agent->name;
                std::string func_name = func_des->name;

                const CUDAAgent& cuda_agent = getCUDAAgent(agent_name);

                const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
                if (state_list_size == 0) {
                    ++streamIdx;
                    continue;
                }

                int blockSize = 0;  // The launch configurator returned block size
                int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
                int gridSize = 0;  // The actual grid size needed, based on input size

                //  Agent function condition kernel wrapper args
                detail::curve::Curve::NamespaceHash agentname_hash = detail::curve::Curve::variableRuntimeHash(agent_name.c_str());
                detail::curve::Curve::NamespaceHash funcname_hash = detail::curve::Curve::variableRuntimeHash(func_name.c_str());
                detail::curve::Curve::NamespaceHash agent_func_name_hash = agentname_hash + funcname_hash + instance_id;
                curandState *t_rng = d_rng + totalThreads;
                unsigned int *scanFlag_agentDeath = this->singletons->scatter.Scan().Config(CUDAScanCompaction::Type::AGENT_DEATH, streamIdx).d_ptrs.scan_flag;
                unsigned int sm_size = 0;
#if !defined(SEATBELTS) || SEATBELTS
                auto *error_buffer = this->singletons->exception.getDevicePtr(streamIdx, this->getStream(streamIdx));
                sm_size = sizeof(error_buffer);
#endif
                // switch between normal and RTC agent function condition
                if (func_des->condition) {
                    // calculate the grid block size for agent function condition
                    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func_des->condition, 0, state_list_size);

                    //! Round up according to CUDAAgent state list size
                    gridSize = (state_list_size + blockSize - 1) / blockSize;
                    (func_des->condition) << <gridSize, blockSize, sm_size, this->getStream(streamIdx) >> > (
#if !defined(SEATBELTS) || SEATBELTS
                    error_buffer,
#endif
                    instance_id,
                    agent_func_name_hash,
                    state_list_size,
                    t_rng,
                    scanFlag_agentDeath);
                    gpuErrchkLaunch();
                } else {  // RTC function
                    std::string func_condition_identifier = func_name + "_condition";
                    // get instantiation
                    const jitify::experimental::KernelInstantiation& instance = cuda_agent.getRTCInstantiation(func_condition_identifier);
                    // calculate the grid block size for main agent function
                    CUfunction cu_func = (CUfunction)instance;
                    cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cu_func, 0, 0, state_list_size);
                    //! Round up according to CUDAAgent state list size
                    gridSize = (state_list_size + blockSize - 1) / blockSize;
                    // launch the kernel
                    CUresult a = instance.configure(gridSize, blockSize, sm_size, this->getStream(streamIdx)).launch({
#if !defined(SEATBELTS) || SEATBELTS
                        reinterpret_cast<void*>(&error_buffer),
#endif
                        const_cast<void*>(reinterpret_cast<const void*>(&instance_id)),
                        reinterpret_cast<void*>(&agent_func_name_hash),
                        const_cast<void *>(reinterpret_cast<const void*>(&state_list_size)),
                        reinterpret_cast<void*>(&t_rng),
                        reinterpret_cast<void*>(&scanFlag_agentDeath) });
                    if (a != CUresult::CUDA_SUCCESS) {
                        const char* err_str = nullptr;
                        cuGetErrorString(a, &err_str);
                        THROW exception::InvalidAgentFunc("There was a problem launching the runtime agent function condition '%s': %s", func_des->rtc_func_condition_name.c_str(), err_str);
                    }
                    gpuErrchkLaunch();
                }

                totalThreads += state_list_size;
                ++streamIdx;
            }
        }

        // Ensure that each condition function has finished before unlocking the environment
        // Potentially there might be performance gains within a model by moving this until after the unmapping, although this may block other threads
        this->synchronizeAllStreams();
        env_shared_lock.unlock();
        env_device_lock.unlock();
    }

    // Track stream index
    streamIdx = 0;
    // Unmap agent memory, apply condition.
    for (const auto &func_des : layer->agent_functions) {
        if ((func_des->condition) || (!func_des->rtc_func_condition_name.empty())) {
            auto func_agent = func_des->parent.lock();
            if (!func_agent) {
                THROW exception::InvalidAgentFunc("Agent function condition refers to expired agent.");
            }
            NVTX_RANGE(std::string("condition unmap " + func_agent->name + "::" + func_des->name).c_str());
            CUDAAgent& cuda_agent = getCUDAAgent(func_agent->name);

            // Skip if no agents in the input state
            const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
            if (state_list_size == 0) {
                ++streamIdx;
                continue;
            }

            // unmap the function variables
            cuda_agent.unmapRuntimeVariables(*func_des, instance_id);
#if !defined(SEATBELTS) || SEATBELTS
            // Error check after unmap vars
            this->singletons->exception.checkError("condition " + func_des->name, streamIdx, this->getStream(streamIdx));
#endif
            // Process agent function condition
            cuda_agent.processFunctionCondition(*func_des, this->singletons->scatter, streamIdx, this->getStream(streamIdx));
            // Increment the stream tracker.
            ++streamIdx;
        }
    }

    bool has_rtc_func = false;
    streamIdx = 0;
    // Sum the total number of threads being launched in the layer
    totalThreads = 0;
    // for each func function - Loop through to do all mapping of agent and message variables
    for (const auto &func_des : layer->agent_functions) {
        auto func_agent = func_des->parent.lock();
        if (!func_agent) {
            THROW exception::InvalidAgentFunc("Agent function refers to expired agent.");
        }
        NVTX_RANGE(std::string("map" + func_agent->name + "::" + func_des->name).c_str());

        const CUDAAgent& cuda_agent = getCUDAAgent(func_agent->name);
        const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
        if (state_list_size == 0) {
            ++streamIdx;
            continue;
        }
        // Resize death flag array if necessary
        singletons->scatter.Scan().resize(state_list_size, CUDAScanCompaction::AGENT_DEATH, streamIdx);

        // check if a function has an input message
        if (auto im = func_des->message_input.lock()) {
            std::string inpMessage_name = im->name;
            CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name);
            // Construct PBM here if required!!
            cuda_message.buildIndex(this->singletons->scatter, streamIdx, this->getStream(streamIdx));  // This is synchronous.
            // Map variables after, as index building can swap arrays
            cuda_message.mapReadRuntimeVariables(*func_des, cuda_agent, instance_id);
        }

        // check if a function has an output message
        if (auto om = func_des->message_output.lock()) {
            std::string outpMessage_name = om->name;
            CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);
            // Resize message list if required
            const unsigned int existingMessages = cuda_message.getTruncateMessageListFlag() ? 0 : cuda_message.getMessageCount();
            cuda_message.resize(existingMessages + state_list_size, this->singletons->scatter, streamIdx);
            cuda_message.mapWriteRuntimeVariables(*func_des, cuda_agent, state_list_size, instance_id);
            singletons->scatter.Scan().resize(state_list_size, CUDAScanCompaction::MESSAGE_OUTPUT, streamIdx);
            // Zero the scan flag that will be written to
            if (func_des->message_output_optional)
                singletons->scatter.Scan().zero(CUDAScanCompaction::MESSAGE_OUTPUT, streamIdx);  // @todo - do this in a stream?
        }

        // check if a function has an output agent
        if (auto oa = func_des->agent_output.lock()) {
            // This will act as a reserve word
            // which is added to variable hashes for agent creation on device
            CUDAAgent& output_agent = getCUDAAgent(oa->name);

            // Map vars with curve (this allocates/requests enough new buffer space if an existing version is not available/suitable)
            output_agent.mapNewRuntimeVariables(cuda_agent, *func_des, state_list_size, this->singletons->scatter, instance_id, streamIdx);  // @todo - stream?
        }

        // Configure runtime access of the functions variables within the FLAME_API object
        cuda_agent.mapRuntimeVariables(*func_des, instance_id);

        // Zero the scan flag that will be written to
        if (func_des->has_agent_death) {
            singletons->scatter.Scan().CUDAScanCompaction::zero(CUDAScanCompaction::AGENT_DEATH, streamIdx);  // @todo stream?
        }

        // Push function's RTC cache to device if using RTC
        if (!func_des->rtc_func_name.empty()) {
            has_rtc_func = true;
            auto& rtc_header = cuda_agent.getRTCHeader(func_des->name);
            // Sync EnvManager's RTC cache with RTC header's cache
            rtc_header.updateEnvCache(singletons->environment.getRTCCache(instance_id));
            // Push RTC header's cache to device
            rtc_header.updateDevice(cuda_agent.getRTCInstantiation(func_des->name));
        }

        // Count total threads being launched
        totalThreads += cuda_agent.getStateSize(func_des->initial_state);
        ++streamIdx;
    }

    // If any condition kernel needs to be executed, do so, by checking the number of threads from before.
    if (totalThreads > 0) {
        auto env_shared_lock = this->singletons->environment.getSharedLock();
        auto env_device_lock = this->singletons->environment.getDeviceSharedLock();
        if (!has_rtc_func) {
            this->singletons->environment.updateDevice(instance_id);
            this->singletons->curve.updateDevice();
            this->synchronizeAllStreams();  // This is not strictly required as updateDevice is synchronous.
        }

        // Ensure RandomManager is the correct size to accommodate all threads to be launched
        curandState *d_rng = singletons->rng.resize(totalThreads);
        // Total threads is now used to provide kernel launches an offset to thread-safe thread-index
        totalThreads = 0;
        streamIdx = 0;

        // for each func function - Loop through to launch all agent functions
        for (const auto &func_des : layer->agent_functions) {
            auto func_agent = func_des->parent.lock();
            if (!func_agent) {
                THROW exception::InvalidAgentFunc("Agent function refers to expired agent.");
            }
            NVTX_RANGE(std::string(func_agent->name + "::" + func_des->name).c_str());
            const void *d_in_messagelist_metadata = nullptr;
            const void *d_out_messagelist_metadata = nullptr;
            id_t *d_agentOut_nextID = nullptr;
            std::string agent_name = func_agent->name;
            std::string func_name = func_des->name;
            detail::curve::Curve::NamespaceHash agentname_hash = detail::curve::Curve::variableRuntimeHash(agent_name.c_str());
            detail::curve::Curve::NamespaceHash funcname_hash = detail::curve::Curve::variableRuntimeHash(func_name.c_str());
            detail::curve::Curve::NamespaceHash agent_func_name_hash = agentname_hash + funcname_hash + instance_id;
            detail::curve::Curve::NamespaceHash message_name_inp_hash = 0;
            detail::curve::Curve::NamespaceHash message_name_outp_hash = 0;
            detail::curve::Curve::NamespaceHash agentoutput_hash = 0;

            // check if a function has an input message
            if (auto im = func_des->message_input.lock()) {
                std::string inpMessage_name = im->name;
                const CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name);

                // hash message name
                message_name_inp_hash = detail::curve::Curve::variableRuntimeHash(inpMessage_name.c_str());

                d_in_messagelist_metadata = cuda_message.getMetaDataDevicePtr();
            }

            // check if a function has an output message
            if (auto om = func_des->message_output.lock()) {
                std::string outpMessage_name = om->name;
                const CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);

                // hash message name
                message_name_outp_hash =  detail::curve::Curve::variableRuntimeHash(outpMessage_name.c_str());
                d_out_messagelist_metadata = cuda_message.getMetaDataDevicePtr();
            }

            // check if a function has agent output
            if (auto oa = func_des->agent_output.lock()) {
                agentoutput_hash = (detail::curve::Curve::variableRuntimeHash("_agent_birth") ^ funcname_hash) + instance_id;
                CUDAAgent& output_agent = getCUDAAgent(oa->name);
                d_agentOut_nextID = output_agent.getDeviceNextID();
            }

            const CUDAAgent& cuda_agent = getCUDAAgent(agent_name);

            const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
            if (state_list_size == 0) {
                ++streamIdx;
                continue;
            }

            int blockSize = 0;  // The launch configurator returned block size
            int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
            int gridSize = 0;  // The actual grid size needed, based on input size

            // Agent function kernel wrapper args
            curandState * t_rng = d_rng + totalThreads;
            unsigned int *scanFlag_agentDeath = func_des->has_agent_death ? this->singletons->scatter.Scan().Config(CUDAScanCompaction::Type::AGENT_DEATH, streamIdx).d_ptrs.scan_flag : nullptr;
            unsigned int *scanFlag_messageOutput = this->singletons->scatter.Scan().Config(CUDAScanCompaction::Type::MESSAGE_OUTPUT, streamIdx).d_ptrs.scan_flag;
            unsigned int *scanFlag_agentOutput = this->singletons->scatter.Scan().Config(CUDAScanCompaction::Type::AGENT_OUTPUT, streamIdx).d_ptrs.scan_flag;
            unsigned int sm_size = 0;
    #if !defined(SEATBELTS) || SEATBELTS
            auto *error_buffer = this->singletons->exception.getDevicePtr(streamIdx, this->getStream(streamIdx));
            sm_size = sizeof(error_buffer);
    #endif

            if (func_des->func) {   // compile time specified agent function launch
                // calculate the grid block size for main agent function
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func_des->func, 0, state_list_size);
                //! Round up according to CUDAAgent state list size
                gridSize = (state_list_size + blockSize - 1) / blockSize;

                (func_des->func) << <gridSize, blockSize, sm_size, this->getStream(streamIdx) >> > (
    #if !defined(SEATBELTS) || SEATBELTS
                    error_buffer,
    #endif
                    instance_id,
                    agent_func_name_hash,
                    message_name_inp_hash,
                    message_name_outp_hash,
                    agentoutput_hash,
                    d_agentOut_nextID,
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
                const jitify::experimental::KernelInstantiation& instance = cuda_agent.getRTCInstantiation(func_name);
                // calculate the grid block size for main agent function
                CUfunction cu_func = (CUfunction)instance;
                cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cu_func, 0, 0, state_list_size);
                //! Round up according to CUDAAgent state list size
                gridSize = (state_list_size + blockSize - 1) / blockSize;
                // launch the kernel
                CUresult a = instance.configure(gridSize, blockSize, sm_size, this->getStream(streamIdx)).launch({
#if !defined(SEATBELTS) || SEATBELTS
                    reinterpret_cast<void*>(&error_buffer),
#endif
                    const_cast<void*>(reinterpret_cast<const void*>(&instance_id)),
                    reinterpret_cast<void*>(&agent_func_name_hash),
                    reinterpret_cast<void*>(&message_name_inp_hash),
                    reinterpret_cast<void*>(&message_name_outp_hash),
                    reinterpret_cast<void*>(&agentoutput_hash),
                    reinterpret_cast<void*>(&d_agentOut_nextID),
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
                    THROW exception::InvalidAgentFunc("There was a problem launching the runtime agent function '%s': %s", func_name.c_str(), err_str);
                }
                gpuErrchkLaunch();
            }
            totalThreads += state_list_size;
            ++streamIdx;
        }

        // Ensure that each stream of work has finished before releasing the environment lock.
        this->synchronizeAllStreams();
        env_shared_lock.unlock();
        env_device_lock.unlock();
    }

    streamIdx = 0;
    // for each func function - Loop through to un-map all agent and message variables
    for (const auto &func_des : layer->agent_functions) {
        auto func_agent = func_des->parent.lock();
        if (!func_agent) {
            THROW exception::InvalidAgentFunc("Agent function refers to expired agent.");
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
                cuda_message.unmapRuntimeVariables(*func_des, instance_id);
            }

            // check if a function has an output message
            if (auto om = func_des->message_output.lock()) {
                std::string outpMessage_name = om->name;
                CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);
                cuda_message.unmapRuntimeVariables(*func_des, instance_id);
                cuda_message.swap(func_des->message_output_optional, state_list_size, this->singletons->scatter, streamIdx);
                cuda_message.clearTruncateMessageListFlag();
                cuda_message.setPBMConstructionRequiredFlag();
            }

            // Process agent death (has agent death check is handled by the method)
            // This MUST occur before agent_output, as if agent_output triggers resize then scan_flag for death will be purged
            cuda_agent.processDeath(*func_des, this->singletons->scatter, streamIdx, this->getStream(streamIdx));

            // Process agent state transition (Longer term merge this with process death?)
            cuda_agent.transitionState(func_des->initial_state, func_des->end_state, this->singletons->scatter, streamIdx, this->getStream(streamIdx));
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
                output_agent.scatterNew(*func_des, state_list_size, this->singletons->scatter, streamIdx, this->getStream(streamIdx));
                // unmap vars with curve
                output_agent.unmapNewRuntimeVariables(*func_des, instance_id);
            }

            // unmap the function variables
            cuda_agent.unmapRuntimeVariables(*func_des, instance_id);
#if !defined(SEATBELTS) || SEATBELTS
            // Error check after unmap vars
            // This means that curve is cleaned up before we throw exception (mostly prevents curve being polluted if we catch and handle errors)
            this->singletons->exception.checkError(func_des->name, streamIdx, this->getStream(streamIdx));
#endif
        }

        ++streamIdx;
    }

    // Synchronise to ensure that device memory is in a goood state prior to host layer functions? This can potentially be removed
    this->synchronizeAllStreams();

    // Execute the host functions.
    layerHostFunctions(layer, layerIndex);

#if !defined(SEATBELTS) || SEATBELTS
    // Reset macro-environment read-write flags
    // Note this does not synchronise threads, it relies on synchronizeAllStreams() post host fns
    macro_env.resetFlagsAsync(streams);
#endif

    // Synchronise  after the host layer functions to ensure that the device is up to date? This can potentially be removed.
    this->synchronizeAllStreams();
}

void CUDASimulation::layerHostFunctions(const std::shared_ptr<LayerData>& layer, const unsigned int layerIndex) {
    NVTX_RANGE("CUDASimulation::stepHostFunctions");
    // Execute all host functions attached to layer
    // TODO: Concurrency?
    assert(host_api);
    for (auto &stepFn : layer->host_functions) {
        NVTX_RANGE("hostFunc");
        stepFn(this->host_api.get());
    }
    // Execute all host function callbacks attached to layer
    for (auto &stepFn : layer->host_functions_callbacks) {
        NVTX_RANGE("hostFunc_swig");
        stepFn->run(this->host_api.get());
    }
    // If we have host layer functions, we might have host agent creation
    if (layer->host_functions.size() || (layer->host_functions_callbacks.size())) {
        // @todo - What is the most appropriate stream to use here?
        processHostAgentCreation(0);
    }
}

void CUDASimulation::stepStepFunctions() {
    NVTX_RANGE("CUDASimulation::step::StepFunctions");
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
    if (model->stepFunctions.size() || model->stepFunctionCallbacks.size()) {
        processHostAgentCreation(0);
    }
}

bool CUDASimulation::stepExitConditions() {
    NVTX_RANGE("CUDASimulation::stepExitConditions");
    // Track if any exit conditions were successful. Use this to control return code and skipsteps.
    // early returning makes timing/stepCounter logic more complicated.
    bool exitConditionExit = false;

    // Execute exit conditions
    for (auto &exitCdns : model->exitConditions) {
        if (exitCdns(this->host_api.get()) == EXIT) {
            #ifdef VISUALISATION
                if (visualisation) {
                    visualisation->updateBuffers(step_count+1);
                }
            #endif
            // Set the flag, and bail out of the exit condition loop.
            exitConditionExit = true;
            break;
        }
    }
    // Execute exit condition callbacks
    if (!exitConditionExit) {
        for (auto &exitCdns : model->exitConditionCallbacks) {
            if (exitCdns->run(this->host_api.get()) == EXIT) {
                #ifdef VISUALISATION
                if (visualisation) {
                    visualisation->updateBuffers(step_count+1);
                }
                #endif
                // Set the flag, and bail out of the exit condition loop.
                exitConditionExit = true;
                break;
            }
        }
    }
    // No need for this if any exit conditions passed.
    if (!exitConditionExit) {
        // If we have exit conditions functions, we might have host agent creation
        if (model->exitConditions.size() || model->exitConditionCallbacks.size()) {
            processHostAgentCreation(0);
        }

        #ifdef VISUALISATION
            if (visualisation) {
                visualisation->updateBuffers(step_count+1);
            }
        #endif
    }
    return exitConditionExit;
}

void CUDASimulation::simulate() {
    NVTX_RANGE("CUDASimulation::simulate");

    // Ensure there is work to do.
    if (agent_map.size() == 0) {
        THROW exception::InvalidCudaAgentMapSize("Simulation has no agents, in CUDASimulation::simulate().");
    }

    // Ensure singletons have been initialised
    initialiseSingletons();

    // Create the event timing object, using an appropriate timer implementation.
    std::unique_ptr<util::detail::Timer> simulationTimer = getDriverAppropriateTimer();
    simulationTimer->start();

    // Create as many streams as required
    unsigned int nStreams = getMaximumLayerWidth();
    this->createStreams(nStreams);

    // Init any unset agent IDs
    this->assignAgentIDs();

    // Reinitialise any unmapped agent variables
    if (submodel) {
        int streamIdx = 0;
        for (auto &a : agent_map) {
            a.second->initUnmappedVars(this->singletons->scatter, streamIdx, this->getStream(streamIdx));
            streamIdx++;
        }
    }

    // Reset the class' elapsed time value.
    this->elapsedSecondsSimulation = 0.f;
    this->elapsedSecondsPerStep.clear();
    if (getSimulationConfig().steps > 0) {
        this->elapsedSecondsPerStep.reserve(getSimulationConfig().steps);
    }

    // Execute init functions
    this->initFunctions();

    // Determine which agents will be spatially sorted
    this->determineAgentsToSort();

    // Reset and log initial state to step log 0
    resetLog();
    processStepLog();

    #ifdef VISUALISATION
    // Pre step-loop visualisation update
    if (visualisation) {
        visualisation->updateBuffers();
    }
    #endif

    // Run the required number of simulation steps.
    for (unsigned int i = 0; getSimulationConfig().steps == 0 ? true : i < getSimulationConfig().steps; i++) {
        // Run the step
        bool continueSimulation = step();
        if (!continueSimulation) {
            processStepLog();
            break;
        }
        #ifdef VISUALISATION
        // Special case, if steps == 0 and visualisation has been closed
        if (getSimulationConfig().steps == 0 &&
            visualisation && !visualisation->isRunning()) {
            visualisation->join();  // Vis exists in separate thread, make sure it has actually exited
            break;
        }
        #endif
    }

    // Exit functions
    this->exitFunctions();
    processExitLog();

    // Sync visualistaion after the exit functions
    #ifdef VISUALISATION
    if (visualisation) {
        visualisation->updateBuffers();
    }
    #endif

    // Record, store and output the elapsed simulation time
    simulationTimer->stop();
    elapsedSecondsSimulation = simulationTimer->getElapsedSeconds();
    if (getSimulationConfig().timing) {
        // Resolution is 0.5 microseconds, so print to 1 us.
        fprintf(stdout, "Total Processing time: %.6f s\n", elapsedSecondsSimulation);
    }
    // Export logs
    if (!SimulationConfig().step_log_file.empty())
        exportLog(SimulationConfig().step_log_file, true, false);
    if (!SimulationConfig().exit_log_file.empty())
        exportLog(SimulationConfig().exit_log_file, false, true);
    if (!SimulationConfig().common_log_file.empty())
        exportLog(SimulationConfig().common_log_file, true, true);
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

    // Reset any timing data.
    this->elapsedSecondsSimulation = 0.f;
    this->elapsedSecondsPerStep.clear();
}

void CUDASimulation::setPopulationData(AgentVector& population, const std::string& state_name) {
    // Ensure singletons have been initialised
    initialiseSingletons();
    NVTX_RANGE("CUDASimulation::setPopulationData()");
    auto it = agent_map.find(population.getAgentName());
    if (it == agent_map.end()) {
        THROW exception::InvalidAgent("Agent '%s' was not found, "
            "in CUDASimulation::setPopulationData()",
            population.getAgentName().c_str());
    }
    // This call hierarchy validates agent desc matches and state is valid
    it->second->setPopulationData(population, state_name, this->singletons->scatter, 0, 0);  // Streamid shouldn't matter here, also using default stream.
#ifdef VISUALISATION
    if (visualisation) {
        visualisation->updateBuffers();
    }
#endif
    gpuErrchk(cudaDeviceSynchronize());
    agent_ids_have_init = false;
}
void CUDASimulation::getPopulationData(AgentVector& population, const std::string& state_name) {
    // Ensure singletons have been initialised
    initialiseSingletons();
    NVTX_RANGE("CUDASimulation::getPopulationData()");
    gpuErrchk(cudaDeviceSynchronize());
    auto it = agent_map.find(population.getAgentName());
    if (it == agent_map.end()) {
        THROW exception::InvalidAgent("Agent '%s' was not found, "
            "in CUDASimulation::setPopulationData()",
            population.getAgentName().c_str());
    }
    // This call hierarchy validates agent desc matches and state is valid
    it->second->getPopulationData(population, state_name);
    gpuErrchk(cudaDeviceSynchronize());
}

CUDAAgent& CUDASimulation::getCUDAAgent(const std::string& agent_name) const {
    CUDAAgentMap::const_iterator it;
    it = agent_map.find(agent_name);

    if (it == agent_map.end()) {
        THROW exception::InvalidCudaAgent("CUDA agent ('%s') not found, in CUDASimulation::getCUDAAgent().",
            agent_name.c_str());
    }

    return *(it->second);
}

AgentInterface& CUDASimulation::getAgent(const std::string& agent_name) {
    // Ensure singletons have been initialised
    initialiseSingletons();

    auto it = agent_map.find(agent_name);

    if (it == agent_map.end()) {
        THROW exception::InvalidCudaAgent("CUDA agent ('%s') not found, in CUDASimulation::getAgent().",
            agent_name.c_str());
    }

    return *(it->second);
}

CUDAMessage& CUDASimulation::getCUDAMessage(const std::string& message_name) const {
    CUDAMessageMap::const_iterator it;
    it = message_map.find(message_name);

    if (it == message_map.end()) {
        THROW exception::InvalidCudaMessage("CUDA message ('%s') not found, in CUDASimulation::getCUDAMessage().",
            message_name.c_str());
    }

    return *(it->second);
}

void CUDASimulation::setStepLog(const StepLoggingConfig &stepConfig) {
    // Validate ModelDescription matches
    if (*stepConfig.model != *model) {
        THROW exception::InvalidArgument("Model descriptions attached to LoggingConfig and CUDASimulation do not match, in CUDASimulation::setStepLog()\n");
    }
    // Set internal config
    step_log_config = std::make_shared<StepLoggingConfig>(stepConfig);
}
void CUDASimulation::setExitLog(const LoggingConfig &exitConfig) {
    // Validate ModelDescription matches
    if (*exitConfig.model != *model) {
        THROW exception::InvalidArgument("Model descriptions attached to LoggingConfig and CUDASimulation do not match, in CUDASimulation::setExitLog()\n");
    }
    // Set internal config
    exit_log_config = std::make_shared<LoggingConfig>(exitConfig);
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
        THROW exception::InvalidCUDAdevice("Error finding CUDA devices!  Do you have a CUDA-capable GPU installed?");
    }
    if (device_count == 0) {
        THROW exception::InvalidCUDAdevice("Error no CUDA devices found!");
    }

    // Select device
    if (config.device_id >= device_count) {
        THROW exception::InvalidCUDAdevice("Error setting CUDA device to '%d', only %d available!", config.device_id, device_count);
    }
    if (deviceInitialised !=- 1 && deviceInitialised != config.device_id) {
        THROW exception::InvalidCUDAdevice("Unable to set CUDA device to '%d' after the CUDASimulation has already initialised on device '%d'.", config.device_id, deviceInitialised);
    }

    // Check the compute capability of the device, throw an exception if not valid for the executable.
    if (!util::detail::compute_capability::checkComputeCapability(static_cast<int>(config.device_id))) {
        int min_cc = util::detail::compute_capability::minimumCompiledComputeCapability();
        int cc = util::detail::compute_capability::getComputeCapability(static_cast<int>(config.device_id));
        THROW exception::InvalidCUDAComputeCapability("Error application compiled for CUDA Compute Capability %d and above. Device %u is compute capability %d. Rebuild for SM_%d.", min_cc, config.device_id, cc, cc);
    }

    cudaStatus = cudaSetDevice(static_cast<int>(config.device_id));
    if (cudaStatus != cudaSuccess) {
        THROW exception::InvalidCUDAdevice("Unknown error setting CUDA device to '%d'. (%d available)", config.device_id, device_count);
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
    initialiseSingletons();

    // We init Random through submodel hierarchy after singletons
    reseed(getSimulationConfig().random_seed);
}

void CUDASimulation::reseed(const uint64_t &seed) {
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
        if (!util::detail::compute_capability::checkComputeCapability(static_cast<int>(config.device_id))) {
            int min_cc = util::detail::compute_capability::minimumCompiledComputeCapability();
            int cc = util::detail::compute_capability::getComputeCapability(static_cast<int>(config.device_id));
            THROW exception::InvalidCUDAComputeCapability("Error application compiled for CUDA Compute Capability %d and above. Device %u is compute capability %d. Rebuild for SM_%d.", min_cc, config.device_id, cc, cc);
        }
        gpuErrchk(cudaGetDevice(&deviceInitialised));
        std::unique_lock<std::shared_timed_mutex> maps_lock(active_device_maps_mutex);
        auto &adm = active_device_mutex[deviceInitialised];
        if (active_device_instances.find(deviceInitialised) == active_device_instances.end()) {
            active_device_instances[deviceInitialised] = 0;
        }
        auto &adi = active_device_instances[deviceInitialised];
        std::shared_lock<std::shared_timed_mutex> lock(adm);
        ++(adi);
        // Check if device has been reset
        unsigned int DEVICE_HAS_RESET_CHECK = 0;
        gpuErrchk(cudaMemcpyFromSymbol(&DEVICE_HAS_RESET_CHECK, DEVICE_HAS_RESET, sizeof(unsigned int)));
        if (DEVICE_HAS_RESET_CHECK == DEVICE_HAS_RESET_FLAG) {
            // Device has been reset, purge host mirrors of static objects/singletons
            detail::curve::Curve::getInstance().purge();
            if (singletons) {
                singletons->rng.purge();
                singletons->scatter.purge();
            }
            EnvironmentManager::getInstance().purge();
            macro_env.purge();
            // Reset flag
            DEVICE_HAS_RESET_CHECK = 0;  // Any value that doesnt match DEVICE_HAS_RESET_FLAG
            gpuErrchk(cudaMemcpyToSymbol(DEVICE_HAS_RESET, &DEVICE_HAS_RESET_CHECK, sizeof(unsigned int)));
        }
        lock.unlock();
        maps_lock.unlock();
        // Get references to all required singleton and store in the instance.
        singletons = new Singletons(
            detail::curve::Curve::getInstance(),
            EnvironmentManager::getInstance());

        // Reinitialise random for this simulation instance
        singletons->rng.reseed(getSimulationConfig().random_seed);

        // Pass created RandomManager to host api
        host_api = std::make_unique<HostAPI>(*this, singletons->rng, singletons->scatter, agentOffsets, agentData, macro_env, 0, getStream(0));  // Host fns are currently all serial

        for (auto &cm : message_map) {
            cm.second->init(singletons->scatter, 0);
        }

        // Populate the environment properties
        if (!submodel) {
            singletons->environment.init(instance_id, *model->environment, isPureRTC);
            macro_env.init();
        } else {
            singletons->environment.init(instance_id, *model->environment, isPureRTC, mastermodel->getInstanceID(), *submodel->subenvironment);
            macro_env.init(*submodel->subenvironment, mastermodel->macro_env);
        }

        // Propagate singleton init to submodels
        for (auto &sm : submodel_map) {
            sm.second->initialiseSingletons();
        }

        // Store the WDDM/TCC driver mode status, for timer class decisions. Result is cached in the anon namespace to avoid multiple queries
        deviceUsingWDDM = util::detail::wddm::deviceIsWDDM();

        singletonsInitialised = true;
    } else {
        int t = -1;
        gpuErrchk(cudaGetDevice(&t));
        if (t != deviceInitialised) {
            THROW exception::CUDAError("CUDASimulation initialised on device %d, but stepped on device %d.\n", deviceInitialised, t);
        }
    }
    // Populate the environment properties
    initEnvironmentMgr();

    // Ensure there are enough streams to execute the layer.
    // Taking into consideration if in-layer concurrency is disabled or not.
    unsigned int nStreams = getMaximumLayerWidth();
    this->createStreams(nStreams);

    // Ensure RTC is set up.
    initialiseRTC();
}

void CUDASimulation::initialiseRTC() {
    // Only do this once.
    if (!rtcInitialised) {
        NVTX_RANGE("CUDASimulation::initialiseRTC");
        std::unique_ptr<util::detail::Timer> rtcTimer(new util::detail::SteadyClockTimer());
        rtcTimer->start();
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
                    a_it->second->addInstantitateRTCFunction(*it_f->second, macro_env);
                }
                // check rtc source to see if the function condition is an rtc condition
                if (!it_f->second->rtc_condition_source.empty()) {
                    // create CUDA agent RTC function condition by calling addInstantitateRTCFunction on CUDAAgent with AgentFunctionData
                    a_it->second->addInstantitateRTCFunction(*it_f->second, macro_env, true);
                }
            }
        }

        // Initialise device environment for RTC
        singletons->environment.initRTC(*this);

        rtcInitialised = true;

        // Record, store and output the elapsed time of the step.
        rtcTimer->stop();
        this->elapsedSecondsRTCInitialisation = rtcTimer->getElapsedSeconds();
        if (getSimulationConfig().timing) {
            fprintf(stdout, "RTC Initialisation Processing time: %.6f s\n", this->elapsedSecondsRTCInitialisation);
        }
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
visualiser::ModelVis &CUDASimulation::getVisualisation() {
    if (!visualisation)
        visualisation = std::make_unique<visualiser::ModelVis>(*this);
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
        agentData.emplace(agent.first, std::move(agent_states));
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
                gpuErrchk(cudaMemcpyAsync(dt_buff, t_buff, size_req, cudaMemcpyHostToDevice, this->getStream(streamId)));
                // Scatter to device
                auto &cudaagent = agent_map.at(agent.first);
                cudaagent->scatterHostCreation(state.first, static_cast<unsigned int>(state.second.size()), dt_buff, offsets, this->singletons->scatter, streamId, this->getStream(streamId));
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

void CUDASimulation::incrementStepCounter() {
    this->step_count++;
    this->singletons->environment.setProperty({instance_id, "_stepCount"}, this->step_count);
}

double CUDASimulation::getElapsedTimeSimulation() const {
    // Get the value
    return this->elapsedSecondsSimulation;
}

double CUDASimulation::getElapsedTimeInitFunctions() const {
    // Get the value
    return this->elapsedSecondsInitFunctions;
}

double CUDASimulation::getElapsedTimeExitFunctions() const {
    // Get the value
    return this->elapsedSecondsExitFunctions;
}
double CUDASimulation::getElapsedTimeRTCInitialisation() const {
    // Get the value
    return this->elapsedSecondsRTCInitialisation;
}

std::vector<double> CUDASimulation::getElapsedTimeSteps() const {
    // returns a copy of the timing vector, to avoid mutabililty issues. This should not be called in a performacne intensive part of the application.
    std::vector<double> rtn = this->elapsedSecondsPerStep;
    return rtn;
}

double CUDASimulation::getElapsedTimeStep(unsigned int step) const {
    if (step > this->elapsedSecondsPerStep.size()) {
        THROW exception::OutOfBoundsException("getElapsedTimeStep out of bounds.\n");
    }
    return this->elapsedSecondsPerStep.at(step);
}

void CUDASimulation::initEnvironmentMgr() {
    if (!singletons) {
        THROW exception::UnknownInternalError("CUDASimulation::initEnvironmentMgr() called before singletons member initialised.");
    }

    // Set any properties loaded from file during arg parse stage
    for (const auto &prop : env_init) {
        const EnvironmentManager::NamePair np = { instance_id , prop.first.first };
        if (!singletons->environment.containsProperty(np)) {
            THROW exception::InvalidEnvProperty("Environment init data contains unexpected environment property '%s', "
                "in CUDASimulation::initEnvironmentMgr()\n", prop.first.first.c_str());
        }
        const std::type_index val_type = singletons->environment.type(np);
        if (val_type == std::type_index(typeid(float))) {
            singletons->environment.setProperty<float>(np, prop.first.second, *static_cast<float*>(prop.second.ptr));
        } else if (val_type == std::type_index(typeid(double))) {
            singletons->environment.setProperty<double>(np, prop.first.second, *static_cast<double*>(prop.second.ptr));
        } else if (val_type == std::type_index(typeid(int64_t))) {
            singletons->environment.setProperty<int64_t>(np, prop.first.second, *static_cast<int64_t*>(prop.second.ptr));
        } else if (val_type == std::type_index(typeid(uint64_t))) {
            singletons->environment.setProperty<uint64_t>(np, prop.first.second, *static_cast<uint64_t*>(prop.second.ptr));
        } else if (val_type == std::type_index(typeid(int32_t))) {
            singletons->environment.setProperty<int32_t>(np, prop.first.second, *static_cast<int32_t*>(prop.second.ptr));
        } else if (val_type == std::type_index(typeid(uint32_t))) {
            singletons->environment.setProperty<uint32_t>(np, prop.first.second, *static_cast<uint32_t*>(prop.second.ptr));
        } else if (val_type == std::type_index(typeid(int16_t))) {
            singletons->environment.setProperty<int16_t>(np, prop.first.second, *static_cast<int16_t*>(prop.second.ptr));
        } else if (val_type == std::type_index(typeid(uint16_t))) {
            singletons->environment.setProperty<uint16_t>(np, prop.first.second, *static_cast<uint16_t*>(prop.second.ptr));
        } else if (val_type == std::type_index(typeid(int8_t))) {
            singletons->environment.setProperty<int8_t>(np, prop.first.second, *static_cast<int8_t*>(prop.second.ptr));
        } else if (val_type == std::type_index(typeid(uint8_t))) {
            singletons->environment.setProperty<uint8_t>(np, prop.first.second, *static_cast<uint8_t*>(prop.second.ptr));
        } else {
            THROW exception::InvalidEnvProperty("Environment init data contains environment property '%s' of unsupported type '%s', "
                "this should have been caught during file parsing, "
                "in CUDASimulation::initEnvironmentMgr()\n", prop.first.first.c_str(), val_type.name());
        }
    }
    // Clear init
    env_init.clear();
}
void CUDASimulation::resetLog() {
    run_log->step.clear();
    run_log->exit = LogFrame();
    run_log->random_seed = SimulationConfig().random_seed;
    run_log->step_log_frequency = step_log_config ? step_log_config->frequency : 0;
}
void CUDASimulation::processStepLog() {
    if (!step_log_config)
        return;
    if (step_count % step_log_config->frequency != 0)
        return;
    // Iterate members of step log to build the step log frame
    std::map<std::string, util::Any> environment_log;
    for (const auto &prop_name : step_log_config->environment) {
        // Fetch the named environment prop
        environment_log.emplace(prop_name, singletons->environment.getPropertyAny(instance_id, prop_name));
    }
    std::map<util::StringPair, std::pair<std::map<LoggingConfig::NameReductionFn, util::Any>, unsigned int>> agents_log;
    for (const auto &name_state : step_log_config->agents) {
        // Create the named sub map
        const std::string &agent_name = name_state.first.first;
        const std::string &agent_state = name_state.first.second;
        HostAgentAPI host_agent = host_api->agent(agent_name, agent_state);
        auto &agent_state_log = agents_log.emplace(name_state.first, std::make_pair(std::map<LoggingConfig::NameReductionFn, util::Any>(), UINT_MAX)).first->second;
        // Log individual variable reductions
        for (const auto &name_reduction : *name_state.second.first) {
            // Perform the corresponding reduction
            auto result = name_reduction.function(host_agent, name_reduction.name);
            // Store the result
            agent_state_log.first.emplace(name_reduction, std::move(result));
        }
        // Log count of agents in state
        if (name_state.second.second) {
            agent_state_log.second = host_api->agent(agent_name, agent_state).count();
        }
    }

    // Append to step log
    run_log->step.push_back(LogFrame(std::move(environment_log), std::move(agents_log), step_count));
}

void CUDASimulation::processExitLog() {
    if (!exit_log_config)
        return;
    // Iterate members of step log to build the step log frame
    std::map<std::string, util::Any> environment_log;
    for (const auto &prop_name : exit_log_config->environment) {
        // Fetch the named environment prop
        environment_log.emplace(prop_name, singletons->environment.getPropertyAny(instance_id, prop_name));
    }
    std::map<util::StringPair, std::pair<std::map<LoggingConfig::NameReductionFn, util::Any>, unsigned int>> agents_log;
    for (const auto &name_state : exit_log_config->agents) {
        // Create the named sub map
        const std::string &agent_name = name_state.first.first;
        const std::string &agent_state = name_state.first.second;
        HostAgentAPI host_agent = host_api->agent(agent_name, agent_state);
        auto &agent_state_log = agents_log.emplace(name_state.first, std::make_pair(std::map<LoggingConfig::NameReductionFn, util::Any>(), UINT_MAX)).first->second;
        // Log individual variable reductions
        for (const auto &name_reduction : *name_state.second.first) {
            // Perform the corresponding reduction
            auto result = name_reduction.function(host_agent, name_reduction.name);
            // Store the result
            agent_state_log.first.emplace(name_reduction, std::move(result));
        }
        // Log count of agents in state
        if (name_state.second.second) {
            agent_state_log.second = host_api->agent(agent_name, agent_state).count();
        }
    }

    // Set Log
    run_log->exit = LogFrame(std::move(environment_log), std::move(agents_log), step_count);
}
const RunLog &CUDASimulation::getRunLog() const {
    return *run_log;
}

void CUDASimulation::createStreams(const unsigned int nStreams) {
    // There should always be atleast 1 stream, as some tests require the 0th stream even when there is no concurrent work to be done.
    unsigned int totalStreams = std::max(nStreams, 1u);
    while (streams.size() < totalStreams) {
        cudaStream_t stream = 0;
        gpuErrchk(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }
}

cudaStream_t CUDASimulation::getStream(const unsigned int n) {
    // Return the appropriate stream, unless concurrency is disabled in which case always stream 0.
    if (this->streams.size() <= n) {
        unsigned int nStreams = getMaximumLayerWidth();
        this->createStreams(nStreams);
    }

    if (getCUDAConfig().inLayerConcurrency && n < streams.size()) {
        return streams.at(n);
    } else {
        return streams.at(0);
    }
}

void CUDASimulation::destroyStreams() {
    // Destroy streams.
    for (auto stream : streams) {
        gpuErrchk(cudaStreamDestroy(stream));
    }
    streams.clear();
}

void CUDASimulation::synchronizeAllStreams() {
    // Destroy streams.
    for (auto stream : streams) {
        gpuErrchk(cudaStreamSynchronize(stream));
    }
}

void CUDASimulation::assignAgentIDs() {
    NVTX_RANGE("CUDASimulation::assignAgentIDs");
    if (!agent_ids_have_init) {
        // Ensure singletons have been initialised
        initialiseSingletons();

        for (auto &a : agent_map) {
            a.second->assignIDs(*host_api);  // This is cheap if the CUDAAgent thinks it's IDs are already assigned
        }
        agent_ids_have_init = true;
    }
}

}  // namespace flamegpu
