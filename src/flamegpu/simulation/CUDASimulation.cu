#include "flamegpu/simulation/CUDASimulation.h"


#include <algorithm>
#include <string>
#include <map>
#include <numeric>
#include <cstdio>
#include <vector>
#include <utility>
#include <functional>
#include <memory>

#include "jitify/jitify2.hpp"

#include "flamegpu/detail/curand.cuh"
#include "flamegpu/model/AgentFunctionData.cuh"
#include "flamegpu/model/LayerData.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/model/SubAgentData.h"
#include "flamegpu/runtime/HostAPI.h"
#include "flamegpu/simulation/detail/CUDAEnvironmentDirectedGraphBuffers.cuh"
#include "flamegpu/simulation/detail/CUDAScanCompaction.h"
#include "flamegpu/util/nvtx.h"
#include "flamegpu/detail/compute_capability.cuh"
#include "flamegpu/detail/SignalHandlers.h"
#include "flamegpu/detail/wddm.cuh"
#include "flamegpu/detail/SteadyClockTimer.h"
#include "flamegpu/detail/CUDAEventTimer.cuh"
#include "flamegpu/runtime/detail/curve/curve_rtc.cuh"
#include "flamegpu/runtime/HostFunctionCallback.h"
#include "flamegpu/runtime/messaging.h"
#include "flamegpu/simulation/detail/CUDAAgent.h"
#include "flamegpu/simulation/detail/CUDAMessage.h"
#include "flamegpu/simulation/LoggingConfig.h"
#include "flamegpu/simulation/LogFrame.h"
#include "flamegpu/simulation/RunPlan.h"
#include "flamegpu/version.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/SubEnvironmentData.h"
#include "flamegpu/io/Telemetry.h"
#ifdef FLAMEGPU_VISUALISATION
#include "flamegpu/visualiser/FLAMEGPU_Visualisation.h"
#endif

namespace flamegpu {

namespace {
    // file-scope only variable used to cache the driver mode
    bool deviceUsingWDDM = false;
    // Inlined method in the anonymous namespace to create a new timer, subject to the driver model.
    std::unique_ptr<detail::Timer> getDriverAppropriateTimer(bool is_ensemble) {
        if (!deviceUsingWDDM && !is_ensemble) {
            return std::unique_ptr<detail::Timer>(new detail::CUDAEventTimer());
        } else {
            return std::unique_ptr<detail::Timer>(new detail::SteadyClockTimer());
        }
    }
}  // anonymous namespace

CUDASimulation::CUDASimulation(const ModelDescription& _model, int argc, const char** argv, bool _isSWIG)
    : CUDASimulation(_model.model, _isSWIG) {
    if (argc && argv) {
        initialise(argc, argv);
    }
}
CUDASimulation::CUDASimulation(const std::shared_ptr<const ModelData> &_model, bool _isSWIG)
    : Simulation(_model)
    , step_count(0)
    , elapsedSecondsSimulation(0.)
    , elapsedSecondsInitFunctions(0.)
    , elapsedSecondsExitFunctions(0.)
    , elapsedSecondsRTCInitialisation(0.)
    , macro_env(std::make_shared<detail::CUDAMacroEnvironment>(*_model->environment, *this))
    , config({})
    , run_log(std::make_unique<RunLog>())
    , streams(std::vector<cudaStream_t>())
    , singletons(nullptr)
    , singletonsInitialised(false)
    , rtcInitialised(false)
#if __CUDACC_VER_MAJOR__ >= 12
    , cudaContextID(std::numeric_limits<std::uint64_t>::max())
#endif  // __CUDACC_VER_MAJOR__ >= 12
    , isPureRTC(detectPureRTC(model))
    , isSWIG(_isSWIG) {
    initOffsetsAndMap();
    // Register the signal handler.
    detail::SignalHandlers::registerSignalHandlers();

    // populate the CUDA agent map
    const auto &am = model->agents;
    // create new cuda agent and add to the map
    for (auto it = am.cbegin(); it != am.cend(); ++it) {
        agent_map.emplace(it->first, std::make_unique<detail::CUDAAgent>(*it->second, *this));
    }

    // populate the CUDA message map
    const auto &mm = model->messages;
    // create new cuda message and add to the map
    for (auto it_m = mm.cbegin(); it_m != mm.cend(); ++it_m) {
        message_map.emplace(it_m->first, std::make_unique<detail::CUDAMessage>(*it_m->second, *this));
    }

    // populate the CUDA submodel map
    const auto &smm = model->submodels;
    // create new cuda message and add to the map
    for (auto it_sm = smm.cbegin(); it_sm != smm.cend(); ++it_sm) {
        submodel_map.emplace(it_sm->first, std::unique_ptr<CUDASimulation>(new CUDASimulation(it_sm->second, this)));
    }

    // Determine which agents will be spatially sorted
    this->determineAgentsToSort();
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
    , macro_env(std::make_shared<detail::CUDAMacroEnvironment>(*submodel_desc->submodel->environment, *this))
    , run_log(std::make_unique<RunLog>())
    , streams(std::vector<cudaStream_t>())
    , singletons(nullptr)
    , singletonsInitialised(false)
    , rtcInitialised(false)
#if __CUDACC_VER_MAJOR__ >= 12
    , cudaContextID(std::numeric_limits<std::uint64_t>::max())
#endif  // __CUDACC_VER_MAJOR__ >= 12
    , isPureRTC(master_model->isPureRTC)
    , isSWIG(master_model->isSWIG) {
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
            std::unique_ptr<detail::CUDAAgent> &masterAgent = master_model->agent_map.at(masterAgentDesc->name);
            agent_map.emplace(it->first, std::make_unique<detail::CUDAAgent>(*it->second, *this, masterAgent, mapping));
        } else {
            // Agent is not mapped, create regular agent
            agent_map.emplace(it->first, std::make_unique<detail::CUDAAgent>(*it->second, *this));
        }
    }  // insert into map using value_type

    // populate the CUDA message map (Sub Messages not currently supported)
    const auto &mm = model->messages;
    // create new cuda message and add to the map
    for (auto it_m = mm.cbegin(); it_m != mm.cend(); ++it_m) {
        message_map.emplace(it_m->first, std::make_unique<detail::CUDAMessage>(*it_m->second, *this));
    }

    // populate the CUDA submodel map
    const auto &smm = model->submodels;
    // create new cuda model and add to the map
    for (auto it_sm = smm.cbegin(); it_sm != smm.cend(); ++it_sm) {
        submodel_map.emplace(it_sm->first, std::unique_ptr<CUDASimulation>(new CUDASimulation(it_sm->second, this)));
    }
    // Submodels all run quiet/not verbose by default
    SimulationConfig().verbosity = Verbosity::Default;
    SimulationConfig().steps = submodel_desc->max_steps;
    CUDAConfig().is_submodel = true;

    // Determine which agents will be spatially sorted
    this->determineAgentsToSort();
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
        delete singletons;
        singletons = nullptr;
    }

    // We can't cleanly wrap the jitify2::KernelData destructor
    // Instead perform a check and release them if necessary
    this->safeDestroyJitify();

    // We must explicitly delete all cuda members before we cuda device reset
    agent_map.clear();
    message_map.clear();
    submodel_map.clear();
    directed_graph_map.clear();
    host_api.reset();
    macro_env->free();
#ifdef FLAMEGPU_VISUALISATION
    visualisation.reset();  // Might want to force destruct this, as user could hold a ModelVis that has shared ptr
#endif

    // Destroy streams, potentially unsafe in a destructor as it will invoke cuda commands.
    // Do this once to re-use existing streams rather than per-step.
    this->destroyStreams();

    // Reset the active device if not the device used for this simulation
    if (t_device_id != deviceInitialised) {
        gpuErrchk(cudaSetDevice(t_device_id));
    }
}

void CUDASimulation::initFunctions() {
    flamegpu::util::nvtx::Range range{"CUDASimulation::initFunctions"};
    std::unique_ptr<detail::Timer> initFunctionsTimer(new detail::SteadyClockTimer());
    initFunctionsTimer->start();

    // Ensure singletons have been initialised
    initialiseSingletons();

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
        // Sync any device vectors, before performing host agent creation
        for (auto& ca : agent_map) {
            ca.second->resetPopulationVecs();
        }
        processHostAgentCreation(0);
        // If we have host layer functions, a static graph may need updating
        for (auto& [name, sg] : directed_graph_map) {
            sg->syncDevice_async(singletons->scatter, 0, getStream(0));
        }
    }

    // Record, store and output the elapsed time of the step.
    initFunctionsTimer->stop();
    this->elapsedSecondsInitFunctions = initFunctionsTimer->getElapsedSeconds();
    if (getSimulationConfig().timing || getSimulationConfig().verbosity >= Verbosity::Verbose) {
        fprintf(stdout, "Init Function Processing time: %.6f s\n", this->elapsedSecondsInitFunctions);
    }
}

void CUDASimulation::exitFunctions() {
    flamegpu::util::nvtx::Range range{"CUDASimulation::exitFunctions"};
    std::unique_ptr<detail::Timer> exitFunctionsTimer(new detail::SteadyClockTimer());
    exitFunctionsTimer->start();

    // Ensure singletons have been initialised
    initialiseSingletons();

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
    if (getSimulationConfig().timing || getSimulationConfig().verbosity >= Verbosity::Verbose) {
        fprintf(stdout, "Exit Function Processing time: %.6f s\n", this->elapsedSecondsExitFunctions);
    }
}

namespace detail {
template <typename T> struct Dims {
    T x;
    T y;
    T z;
};
}

__global__ void calculateSpatialHashFloat3(float* xyz, unsigned int* binIndex, detail::Dims<float> envMin, detail::Dims<float> envWidth, detail::Dims<unsigned int> gridDim, unsigned int threadCount) {
    const unsigned int TID = blockIdx.x * blockDim.x + threadIdx.x;
    if (TID < threadCount) {
        // Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
        int gridPos[3] = {
            static_cast<int>(floorf(((xyz[TID * 3 + 0] - envMin.x) / envWidth.x) * gridDim.x)),
            static_cast<int>(floorf(((xyz[TID * 3 + 1] - envMin.y) / envWidth.y) * gridDim.y)),
            static_cast<int>(floorf(((xyz[TID * 3 + 2] - envMin.z) / envWidth.z) * gridDim.z))
        };

        // Compute and set the bin index
        unsigned int bindex;

        bindex = (unsigned int)(
            (gridPos[2] * gridDim.x * gridDim.y +   // z
                (gridPos[1] * gridDim.x) +              // y
                gridPos[0]));                           // x

        binIndex[TID] = bindex;
    }
}
__global__ void calculateSpatialHashFloat2(float* xy, unsigned int* binIndex, detail::Dims<float> envMin, detail::Dims<float> envWidth, detail::Dims<unsigned int> gridDim, unsigned int threadCount) {
    const unsigned int TID = blockIdx.x * blockDim.x + threadIdx.x;
    if (TID < threadCount) {
        // Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
        int gridPos[3] = {
            static_cast<int>(floorf(((xy[TID * 2 + 0] - envMin.x) / envWidth.x) * gridDim.x)),
            static_cast<int>(floorf(((xy[TID * 2 + 1] - envMin.y) / envWidth.y) * gridDim.y)),
            0
        };

        // Compute and set the bin index
        unsigned int bindex;

        bindex = (unsigned int)(
            (gridPos[1] * gridDim.x) +              // y
            gridPos[0]);                            // x

        binIndex[TID] = bindex;
    }
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
                    CAgentDescription ad(it->second);
                    if (ad.hasVariable("x") && ad.hasVariable("y") && ad.hasVariable("z")) {
                        auto& x = it->second->variables.at("x");
                        auto& y = it->second->variables.at("y");
                        auto& z = it->second->variables.at("z");
                        if (x.type == std::type_index(typeid(float)) && x.elements == 1 &&
                            y.type == std::type_index(typeid(float)) && y.elements == 1 &&
                            z.type == std::type_index(typeid(float)) && z.elements == 1) {
                            sortTriggers3D.insert(it_f->first);
                        }
                    } else if (ad.hasVariable("xyz")) {
                        auto& xyz = it->second->variables.at("xyz");
                        if (xyz.type == std::type_index(typeid(float)) && xyz.elements == 3) {
                            sortTriggers3D.insert(it_f->first);
                        }
                    }
                }

                // Check if this agent function uses 2D spatial messages
                if (ptr->getSortingType() == flamegpu::MessageSortingType::spatial2D) {
                    // Agent uses spatial, check it has correct variables
                    CAgentDescription ad(it->second);
                    if (ad.hasVariable("x") && ad.hasVariable("y")) {
                        auto& x = it->second->variables.at("x");
                        auto& y = it->second->variables.at("y");
                        if (x.type == std::type_index(typeid(float)) && x.elements == 1 &&
                            y.type == std::type_index(typeid(float)) && y.elements == 1) {
                            sortTriggers2D.insert(it_f->first);
                        }
                    } else if (ad.hasVariable("xy")) {
                        auto& xy = it->second->variables.at("xy");
                        if (xy.type == std::type_index(typeid(float)) && xy.elements == 2) {
                            sortTriggers2D.insert(it_f->first);
                        }
                    }
                }
            }
        }
    }
}


void CUDASimulation::spatialSortAgent_async(const std::string& funcName, const std::string& agentName, const std::string& state, const int mode, cudaStream_t stream, unsigned int streamId) {
    // Fetch the appropriate message name
    detail::CUDAAgent& cuda_agent = getCUDAAgent(agentName);

    const unsigned int state_list_size = cuda_agent.getStateSize(state);
    // Can't sort no agents
    if (!state_list_size)
        return;

    const CAgentDescription cudaAgentData(cuda_agent.getAgentDescription());
    auto funcData = cudaAgentData.getFunction(funcName);
    if (!funcData.hasMessageInput()) {
        THROW exception::InvalidAgentFunc("Function %s registered for auto-spatial sorting but input message type not found!\n", funcName.c_str());
    }
    std::string messageName = funcData.getMessageInput().getName();
    MessageBruteForce::Data* msgData = model->messages.at(messageName).get();

    // Get the spatial metadata
    float radius;
    detail::Dims<float> envMin {};
    detail::Dims<float> envMax {};
    detail::Dims<float> envWidth {};
    detail::Dims<unsigned int> gridDim {};

    if (auto messageSpatialData2D = dynamic_cast<MessageSpatial2D::Data*>(msgData)) {
        radius = messageSpatialData2D->radius;
        envMin = {messageSpatialData2D->minX, messageSpatialData2D->minY, 0.0f};
        envMax = {messageSpatialData2D->maxX, messageSpatialData2D->maxY, 0.0f};
    } else if (auto messageSpatialData3D = dynamic_cast<MessageSpatial3D::Data*>(msgData)) {
        radius = messageSpatialData3D->radius;
        envMin = {messageSpatialData3D->minX, messageSpatialData3D->minY, messageSpatialData3D->minZ};
        envMax = {messageSpatialData3D->maxX, messageSpatialData3D->maxY, messageSpatialData3D->maxZ};
    } else {
        radius = 0.0f;
        envMin = {0.0f, 0.0f, 0.0f};
        envMax = {0.0f, 0.0f, 0.0f};
    }
    if (radius > 0.0f) {
        envWidth = {(envMax.x-envMin.x), (envMax.y-envMin.y), (envMax.z-envMin.z)};
        gridDim = {
            envWidth.x ? static_cast<unsigned int>(ceilf(envWidth.x / radius)) : 1,
            envWidth.y ? static_cast<unsigned int>(ceilf(envWidth.y / radius)) : 1,
            envWidth.z ? static_cast<unsigned int>(ceilf(envWidth.z / radius)) : 1
        };
    }


    // Any agent in this list is guaranteed to have x, y, z (or xyz vec versions) and _auto_sort_bin_index vars - used in the computation of spatial hash
    // TODO: User could supply alternatives to "x", "y", "z" to use alternative variables?
    void* xPtr = nullptr, *yPtr = nullptr, *zPtr = nullptr;
    void* xyPtr = nullptr, * xyzPtr = nullptr;
    if (mode == Agent3D && cudaAgentData.hasVariable("xyz")) {
        xyzPtr = cuda_agent.getStateVariablePtr(state, "xyz");
    } else if (mode == Agent2D && cudaAgentData.hasVariable("xy")) {
        xyPtr = cuda_agent.getStateVariablePtr(state, "xy");
    } else {
        xPtr = cuda_agent.getStateVariablePtr(state, "x");
        yPtr = cuda_agent.getStateVariablePtr(state, "y");
        zPtr = mode == Agent3D ? cuda_agent.getStateVariablePtr(state, "z") : 0;
    }

    void* binIndexPtr = cuda_agent.getStateVariablePtr(state, "_auto_sort_bin_index");

    // Compute occupancy
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateSpatialHash, 0, state_list_size));

    //! Round up according to CUDAAgent state list size
    gridSize = (state_list_size + blockSize - 1) / blockSize;

    unsigned int sm_size = 0;
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    auto *error_buffer = this->singletons->exception.getDevicePtr(streamId, stream);
    sm_size = sizeof(error_buffer);
#endif

    // Launch kernel
    if (xyzPtr) {
        calculateSpatialHashFloat3<<<gridSize, blockSize, sm_size, stream>>>(reinterpret_cast<float*>(xyzPtr),
            reinterpret_cast<unsigned int*>(binIndexPtr),
            envMin,
            envWidth,
            gridDim,
            state_list_size);
    } else if (xyPtr) {
        calculateSpatialHashFloat2<<<gridSize, blockSize, sm_size, stream>>>(reinterpret_cast<float*>(xyPtr),
            reinterpret_cast<unsigned int*>(binIndexPtr),
            envMin,
            envWidth,
            gridDim,
            state_list_size);
    } else {
        calculateSpatialHash<<<gridSize, blockSize, sm_size, stream>>>(reinterpret_cast<float*>(xPtr),
            reinterpret_cast<float*>(yPtr),
            reinterpret_cast<float*>(zPtr),
            reinterpret_cast<unsigned int*>(binIndexPtr),
            envMin,
            envWidth,
            gridDim,
            state_list_size);
    }
    gpuErrchkLaunch();

    assert(host_api);
    // Calculate max bit (cub::DeviceRadixSort end bit is exclusive and 0-indexed)
    // https://math.stackexchange.com/a/160299/126129
    const int max_bit = static_cast<int>(floor(log2(gridDim.x * gridDim.y * gridDim.z))) + 1;
    host_api->agent(agentName, state).sort_async<unsigned int>("_auto_sort_bin_index", HostAgentAPI::Asc, 0, max_bit, stream, streamId);
}

bool CUDASimulation::step() {
    flamegpu::util::nvtx::Range range{std::string("CUDASimulation::step " + std::to_string(step_count)).c_str()};
    // Ensure singletons have been initialised
    initialiseSingletons();

    // Time the individual step, using a CUDAEventTimer if possible, else a steadyClockTimer.
    std::unique_ptr<detail::Timer> stepTimer = getDriverAppropriateTimer(getCUDAConfig().is_ensemble || getCUDAConfig().is_submodel);
    stepTimer->start();

    // Init any unset agent IDs
    this->assignAgentIDs();

    // If verbose, print the step number.
    if (getSimulationConfig().verbosity == Verbosity::Verbose) {
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

    // Set message counts to zero, and set flags to update state of non-persistent message lists
    for (auto &a : message_map) {
        if (!a.second->getMessageData().persistent) {
            a.second->setMessageCount(0);
            a.second->setTruncateMessageListFlag();
            a.second->setPBMConstructionRequiredFlag();
        }
    }

    // Record, store and output the elapsed time of the step.
    stepTimer->stop();
    float stepMilliseconds = stepTimer->getElapsedSeconds();
    this->elapsedSecondsPerStep.push_back(stepMilliseconds);
    if (getSimulationConfig().timing || getSimulationConfig().verbosity >= Verbosity::Verbose) {
        // Resolution is 0.5 microseconds, so print to 1 us.
        fprintf(stdout, "Step %d Processing time: %.6f s\n", this->step_count, stepMilliseconds);
    }

    // Update step count at the end of the step - when it has completed.
    incrementStepCounter();
    // Update the log for the step.
    processStepLog(this->elapsedSecondsPerStep.back());
    // Return false if any exit condition's passed.
    return !exitRequired;
}

void CUDASimulation::stepLayer(const std::shared_ptr<LayerData>& layer, const unsigned int layerIndex) {
    flamegpu::util::nvtx::Range range{std::string("stepLayer " + std::to_string(layerIndex)).c_str()};

    std::string message_name;

    // If the layer contains a sub model, it can only execute the sub model.
    if (layer->sub_model) {
        this->synchronizeAllStreams();
        auto &sm = submodel_map.at(layer->sub_model->name);
        sm->resetStepCounter();
        sm->simulate();
        sm->reset(true);
        // Next layer, this layer cannot also contain agent functions
        // Ensure synchronisation has occurred.
        this->synchronizeAllStreams();
        return;
    }

    // Track stream index
    int streamIdx = 0;
    // Sum the total number of threads being launched in the layer
    unsigned int totalThreads = 0;

    // Sync the environment once per layer (incase Host Fns, or submodel have changed it)
    singletons->environment->updateDevice_async(getStream(0));

    // Spatially sort the agents
    for (const auto &func_des : layer->agent_functions) {
        auto func_agent = func_des->parent.lock();
        if ((func_agent->sortPeriod != 0) && (step_count % func_agent->sortPeriod == 0)) {
            if (sortTriggers3D.find(func_des->name) != sortTriggers3D.end()) {
                this->spatialSortAgent_async(func_des->name, func_agent->name, func_des->initial_state, Agent3D, getStream(streamIdx), streamIdx);
            } else if (sortTriggers2D.find(func_des->name) != sortTriggers2D.end()) {
                this->spatialSortAgent_async(func_des->name, func_agent->name, func_des->initial_state, Agent2D, getStream(streamIdx), streamIdx);
            }
        }
        ++streamIdx;
    }
    // No explicit sync, sorts should be in same stream as eventual kernel launch (digging deep, the underlying scatter method does have a sync though)
    streamIdx = 0;

    // Map agent memory
    for (const auto &func_des : layer->agent_functions) {
        if ((func_des->condition) || (!func_des->rtc_func_condition_name.empty())) {
            auto func_agent = func_des->parent.lock();
            flamegpu::util::nvtx::Range condition_range{std::string("condition map " + func_agent->name + "::" + func_des->name).c_str()};
            const detail::CUDAAgent& cuda_agent = getCUDAAgent(func_agent->name);

            const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
            if (state_list_size == 0) {
                ++streamIdx;
                continue;
            }
            singletons->scatter.Scan().resize(state_list_size, detail::CUDAScanCompaction::AGENT_DEATH, streamIdx);

            // Configure runtime access of the functions variables within the FLAME_API object
            cuda_agent.mapRuntimeVariables(*func_des, instance_id);

            // Zero the scan flag that will be written to
            singletons->scatter.Scan().zero_async(detail::CUDAScanCompaction::AGENT_DEATH, getStream(streamIdx), streamIdx);
            // No sync, this occurs in same stream as dependent kernel launch

            // Push function's RTC cache to device if using RTC
            if (!func_des->rtc_func_condition_name.empty()) {
                auto &rtc_header = cuda_agent.getRTCHeader(func_des->name + "_condition");
                // Sync EnvManager's RTC cache with RTC header's cache
                rtc_header.updateEnvCache(singletons->environment->getHostBuffer(), singletons->environment->getBufferLen());
                // Push RTC header's cache to device
                rtc_header.updateDevice_async(cuda_agent.getRTCInstantiation(func_des->name + "_condition"), getStream(streamIdx));
            } else {
                auto& curve = cuda_agent.getCurve(func_des->name + "_condition");
                curve.updateDevice_async(this->getStream(streamIdx));
            }
            // No sync, kernel launch should be in same stream

            totalThreads += state_list_size;
        }
        ++streamIdx;
    }

    // If any condition kernel needs to be executed, do so, by checking the number of threads from before.
    if (totalThreads > 0) {
        // Ensure RandomManager is the correct size to accommodate all threads to be launched
        detail::curandState *d_rng = singletons->rng.resize(totalThreads, getStream(0));
        // Track which stream to use for concurrency
        streamIdx = 0;
        // Sum the total number of threads being launched in the layer, for rng offsetting.
        totalThreads = 0;
        // Launch function condition kernels
        for (const auto &func_des : layer->agent_functions) {
            if ((func_des->condition) || (!func_des->rtc_func_condition_name.empty())) {
                auto func_agent = func_des->parent.lock();
                flamegpu::util::nvtx::Range condition_range{std::string("condition " + func_agent->name + "::" + func_des->name).c_str()};
                if (!func_agent) {
                    THROW exception::InvalidAgentFunc("Agent function condition refers to expired agent.");
                }
                std::string agent_name = func_agent->name;
                std::string func_name = func_des->name;

                const detail::CUDAAgent& cuda_agent = getCUDAAgent(agent_name);

                const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
                if (state_list_size == 0) {
                    ++streamIdx;
                    continue;
                }

                int blockSize = 0;  // The launch configurator returned block size
                int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
                int gridSize = 0;  // The actual grid size needed, based on input size

                //  Agent function condition kernel wrapper args
                detail::curandState *t_rng = d_rng + totalThreads;
                unsigned int *scanFlag_agentDeath = this->singletons->scatter.Scan().Config(detail::CUDAScanCompaction::Type::AGENT_DEATH, streamIdx).d_ptrs.scan_flag;
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                auto *error_buffer = this->singletons->exception.getDevicePtr(streamIdx, this->getStream(streamIdx));
#endif
                // switch between normal and RTC agent function condition
                if (func_des->condition) {
                    // calculate the grid block size for agent function condition
                    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func_des->condition, 0, state_list_size));

                    //! Round up according to CUDAAgent state list size
                    gridSize = (state_list_size + blockSize - 1) / blockSize;
                    (func_des->condition) << <gridSize, blockSize, 0, this->getStream(streamIdx) >> > (
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                    error_buffer,
#endif
                    cuda_agent.getCurve(func_des->name + "_condition").getDevicePtr(),
                    this->singletons->strings.getDeviceString(func_agent->name),
                    this->singletons->strings.getDeviceString(func_des->initial_state),
                    static_cast<const char *>(this->singletons->environment->getDeviceBuffer()),
                    state_list_size,
                    t_rng,
                    scanFlag_agentDeath);
                    gpuErrchkLaunch();
                } else {  // RTC function
                    std::string func_condition_identifier = func_name + "_condition";
                    // get instantiation
                    const jitify2::KernelData& instance = cuda_agent.getRTCInstantiation(func_condition_identifier);
                    // calculate the grid block size for main agent function
                    CUfunction cu_func = (CUfunction)instance.function();
                    gpuErrchkDriverAPI(cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cu_func, 0, 0, state_list_size));
                    //! Round up according to CUDAAgent state list size
                    gridSize = (state_list_size + blockSize - 1) / blockSize;
                    // launch the kernel
                    jitify2::ErrorMsg a = instance.configure(gridSize, blockSize, 0, this->getStream(streamIdx))->launch_raw({
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                        reinterpret_cast<void*>(&error_buffer),
#endif
                        const_cast<void *>(reinterpret_cast<const void*>(&state_list_size)),
                        reinterpret_cast<void*>(&t_rng),
                        reinterpret_cast<void*>(&scanFlag_agentDeath) });
                    if (!a.empty()) {
                        THROW exception::InvalidAgentFunc("There was a problem launching the runtime agent function condition '%s': %s", func_des->rtc_func_condition_name.c_str(), a.c_str());
                    }
                    gpuErrchkLaunch();
                }

                totalThreads += state_list_size;
            }
            ++streamIdx;
        }
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
            flamegpu::util::nvtx::Range unmap_range{std::string("condition unmap " + func_agent->name + "::" + func_des->name).c_str()};
            detail::CUDAAgent& cuda_agent = getCUDAAgent(func_agent->name);

            // Skip if no agents in the input state
            const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
            if (state_list_size == 0) {
                ++streamIdx;
                continue;
            }

#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
            // Error check after unmap vars
            this->singletons->exception.checkError("condition " + func_des->name, streamIdx, this->getStream(streamIdx));
#endif
            // Process agent function condition
            cuda_agent.processFunctionCondition(*func_des, this->singletons->scatter, streamIdx, this->getStream(streamIdx));
        }
        // Increment the stream tracker.
        ++streamIdx;
    }

    streamIdx = 0;
    // Sum the total number of threads being launched in the layer
    totalThreads = 0;
    // for each func function - Loop through to do all mapping of agent and message variables
    for (const auto &func_des : layer->agent_functions) {
        auto func_agent = func_des->parent.lock();
        if (!func_agent) {
            THROW exception::InvalidAgentFunc("Agent function refers to expired agent.");
        }
        flamegpu::util::nvtx::Range map_range{std::string("map" + func_agent->name + "::" + func_des->name).c_str()};

        const detail::CUDAAgent& cuda_agent = getCUDAAgent(func_agent->name);
        const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
        if (state_list_size == 0) {
            ++streamIdx;
            continue;
        }
        // Resize death flag array if necessary
        singletons->scatter.Scan().resize(state_list_size, detail::CUDAScanCompaction::AGENT_DEATH, streamIdx);

        // check if a function has an input message
        if (auto im = func_des->message_input.lock()) {
            std::string inpMessage_name = im->name;
            detail::CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name);
            // Construct PBM here if required!!
            cuda_message.buildIndex(this->singletons->scatter, streamIdx, this->getStream(streamIdx));  // This is synchronous.
            // Map variables after, as index building can swap arrays
            cuda_message.mapReadRuntimeVariables(*func_des, cuda_agent);
        }

        // check if a function has an output message
        if (auto om = func_des->message_output.lock()) {
            std::string outpMessage_name = om->name;
            detail::CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);
            // Resize message list if required
            const unsigned int existingMessages = cuda_message.getTruncateMessageListFlag() ? 0 : cuda_message.getMessageCount();
            cuda_message.resize(existingMessages + state_list_size, this->singletons->scatter, getStream(streamIdx), streamIdx, existingMessages);  // This could have it's internal syncs delayed
            cuda_message.mapWriteRuntimeVariables(*func_des, cuda_agent, state_list_size, getStream(streamIdx));
            singletons->scatter.Scan().resize(state_list_size, detail::CUDAScanCompaction::MESSAGE_OUTPUT, streamIdx);
            // Zero the scan flag that will be written to
            if (func_des->message_output_optional)
                singletons->scatter.Scan().zero_async(detail::CUDAScanCompaction::MESSAGE_OUTPUT, getStream(streamIdx), streamIdx);
                // No Sync, any subsequent use should be in same stream
        }

        // check if a function has an output agent
        if (auto oa = func_des->agent_output.lock()) {
            // This will act as a reserve word
            // which is added to variable hashes for agent creation on device
            detail::CUDAAgent& output_agent = getCUDAAgent(oa->name);

            // Map vars with curve (this allocates/requests enough new buffer space if an existing version is not available/suitable)
            output_agent.mapNewRuntimeVariables_async(cuda_agent, *func_des, state_list_size, this->singletons->scatter, instance_id, getStream(streamIdx), streamIdx);
            // No Sync, any subsequent use should be in same stream
        }

        // Configure runtime access of the functions variables within the FLAME_API object
        cuda_agent.mapRuntimeVariables(*func_des, instance_id);

        // Zero the scan flag that will be written to
        if (func_des->has_agent_death) {
            singletons->scatter.Scan().zero_async(detail::CUDAScanCompaction::AGENT_DEATH, getStream(streamIdx), streamIdx);
            // No Sync, any subsequent use should be in same stream
        }

        // Push function's RTC cache to device if using RTC
        if (!func_des->rtc_func_name.empty()) {
            auto& rtc_header = cuda_agent.getRTCHeader(func_des->name);
            // Sync EnvManager's RTC cache with RTC header's cache
            rtc_header.updateEnvCache(singletons->environment->getHostBuffer(), singletons->environment->getBufferLen());
            // Push RTC header's cache to device
            rtc_header.updateDevice_async(cuda_agent.getRTCInstantiation(func_des->name), getStream(streamIdx));
        } else {
            auto& curve = cuda_agent.getCurve(func_des->name);
            curve.updateDevice_async(this->getStream(streamIdx));
        }
        // No sync, kernel launch should be in the same stream

        // Count total threads being launched
        totalThreads += cuda_agent.getStateSize(func_des->initial_state);
        ++streamIdx;
    }

    // If any kernel needs to be executed, do so, by checking the number of threads from before.
    if (totalThreads > 0) {
        // Ensure RandomManager is the correct size to accommodate all threads to be launched
        detail::curandState *d_rng = singletons->rng.resize(totalThreads, getStream(0));
        // Total threads is now used to provide kernel launches an offset to thread-safe thread-index
        totalThreads = 0;
        streamIdx = 0;

        // for each func function - Loop through to launch all agent functions
        for (const auto &func_des : layer->agent_functions) {
            auto func_agent = func_des->parent.lock();
            if (!func_agent) {
                THROW exception::InvalidAgentFunc("Agent function refers to expired agent.");
            }
            flamegpu::util::nvtx::Range func_range{std::string(func_agent->name + "::" + func_des->name).c_str()};
            const void *d_in_messagelist_metadata = nullptr;
            const void *d_out_messagelist_metadata = nullptr;
            id_t *d_agentOut_nextID = nullptr;
            std::string agent_name = func_agent->name;
            std::string func_name = func_des->name;

            // check if a function has an input message
            if (auto im = func_des->message_input.lock()) {
                std::string inpMessage_name = im->name;
                const detail::CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name);

                d_in_messagelist_metadata = cuda_message.getMetaDataDevicePtr();
            }

            // check if a function has an output message
            if (auto om = func_des->message_output.lock()) {
                std::string outpMessage_name = om->name;
                const detail::CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);

                d_out_messagelist_metadata = cuda_message.getMetaDataDevicePtr();
            }

            // check if a function has agent output
            if (auto oa = func_des->agent_output.lock()) {
                detail::CUDAAgent& output_agent = getCUDAAgent(oa->name);
                d_agentOut_nextID = output_agent.getDeviceNextID();
            }

            const detail::CUDAAgent& cuda_agent = getCUDAAgent(agent_name);

            const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
            if (state_list_size == 0) {
                ++streamIdx;
                continue;
            }

            int blockSize = 0;  // The launch configurator returned block size
            int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
            int gridSize = 0;  // The actual grid size needed, based on input size

            // Agent function kernel wrapper args
            detail::curandState *t_rng = d_rng + totalThreads;
            unsigned int *scanFlag_agentDeath = func_des->has_agent_death ? this->singletons->scatter.Scan().Config(detail::CUDAScanCompaction::Type::AGENT_DEATH, streamIdx).d_ptrs.scan_flag : nullptr;
            unsigned int *scanFlag_messageOutput = this->singletons->scatter.Scan().Config(detail::CUDAScanCompaction::Type::MESSAGE_OUTPUT, streamIdx).d_ptrs.scan_flag;
            unsigned int *scanFlag_agentOutput = this->singletons->scatter.Scan().Config(detail::CUDAScanCompaction::Type::AGENT_OUTPUT, streamIdx).d_ptrs.scan_flag;
    #if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
            auto *error_buffer = this->singletons->exception.getDevicePtr(streamIdx, this->getStream(streamIdx));
    #endif

            if (func_des->func) {   // compile time specified agent function launch
                // calculate the grid block size for main agent function
                cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func_des->func, 0, state_list_size);
                //! Round up according to CUDAAgent state list size
                gridSize = (state_list_size + blockSize - 1) / blockSize;

                (func_des->func) << <gridSize, blockSize, 0, this->getStream(streamIdx) >> > (
    #if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                    error_buffer,
    #endif
                    cuda_agent.getCurve(func_des->name).getDevicePtr(),
                    this->singletons->strings.getDeviceString(func_agent->name),
                    this->singletons->strings.getDeviceString(func_des->initial_state),
                    static_cast<const char *>(this->singletons->environment->getDeviceBuffer()),
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
                const jitify2::KernelData& instance = cuda_agent.getRTCInstantiation(func_name);
                // calculate the grid block size for main agent function
                CUfunction cu_func = (CUfunction)instance.function();
                gpuErrchkDriverAPI(cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cu_func, 0, 0, state_list_size));
                //! Round up according to CUDAAgent state list size
                gridSize = (state_list_size + blockSize - 1) / blockSize;
                // launch the kernel
                jitify2::ErrorMsg a = instance.configure(gridSize, blockSize, 0, this->getStream(streamIdx))->launch_raw({
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                    reinterpret_cast<void*>(&error_buffer),
#endif
                    reinterpret_cast<void*>(&d_agentOut_nextID),
                    const_cast<void*>(reinterpret_cast<const void*>(&state_list_size)),
                    const_cast<void*>(reinterpret_cast<const void*>(&d_in_messagelist_metadata)),
                    const_cast<void*>(reinterpret_cast<const void*>(&d_out_messagelist_metadata)),
                    const_cast<void*>(reinterpret_cast<const void*>(&t_rng)),
                    reinterpret_cast<void*>(&scanFlag_agentDeath),
                    reinterpret_cast<void*>(&scanFlag_messageOutput),
                    reinterpret_cast<void*>(&scanFlag_agentOutput)});
                if (!a.empty()) {
                    THROW exception::InvalidAgentFunc("There was a problem launching the runtime agent function '%s': %s", func_name.c_str(), a.c_str());
                }
                gpuErrchkLaunch();
            }
            totalThreads += state_list_size;
            ++streamIdx;
        }
    }

    streamIdx = 0;
    // for each func function - Loop through to un-map all agent and message variables
    for (const auto &func_des : layer->agent_functions) {
        auto func_agent = func_des->parent.lock();
        if (!func_agent) {
            THROW exception::InvalidAgentFunc("Agent function refers to expired agent.");
        }
        flamegpu::util::nvtx::Range unmap_range{std::string("unmap" + func_agent->name + "::" + func_des->name).c_str()};
        detail::CUDAAgent& cuda_agent = getCUDAAgent(func_agent->name);

        const unsigned int state_list_size = cuda_agent.getStateSize(func_des->initial_state);
        // If agent function wasn't executed, these are redundant
        if (state_list_size > 0) {
            // check if a function has an output message
            if (auto om = func_des->message_output.lock()) {
                std::string outpMessage_name = om->name;
                detail::CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);
                cuda_message.swap(func_des->message_output_optional, state_list_size, this->singletons->scatter, getStream(streamIdx), streamIdx);
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
                detail::CUDAAgent& output_agent = getCUDAAgent(oa->name);
                // Scatter the agent birth
                output_agent.scatterNew(*func_des, state_list_size, this->singletons->scatter, streamIdx, this->getStream(streamIdx));
                output_agent.releaseNewBuffer(*func_des);
            }

#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
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

#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    // Reset macro-environment read-write flags
    // Note this does not synchronise threads, it relies on synchronizeAllStreams() post host fns
    macro_env->resetFlagsAsync(streams);
#endif

    // Synchronise  after the host layer functions to ensure that the device is up to date? This can potentially be removed.
    this->synchronizeAllStreams();
}

void CUDASimulation::layerHostFunctions(const std::shared_ptr<LayerData>& layer, const unsigned int layerIndex) {
    flamegpu::util::nvtx::Range range{"CUDASimulation::stepHostFunctions"};
    // Execute all host functions attached to layer
    // TODO: Concurrency?
    assert(host_api);
    for (auto &stepFn : layer->host_functions) {
        flamegpu::util::nvtx::Range hostfn_range{"hostFunc"};
        stepFn(this->host_api.get());
    }
    // Execute all host function callbacks attached to layer
    for (auto &stepFn : layer->host_functions_callbacks) {
        flamegpu::util::nvtx::Range hostfncallback_range{"hostFunc_swig"};
        stepFn->run(this->host_api.get());
    }
    if (layer->host_functions.size() || (layer->host_functions_callbacks.size())) {
        // If we have host layer functions, we might have host agent creation
        // Sync any device vectors, before performing host agent creation
        for (auto& ca : agent_map) {
            ca.second->resetPopulationVecs();
        }
        // @todo - What is the most appropriate stream to use here?
        processHostAgentCreation(0);
        // If we have host layer functions, a static graph may need updating
        for (auto& [name, sg] : directed_graph_map) {
            sg->syncDevice_async(singletons->scatter, 0, getStream(0));
        }
    }
}

void CUDASimulation::stepStepFunctions() {
    flamegpu::util::nvtx::Range range{"CUDASimulation::step::StepFunctions"};
    // Execute step functions
    for (auto &stepFn : model->stepFunctions) {
        flamegpu::util::nvtx::Range step_range{"stepFunc"};
        stepFn(this->host_api.get());
    }
    // Execute step function callbacks
    for (auto &stepFn : model->stepFunctionCallbacks) {
        flamegpu::util::nvtx::Range callback_range{"stepFunc_swig"};
        stepFn->run(this->host_api.get());
    }
    // If we have step functions, we might have host agent creation
    if (model->stepFunctions.size() || model->stepFunctionCallbacks.size()) {
        // Sync any device vectors, before performing host agent creation
        for (auto &ca : agent_map) {
            ca.second->resetPopulationVecs();
        }
        processHostAgentCreation(0);
        // If we have host layer functions, a static graph may need updating
        for (auto& [name, sg] : directed_graph_map) {
            sg->syncDevice_async(singletons->scatter, 0, getStream(0));
        }
    }
}

bool CUDASimulation::stepExitConditions() {
    flamegpu::util::nvtx::Range range{"CUDASimulation::stepExitConditions"};
    // Track if any exit conditions were successful. Use this to control return code and skipsteps.
    // early returning makes timing/stepCounter logic more complicated.
    bool exitConditionExit = false;

    // Execute exit conditions
    for (auto &exitCdns : model->exitConditions) {
        if (exitCdns(this->host_api.get()) == EXIT) {
            #ifdef FLAMEGPU_VISUALISATION
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
                #ifdef FLAMEGPU_VISUALISATION
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
        // If we have host layer functions, a static graph may need updating
        for (auto& [name, sg] : directed_graph_map) {
            sg->syncDevice_async(singletons->scatter, 0, getStream(0));
        }

        #ifdef FLAMEGPU_VISUALISATION
            if (visualisation) {
                visualisation->updateBuffers(step_count+1);
            }
        #endif
    }
    return exitConditionExit;
}

void CUDASimulation::simulate() {
    flamegpu::util::nvtx::Range range{"CUDASimulation::simulate"};

    // Ensure there is work to do.
    if (agent_map.size() == 0) {
        THROW exception::InvalidCudaAgentMapSize("Simulation has no agents, in CUDASimulation::simulate().");
    }

    // Ensure singletons have been initialised
    initialiseSingletons();

    // Create the event timing object, using an appropriate timer implementation.
    std::unique_ptr<detail::Timer> simulationTimer = getDriverAppropriateTimer(getCUDAConfig().is_ensemble || getCUDAConfig().is_submodel);
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

    // Reset and log initial state to step log 0
    resetLog();
    processStepLog(this->elapsedSecondsRTCInitialisation + this->elapsedSecondsInitFunctions);

    #ifdef FLAMEGPU_VISUALISATION
    // Pre step-loop visualisation update
    if (visualisation) {
        visualisation->updateBuffers();
    }
    visualiser::ModelVis mv(visualisation, isSWIG);
    #endif

    // Run the required number of simulation steps.
    for (unsigned int i = 0; getSimulationConfig().steps == 0 ? true : i < getSimulationConfig().steps; i++) {
        // Run the step
        bool continueSimulation = step();
        if (!continueSimulation) {
            break;
        }
        #ifdef FLAMEGPU_VISUALISATION

        // Special case, if steps == 0 and visualisation has been closed
        if (getSimulationConfig().steps == 0 &&
            visualisation && !mv.isRunning()) {
            mv.join();  // Vis exists in separate thread, make sure it has actually exited
            break;
        }
        #endif
    }

    // Exit functions
    this->exitFunctions();

    // Sync visualistaion after the exit functions
    #ifdef FLAMEGPU_VISUALISATION
    if (visualisation) {
        visualisation->updateBuffers();
    }
    #endif

    // Record, store and output the elapsed simulation time
    simulationTimer->stop();
    elapsedSecondsSimulation = simulationTimer->getElapsedSeconds();
    if (getSimulationConfig().timing || getSimulationConfig().verbosity >= Verbosity::Verbose) {
        // Resolution is 0.5 microseconds, so print to 1 us.
        fprintf(stdout, "Total Processing time: %.6f s\n", elapsedSecondsSimulation);
    }
    processExitLog();

    // Send Telemetry if not submodel
    if (getSimulationConfig().telemetry && !getCUDAConfig().is_submodel) {
        // Generate some payload items
        std::map<std::string, std::string> payload_items;
        payload_items["GPUDevices"] = flamegpu::detail::compute_capability::getDeviceName(deviceInitialised);
        payload_items["SimTime(s)"] = std::to_string(elapsedSecondsSimulation);
        #if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && defined(__CUDACC_VER_BUILD__)
            payload_items["NVCCVersion"] = std::to_string(__CUDACC_VER_MAJOR__) + "." + std::to_string(__CUDACC_VER_MINOR__) + "." + std::to_string(__CUDACC_VER_BUILD__);
        #endif
        // generate telemtry data
        std::string telemetry_data = flamegpu::io::Telemetry::generateData("simulation-run", payload_items, isSWIG);
        bool telemetrySuccess = flamegpu::io::Telemetry::sendData(telemetry_data);
        // If verbose, print either a successful send, or a misc warning.
        if (getSimulationConfig().verbosity >= Verbosity::Verbose) {
            if (telemetrySuccess) {
                fprintf(stdout, "Telemetry packet sent to '%s' json was: %s\n", flamegpu::io::Telemetry::TELEMETRY_ENDPOINT, telemetry_data.c_str());
            } else {
                fprintf(stderr, "Warning: Usage statistics for CUDASimulation failed to send with json: %s\n", telemetry_data.c_str());
            }
        }
    } else {
        // Encourage users who have opted out to opt back in, unless suppressed.
        if ((getSimulationConfig().verbosity > Verbosity::Quiet))
            flamegpu::io::Telemetry::encourageUsage();
    }

    // Export logs
    if (!SimulationConfig().step_log_file.empty())
        exportLog(SimulationConfig().step_log_file, true, false, step_log_config && step_log_config->log_timing, false);
    if (!SimulationConfig().exit_log_file.empty())
        exportLog(SimulationConfig().exit_log_file, false, true, false, exit_log_config && exit_log_config->log_timing);
    if (!SimulationConfig().common_log_file.empty())
        exportLog(SimulationConfig().common_log_file, true, true, step_log_config && step_log_config->log_timing, exit_log_config && exit_log_config->log_timing);
}


void CUDASimulation::simulate(const RunPlan& plan) {
    // Validate that RunPlan is for same ModelDesc
    // RunPlan only holds a copy of env, so we must compare those
    if (*plan.environment != model->environment->properties) {
        THROW exception::InvalidArgument("RunPlan's associated environment does not match the ModelDescription's environment, "
        "in CUDASimulation::simulate(RunPlan)\n");
    }
    // Backup config
    const uint64_t t_random_seed = SimulationConfig().random_seed;
    const unsigned int t_steps = SimulationConfig().steps;
    // Temp override config
    SimulationConfig().steps = plan.getSteps();
    SimulationConfig().random_seed = plan.getRandomSimulationSeed();
    // Ensure singletons have been initialised (so env actually exists in mgr)
    initialiseSingletons();
    // Override environment properties
    for (auto& ovrd : plan.property_overrides) {
        singletons->environment->setPropertyDirect(ovrd.first, static_cast<char *>(ovrd.second.ptr));
    }
    // Call regular simulate
    simulate();
    // Reset config
    SimulationConfig().random_seed = t_random_seed;
    SimulationConfig().steps = t_steps;
}

void CUDASimulation::reset(bool submodelReset) {
    // Reset step counter
    resetStepCounter();

    if (singletonsInitialised) {
        // Reset environment properties
        singletons->environment->resetModel(*model->environment);

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
        a.second->setPBMConstructionRequiredFlag();
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
    flamegpu::util::nvtx::Range range{"CUDASimulation::setPopulationData()"};
    auto it = agent_map.find(population.getAgentName());
    if (it == agent_map.end()) {
        THROW exception::InvalidAgent("Agent '%s' was not found, "
            "in CUDASimulation::setPopulationData()",
            population.getAgentName().c_str());
    }
    // This call hierarchy validates agent desc matches and state is valid
    it->second->setPopulationData(population, state_name, this->singletons->scatter, 0, getStream(0));  // Streamid shouldn't matter here
#ifdef FLAMEGPU_VISUALISATION
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
    flamegpu::util::nvtx::Range range{"CUDASimulation::getPopulationData()"};
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
detail::CUDAAgent& CUDASimulation::getCUDAAgent(const std::string& agent_name) const {
    CUDAAgentMap::const_iterator it;
    it = agent_map.find(agent_name);

    if (it == agent_map.end()) {
        THROW exception::InvalidCudaAgent("CUDA agent ('%s') not found, in CUDASimulation::getCUDAAgent().",
            agent_name.c_str());
    }

    return *(it->second);
}
detail::CUDAMessage& CUDASimulation::getCUDAMessage(const std::string& message_name) const {
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
    flamegpu::util::nvtx::Range range{"applyConfig_derived"};

    // Handle console_mode
#ifdef FLAMEGPU_VISUALISATION
    if (visualisation) {
        visualiser::ModelVis mv(visualisation, isSWIG);
        if (getSimulationConfig().console_mode) {
            mv.deactivate();
        } else {
            visualisation->updateRandomSeed();
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
    if (!detail::compute_capability::checkComputeCapability(static_cast<int>(config.device_id))) {
        int min_cc = detail::compute_capability::minimumCompiledComputeCapability();
        int cc = detail::compute_capability::getComputeCapability(static_cast<int>(config.device_id));
        THROW exception::InvalidCUDAComputeCapability("Error application compiled for CUDA Compute Capability %d and above. Device %u is compute capability %d. Rebuild for SM_%d.", min_cc, config.device_id, cc, cc);
    }

    cudaStatus = cudaSetDevice(static_cast<int>(config.device_id));
    if (cudaStatus != cudaSuccess) {
        THROW exception::InvalidCUDAdevice("Unknown error setting CUDA device to '%d'. (%d available)", config.device_id, device_count);
    }
    // Call cudaFree to initialise the context early
    gpuErrchk(cudaFree(nullptr));

    // Get the unique ID of the current cuda context, to prevent unsafe stream destruction post flamegpu::cleanup for CUDA 12+
#if __CUDACC_VER_MAJOR__ >= 12
    this->cudaContextID = flamegpu::detail::cuda::cuGetCurrentContextUniqueID();
#endif  // __CUDACC_VER_MAJOR__ >= 12

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

void CUDASimulation::reseed(const uint64_t seed) {
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

void CUDASimulation::initialiseSingletons() {
    // Only do this once.
    if (!singletonsInitialised) {
        // If the device has not been specified, also check the compute capability is OK
        // Check the compute capability of the device, throw an exception if not valid for the executable.
        if (!detail::compute_capability::checkComputeCapability(static_cast<int>(config.device_id))) {
            int min_cc = detail::compute_capability::minimumCompiledComputeCapability();
            int cc = detail::compute_capability::getComputeCapability(static_cast<int>(config.device_id));
            THROW exception::InvalidCUDAComputeCapability("Error application compiled for CUDA Compute Capability %d and above. Device %u is compute capability %d. Rebuild for SM_%d.", min_cc, config.device_id, cc, cc);
        }
        gpuErrchk(cudaGetDevice(&deviceInitialised));
        // Get references to all required singleton and store in the instance.
        singletons = new Singletons((!submodel)?
            detail::EnvironmentManager::create(*model->environment) :
            detail::EnvironmentManager::create(*model->environment, mastermodel->singletons->environment, *submodel->subenvironment));

        // Reinitialise random for this simulation instance
        singletons->rng.reseed(getSimulationConfig().random_seed);

        cudaStream_t stream_0 = getStream(0);

        // Pass created RandomManager to host api
        host_api = std::make_unique<HostAPI>(*this, singletons->rng, singletons->scatter, agentOffsets, agentData, singletons->environment, macro_env, directed_graph_map, 0, stream_0);  // Host fns are currently all serial

        for (auto &cm : message_map) {
            cm.second->init(singletons->scatter, 0, stream_0);
        }

        // Populate the environment properties
        if (!submodel) {
            for (const auto &it : model->environment->directed_graphs) {
                // insert into map using value_type and store a reference to the map pair
                directed_graph_map.emplace(it.first, std::make_shared<detail::CUDAEnvironmentDirectedGraphBuffers>(*it.second));
            }
            macro_env->init(stream_0);
        } else {
            for (const auto& it : model->environment->directed_graphs) {
                const auto sub_it = submodel->subenvironment->directed_graphs.find(it.first);
                if (sub_it != submodel->subenvironment->directed_graphs.end()) {
                    // if is linked to parent graph, instead store that graph's ptr
                    // insert into map using value_type and store a reference to the map pair
                    const auto master_graph_it = mastermodel->directed_graph_map.find(sub_it->second);
                    if (master_graph_it != mastermodel->directed_graph_map.end()) {
                        directed_graph_map.emplace(it.first, master_graph_it->second);
                    } else {
                        THROW exception::UnknownInternalError("Failed to find master model '%s's directed graph '%s' when intialising submodel '%s', this should not happen, please report this as a bug.\n",
                            mastermodel->model->name.c_str(), sub_it->second.c_str(), this->model->name.c_str());
                    }
                } else {
                    // insert into map using value_type and store a reference to the map pair
                    directed_graph_map.emplace(it.first, std::make_shared<detail::CUDAEnvironmentDirectedGraphBuffers>(*it.second));
                }
            }
            macro_env->init(*submodel->subenvironment, mastermodel->macro_env, stream_0);
        }

        // Populate device strings
        for (const auto& [agent_name, agent] : model->agents) {
            singletons->strings.registerDeviceString(agent_name);
            for (const auto& state_name : agent->states) {
                singletons->strings.registerDeviceString(state_name);
            }
        }

        // Propagate singleton init to submodels
        for (auto &sm : submodel_map) {
            sm.second->initialiseSingletons();
        }

        // Store the WDDM/TCC driver mode status, for timer class decisions. Result is cached in the anon namespace to avoid multiple queries
        deviceUsingWDDM = detail::wddm::deviceIsWDDM();

#ifdef FLAMEGPU_VISUALISATION
        if (visualisation) {
            visualisation->updateRandomSeed();  // Incase user hasn't triggered applyConfig()
            visualisation->registerEnvProperties();
        }
#endif

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
    initMacroEnvironment();

    // Ensure there are enough streams to execute the layer.
    // Taking into consideration if in-layer concurrency is disabled or not.
    unsigned int nStreams = getMaximumLayerWidth();
    this->createStreams(nStreams);

    // Ensure RTC is set up.
    initialiseRTC();

#ifdef FLAMEGPU_VISUALISATION
    if (visualisation) {
        visualisation->hookVis(visualisation, directed_graph_map);
    }
#endif
}

void CUDASimulation::initialiseRTC() {
    // Only do this once.
    if (!rtcInitialised) {
        flamegpu::util::nvtx::Range range{"CUDASimulation::initialiseRTC"};
        std::unique_ptr<detail::Timer> rtcTimer(new detail::SteadyClockTimer());
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
                    a_it->second->addInstantitateRTCFunction(*it_f->second, singletons->environment, macro_env, directed_graph_map);
                } else {
                    // Init curve for non-rtc functions
                    a_it->second->addInstantitateFunction(*it_f->second, singletons->environment, macro_env, directed_graph_map);
                }
                // check rtc source to see if the function condition is an rtc condition
                if (!it_f->second->rtc_condition_source.empty()) {
                    // create CUDA agent RTC function condition by calling addInstantitateRTCFunction on CUDAAgent with AgentFunctionData
                    a_it->second->addInstantitateRTCFunction(*it_f->second, singletons->environment, macro_env, directed_graph_map, true);
                } else if (it_f->second->condition) {
                    // Init curve for non-rtc function conditionss
                    a_it->second->addInstantitateFunction(*it_f->second, singletons->environment, macro_env, directed_graph_map, true);
                }
            }
        }

        rtcInitialised = true;

        // Record, store and output the elapsed time of the step.
        rtcTimer->stop();
        this->elapsedSecondsRTCInitialisation = rtcTimer->getElapsedSeconds();
        if (getSimulationConfig().timing) {
            fprintf(stdout, "RTC Initialisation Processing time: %.6f s\n", this->elapsedSecondsRTCInitialisation);
        }
    }
}

CUDASimulation::Config &CUDASimulation::CUDAConfig() {
    return config;
}
const CUDASimulation::Config &CUDASimulation::getCUDAConfig() const {
    return config;
}
#ifdef FLAMEGPU_VISUALISATION
visualiser::ModelVis CUDASimulation::getVisualisation() {
    if (!visualisation) {
        visualisation = std::make_shared<visualiser::ModelVisData>(*this);
        // If visualisation is init after sim has been init ensure graphs are linked to vis
        if (directed_graph_map.size()) {
            visualisation->hookVis(visualisation, directed_graph_map);
        }
    }
    return visualiser::ModelVis(visualisation, isSWIG);
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

void CUDASimulation::processHostAgentCreation(const unsigned int streamId) {
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
                            gpuErrchk(flamegpu::detail::cuda::cudaFree(dt_buff));
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
        gpuErrchk(flamegpu::detail::cuda::cudaFree(dt_buff));
    }
}

void CUDASimulation::incrementStepCounter() {
    this->step_count++;
    this->singletons->environment->setProperty<unsigned int>("_stepCount", this->step_count);
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
    if (!singletons || !singletons->environment) {
        THROW exception::UnknownInternalError("CUDASimulation::initEnvironmentMgr() called before singletons member initialised.");
    }

    // Set any properties loaded from file during arg parse stage
    for (const auto &prop : env_init) {
        const std::string np = prop.first;
        const auto it = singletons->environment->properties.find(np);
        if (it == singletons->environment->properties.end()) {
            THROW exception::InvalidEnvProperty("Environment init data contains unexpected environment property '%s', "
                "in CUDASimulation::initEnvironmentMgr()\n", prop.first.c_str());
        }
        if (prop.second.type != it->second.type) {
            THROW exception::InvalidEnvProperty("Environment init data contains environment property '%s' with type mismatch '%s' != '%s', "
                "this should have been caught during file parsing, "
                "in CUDASimulation::initEnvironmentMgr()\n", prop.first.c_str(), prop.second.type.name(), it->second.type.name());
        } else if (prop.second.elements != it->second.elements) {
            THROW exception::InvalidEnvProperty("Environment init data contains environment property '%s' with type length mismatch '%u' != '%u', "
                "this should have been caught during file parsing, "
                "in CUDASimulation::initEnvironmentMgr()\n", prop.first.c_str(), prop.second.elements, it->second.elements);
        } else {
            singletons->environment->setPropertyDirect(np, static_cast<char*>(prop.second.ptr));
        }
    }
    // Clear init
    env_init.clear();
}
void CUDASimulation::initMacroEnvironment() {
    const auto &mp_map = macro_env->getPropertiesMap();
    if (mp_map.size() && !mp_map.begin()->second.d_ptr) {
        THROW exception::UnknownInternalError("CUDASimulation::initMacroEnvironment() called before macro environment initialised.");
    }
    const cudaStream_t stream = getStream(0);
    // Set any properties loaded from file during arg parse stage
    for (const auto &[name, buff] : macro_env_init) {
        const auto it =  mp_map.find(name);
        if (it == mp_map.end()) {
            THROW exception::InvalidEnvProperty("Macro environment init data contains unexpected property '%s', "
                "in CUDASimulation::initMacroEnvironment()\n", name.c_str());
        }
        // @todo Not tracking type to validate that here?
        const unsigned int elements = std::accumulate(it->second.elements.begin(), it->second.elements.end(), 1, std::multiplies<unsigned int>());
        if (elements * it->second.type_size != buff.size()) {
            THROW exception::InvalidEnvProperty("Macro environment init data contains property '%s' with buffer length mismatch '%u' != '%u', "
                "this should have been caught during file parsing, "
                "in CUDASimulation::initMacroEnvironment()\n", name.c_str(), static_cast<unsigned int>(buff.size()), elements * it->second.type_size);
        } else {
            gpuErrchk(cudaMemcpyAsync(it->second.d_ptr, buff.data(), buff.size() * sizeof(char), cudaMemcpyHostToDevice, stream));
        }
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    // Clear init
    macro_env_init.clear();
}
void CUDASimulation::resetLog() {
    // Track previous device id, so we can avoid costly request for device properties if not required
    static int previous_device_id = -1;
    run_log->step.clear();
    run_log->exit = ExitLogFrame();
    run_log->random_seed = SimulationConfig().random_seed;
    run_log->step_log_frequency = step_log_config ? step_log_config->frequency : 0;
    if (run_log->performance_specs.device_name.empty() || CUDAConfig().device_id != previous_device_id) {
        cudaDeviceProp d_props = {};
        gpuErrchk(cudaGetDeviceProperties(&d_props, CUDAConfig().device_id));
        run_log->performance_specs.device_name = d_props.name;
        previous_device_id = CUDAConfig().device_id;
    }
    gpuErrchk(cudaDeviceGetAttribute(&run_log->performance_specs.device_cc_major, cudaDevAttrComputeCapabilityMajor, CUDAConfig().device_id));
    gpuErrchk(cudaDeviceGetAttribute(&run_log->performance_specs.device_cc_minor,  cudaDevAttrComputeCapabilityMinor, CUDAConfig().device_id));
    gpuErrchk(cudaRuntimeGetVersion(&run_log->performance_specs.cuda_version));
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    run_log->performance_specs.seatbelts = true;
#else
    run_log->performance_specs.seatbelts = false;
#endif
    run_log->performance_specs.flamegpu_version = VERSION_FULL;
}
void CUDASimulation::processStepLog(const double step_time_seconds) {
    if (!step_log_config)
        return;
    if (step_count % step_log_config->frequency != 0)
        return;
    // Iterate members of step log to build the step log frame
    std::map<std::string, detail::Any> environment_log;
    for (const auto &prop_name : step_log_config->environment) {
        // Fetch the named environment prop
        environment_log.emplace(prop_name, singletons->environment->getPropertyAny(prop_name));
    }
    std::map<util::StringPair, std::pair<std::map<LoggingConfig::NameReductionFn, detail::Any>, unsigned int>> agents_log;
    for (const auto &name_state : step_log_config->agents) {
        // Create the named sub map
        const std::string &agent_name = name_state.first.first;
        const std::string &agent_state = name_state.first.second;
        HostAgentAPI host_agent = host_api->agent(agent_name, agent_state);
        auto &agent_state_log = agents_log.emplace(name_state.first, std::make_pair(std::map<LoggingConfig::NameReductionFn, detail::Any>(), UINT_MAX)).first->second;
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
    run_log->step.push_back(StepLogFrame(std::move(environment_log), std::move(agents_log), step_count));
    run_log->step.back().step_time = step_time_seconds;
}

void CUDASimulation::processExitLog() {
    if (!exit_log_config)
        return;
    // Iterate members of step log to build the step log frame
    std::map<std::string, detail::Any> environment_log;
    for (const auto &prop_name : exit_log_config->environment) {
        // Fetch the named environment prop
        environment_log.emplace(prop_name, singletons->environment->getPropertyAny(prop_name));
    }
    std::map<util::StringPair, std::pair<std::map<LoggingConfig::NameReductionFn, detail::Any>, unsigned int>> agents_log;
    for (const auto &name_state : exit_log_config->agents) {
        // Create the named sub map
        const std::string &agent_name = name_state.first.first;
        const std::string &agent_state = name_state.first.second;
        HostAgentAPI host_agent = host_api->agent(agent_name, agent_state);
        auto &agent_state_log = agents_log.emplace(name_state.first, std::make_pair(std::map<LoggingConfig::NameReductionFn, detail::Any>(), UINT_MAX)).first->second;
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
    run_log->exit = ExitLogFrame(std::move(environment_log), std::move(agents_log), step_count);
    // Add the timing info
    run_log->exit.rtc_time = getElapsedTimeRTCInitialisation();
    run_log->exit.init_time = getElapsedTimeInitFunctions();
    run_log->exit.exit_time = getElapsedTimeExitFunctions();
    run_log->exit.total_time = getElapsedTimeSimulation();
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
    // early exit if there are no streams to reset
    if (streams.size() == 0) {
        return;
    }
    /*
    This method is called by ~CUDASimulation(), which may be after a device reset and / or CUDA shutdown (if static, or if GC'd by python implementation)
    cudaStreamDestroy and cudaStreamQuery under linux with CUDA 11.8 (and potentially others) would occasionally segfault after a reset, as passing invalid stream handles to various methods is UB, so these methods cannot be used to check for safe destruction.
    The Driver API equivalents include the same UB.
    Instead, we can use the CUDA driver API to check the primary context is correct / valid for the device, and if it is attempt to destroy the stream, which works for CUDA 11.x.
    CUDA 12.x however claims the ctx is active for the specified device, even after a reset. Getting the active context returns the same handle after a reset, so we cannot store and compare CUcontexts, but CUDA 12 does include a new method to get the unique ID for a context which we can store and check is a match.
    */
    bool safeToDestroy = flamegpu::detail::cuda::cuDevicePrimaryContextIsActive(deviceInitialised);
    #if __CUDACC_VER_MAJOR__ >= 12
        std::uint64_t currentContextID = flamegpu::detail::cuda::cuGetCurrentContextUniqueID();
        safeToDestroy = safeToDestroy && currentContextID == this->cudaContextID;
    #endif  // __CUDACC_VER_MAJOR__ >= 12
    if (safeToDestroy) {
        // Destroy streams.
        for (auto stream : streams) {
            gpuErrchk(cudaStreamDestroy(stream));
        }
    }
    streams.clear();
}
void CUDASimulation::safeDestroyJitify() {
    // Can't have done any RTC if device is not init
    if (deviceInitialised == -1)
        return;
    // See note in destroyStreams()
    bool safeToDestroy = flamegpu::detail::cuda::cuDevicePrimaryContextIsActive(deviceInitialised);
#if __CUDACC_VER_MAJOR__ >= 12
    std::uint64_t currentContextID = flamegpu::detail::cuda::cuGetCurrentContextUniqueID();
    safeToDestroy = safeToDestroy && currentContextID == this->cudaContextID;
#endif  // __CUDACC_VER_MAJOR__ >= 12
    // Destroy any RTC functions
    for (auto &[_, ca] : agent_map) {
        ca->destroyRTCInstances(safeToDestroy);
    }
}

void CUDASimulation::synchronizeAllStreams() {
    // Sync streams.
    for (auto stream : streams) {
        gpuErrchk(cudaStreamSynchronize(stream));
    }
}

std::shared_ptr<detail::EnvironmentManager> CUDASimulation::getEnvironment() const {
    if (singletons)
        return singletons->environment;
    return nullptr;
}
std::shared_ptr<const detail::CUDAMacroEnvironment> CUDASimulation::getMacroEnvironment() const {
    return macro_env;
}
void CUDASimulation::assignAgentIDs() {
    flamegpu::util::nvtx::Range range{"CUDASimulation::assignAgentIDs"};
    if (!agent_ids_have_init) {
        // Ensure singletons have been initialised
        initialiseSingletons();

        for (auto &a : agent_map) {
            a.second->assignIDs(*host_api, singletons->scatter, getStream(0), 0);  // This could be made concurrent, 1 stream per agent
        }
        agent_ids_have_init = true;
    }
}

}  // namespace flamegpu
