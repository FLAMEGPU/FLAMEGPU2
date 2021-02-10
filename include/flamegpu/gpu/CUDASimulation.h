#ifndef INCLUDE_FLAMEGPU_GPU_CUDASIMULATION_H_
#define INCLUDE_FLAMEGPU_GPU_CUDASIMULATION_H_
#include <atomic>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>

#include "flamegpu/exception/FGPUDeviceException.h"
#include "flamegpu/sim/Simulation.h"

#include "flamegpu/gpu/CUDAScatter.h"
#include "flamegpu/gpu/CUDAEnsemble.h"
#include "flamegpu/runtime/utility/RandomManager.cuh"
#include "flamegpu/runtime/flamegpu_host_new_agent_api.h"
#include "flamegpu/visualiser/ModelVis.h"

#ifdef _MSC_VER
#pragma warning(push, 2)
#include "jitify/jitify.hpp"
#pragma warning(pop)
#else
#include "jitify/jitify.hpp"
#endif

class CUDAAgent;
class CUDAMessage;
class LoggingConfig;
class StepLoggingConfig;

struct RunLog;

/**
 * CUDA runner for Simulation interface
 * Executes a FGPU2 model using GPU
 */
class CUDASimulation : public Simulation {
    /**
     * Requires internal access to scan/scatter singletons
     */
    friend class HostAgentInstance;
    friend class SimRunner;
    /**
     * Map of a number of CUDA agents by name.
     * The CUDA agents are responsible for allocating and managing all the device memory
     */
    typedef std::unordered_map<std::string, std::unique_ptr<CUDAAgent>> CUDAAgentMap;
    /**
     * Map of a number of CUDA messages by name.
     * The CUDA messages are responsible for allocating and managing all the device memory
     */
    typedef std::unordered_map<std::string, std::unique_ptr<CUDAMessage>> CUDAMessageMap;
    /**
     * Map of a number of CUDA sub models by name.
     * The CUDA submodels are responsible for allocating and managing all the device memory of non mapped agent vars
     * Ordered is used, so that random seed mutation always occurs same order.
     */
    typedef std::map<std::string, std::unique_ptr<CUDASimulation>> CUDASubModelMap;

 public:
     /**
      * CUDA runner specific config
      */
    struct Config {
        /**
         * GPU to execute model on
         * Defaults to device 0, this is most performant device as detected by CUDA
         */
         int device_id = 0;
    };
    /**
     * Initialise cuda runner
     * Allocates memory for agents/messages, copies environment properties to device etc
     * If provided, you can pass runtime arguments to this constructor, to automatically call inititialise()
     * This is not required, you can call inititialise() manually later, or not at all.
     * @param model The model description to initialise the runner to execute
     * @param argc Runtime argument count
     * @param argv Runtime argument list ptr
     */
    explicit CUDASimulation(const ModelDescription& model, int argc = 0, const char** argv = nullptr);

 private:
    /**
     * Alt constructor used by CUDAEnsemble
     */
    explicit CUDASimulation(const std::shared_ptr<const ModelData> &model);
    /**
     * Private constructor, used to initialise submodels
     * Allocates CUDASubAgents, and handles mappings
     * @param submodel_desc The submodel description of the submodel (this should be from the already cloned model hierarchy)
     * @param master_model The CUDASimulation of the master model
     * @todo Move common components (init list and initOffsetsAndMap()) into a common/shared constructor
     */
    CUDASimulation(const std::shared_ptr<SubModelData>& submodel_desc, CUDASimulation *master_model);

 public:
    /**
     * Inverse operation of contructor
     */
    virtual ~CUDASimulation();
    /**
     * Steps the simulation once
     * @return False if an exit condition was triggered
     */
    bool step() override;
    /**
     * Execute the simulation until config.steps have been executed, or an exit condition trips
     */
    void simulate() override;
    /**
     * Replaces internal population data for the specified agent
     * @param population The agent type and data to replace agents with
     * @throw InvalidCudaAgent If the agent type is not recognised
     */
    void setPopulationData(AgentPopulation& population) override;
    /**
     * Returns the internal population data for the specified agent
     * @param population The agent type and data to fetch
     * @throw InvalidCudaAgent If the agent type is not recognised
     */
    void getPopulationData(AgentPopulation& population) override;
    /**
     * Returns the manager for the specified agent
     * @todo remove? this is mostly internal methods that modeller doesn't need access to
     */
    CUDAAgent& getCUDAAgent(const std::string &agent_name) const;
    AgentInterface &getAgent(const std::string &name) override;
    /**
     * Returns the manager for the specified agent
     * @todo remove? this is mostly internal methods that modeller doesn't need access to
     */
    CUDAMessage& getCUDAMessage(const std::string &message_name) const;
    /**
     * @return A mutable reference to the cuda model specific configuration struct
     * @see Simulation::applyConfig() Should be called afterwards to apply changes
     */
    Config &CUDAConfig();
    /**
     * Returns the number of times step() has been called since the simulation was last reset/init
     */
    unsigned int getStepCounter() override;
    /**
     * Manually resets the step counter
     */
    void resetStepCounter() override;
    /**
     * @return An immutable reference to the cuda model specific configuration struct
     */
    const Config &getCUDAConfig() const;
    /**
     * Configure which step data should be logged
     * @param stepConfig The step logging config for the CUDASimulation
     * @note This must be for the same model description hierarchy as the CUDASimulation
     */
    void setStepLog(const StepLoggingConfig &stepConfig);
    /**
     * Configure which exit data should be logged
     * @param exitConfig The logging config for the CUDASimulation
     * @note This must be for the same model description hierarchy as the CUDASimulation
     */
    void setExitLog(const LoggingConfig &exitConfig);
    /**
     * Returns a reference to the current exit log
     */
    const RunLog &getRunLog() const;
#ifdef VISUALISATION
    /**
     * Creates (on first call) and returns the visualisation configuration options for this model instance
     */
    ModelVis &getVisualisation();
#endif

    /**
     * Performs a cudaMemCopyToSymbol in the runtime library and also updates the symbols of any RTC functions (which exist separately within their own cuda module)
     * Will thrown an error if any of the calls fail.
     * @param symbol A device symbol
     * @param rtc_symbol_name The name of the symbol
     * @param src Source memory address
     * @param count Size in bytes to copy
     * @param offset Offset from start of symbol in bytes
     */
    void RTCSafeCudaMemcpyToSymbol(const void* symbol, const char* rtc_symbol_name, const void* src, size_t count, size_t offset = 0) const;

    /**
     * Performs a cudaMemCopy to a pointer in the runtime library and also updates the symbols of any RTC functions (which exist separately within their own cuda module)
     * Will thrown an error if any of the calls fail.
     * @param ptr a pointer to a symbol in device memory
     * @param rtc_symbol_name The name of the symbol
     * @param src Source memory address
     * @param count Size in bytes to copy
     * @param offset Offset from start of symbol in bytes
     */
    void RTCSafeCudaMemcpyToSymbolAddress(void* ptr, const char* rtc_symbol_name, const void* src, size_t count, size_t offset = 0) const;

    /**
     * Updates the environment property cache for all RTC agent functions
     * @param src Source memory address
     * @param count Length of buffer (Probably EnvironmentManager::MAX_BUFFER_SIZE)
     */
    void RTCUpdateEnvironmentVariables(const void* src, size_t count) const;

    /**
     * Get the duration of the last call to simulate() in milliseconds. 
     * With a resolution of around 0.5 microseconds (cudaEventElapsedtime)
     */
    float getSimulationElapsedTime() const;
    /**
     * Returns the unique instance id of this CUDASimulation instance
     * @note This value is used internally for environment property storage
     */
    using Simulation::getInstanceID;

 protected:
    /**
     * Returns the model to a clean state
     * This clears all agents and message lists, resets environment properties and reseeds random generation.
     * Also calls resetStepCounter();
     * @param submodelReset This should only be set to true when called automatically when a submodel reaches it's exit condition during execution. This performs a subset of the regular reset procedure.
     * @note If triggered on a submodel, agent states and environment properties mapped to a parent agent, and random generation are not affected.
     * @note If random was manually seeded, it will return to it's original state. If random was seeded from time, it will return to a new random state.
     */
    void reset(bool submodelReset) override;
    /**
     * Called by Simulation::applyConfig() to trigger any runner specific configs
     * @see CUDASimulation::CUDAConfig()
     */
    void applyConfig_derived() override;
    /**
     * Called by Simulation::checkArgs() to trigger any runner specific argument checking
     * This only handles arg 0 before returning
     * @return True if the argument was handled successfully
     * @see Simulation::checkArgs()
     */
    bool checkArgs_derived(int argc, const char** argv, int &i) override;
    /**
     * Called by Simulation::printHelp() to print help for runner specific runtime args
     * @see Simulation::printHelp()
     */
    void printHelp_derived() override;
    /**
     * Called at the start of Simulation::initialise(int, const char**) to reset runner specific config struct
     * @see Simulation::initialise(int, const char**)
     */
    void resetDerivedConfig() override;

 private:
    /**
     * Reinitalises random generation for this model and all submodels
     * @param seed New random seed (this updates stored seed in config)
     */
    void reseed(const unsigned int &seed);
    /**
     * Number of times step() has been called since sim was last reset/init
     */
    unsigned int step_count;
    /**
     * Duration of the last call to simulate() in milliseconds, with a resolution of around 0.5 microseconds (cudaEventElapsedtime)
     */
    float simulation_elapsed_time;
    /**
     * Update the step counter for host and device.
     */
    void incrementStepCounter();
    /**
     * Map of agent storage 
     */
    CUDAAgentMap agent_map;
    /**
     * Internal model config
     */
    Config config;
    /**
     * Step logging config
     */
    std::shared_ptr<const StepLoggingConfig> step_log_config;
    /**
     * Exit logging config
     */
    std::shared_ptr<const LoggingConfig> exit_log_config;
    /**
     * Collection of currently logged data
     */
    std::unique_ptr<RunLog> run_log;
    /**
     * Clear and reinitialise the current run_log
     */
    void resetLog();
    /**
     * Check if step_count is a divisible by step_log_config.frequency
     * If true, add the current simulation state to the step log
     */
    void processStepLog();
    /**
     * Replace the current exit log with the current simulation state
     */
    void processExitLog();
    /**
     * Map of message storage 
     */
    CUDAMessageMap message_map;
    /**
     * Map of submodel storage
     */
    CUDASubModelMap submodel_map;
    /**
     * Streams created within this cuda context for executing functions within layers in parallel
     */
    std::vector<cudaStream_t> streams;

    /**
     * Struct containing references to the various singletons which may include CUDA code, and therefore can only be initialsed after the deferred arg parsing is completed.
     */
    struct Singletons {
      /**
       * Curve instance used for variable mapping
       * @todo Is this necessary? CUDAAgent/CUDAMessage have their own copy
       */
      Curve &curve;
      /**
       * Resizes device random array during step()
       */
      RandomManager rng;
      /**
       * Held here for tracking when to release cuda memory
       */
      CUDAScatter scatter;
      /**
       * Held here for tracking when to release cuda memory
       */
      EnvironmentManager &environment;
#ifndef NO_SEATBELTS
      /**
       * Provides buffers for device error checking
       */
      DeviceExceptionManager exception;
#endif
      Singletons(Curve &curve, EnvironmentManager &environment) : curve(curve), environment(environment) { }
    } * singletons;
    /**
     * Common method for adding this Model's data to env manager
     */
    void initEnvironmentMgr();
    /**
     * Flag indicating that the model has been initialsed
     */
    bool singletonsInitialised;

    /**
     * Flag indicating that RTC functions have been compiled
     */
    bool rtcInitialised;
    /**
     * Set to the ID of the device on which the simulation was initialised
     * Cannot change device after this point
     */
    int deviceInitialised = -1;

    /**
     * Initialise the instances singletons.
     */
    void initialiseSingletons();
    /**
     * Initialise the rtc by building any RTC functions.
     * This must be done at the start of step to ensure that any device selection has taken place and to preserve the context between runtime and RTC.
     */
    void initialiseRTC();
    /**
     * One instance of host api is used for entire model
     */
    std::unique_ptr<FLAMEGPU_HOST_API> host_api;
    /**
     * Adds any agents stored in agentData to the device
     * Clears agent storage in agentData
     * @param streamId Stream index to perform scatter on
     * @note called at the end of step() and after all init/hostLayer functions and exit conditions have finished
     */
    void processHostAgentCreation(const unsigned int &streamId);

 public:
    typedef std::vector<NewAgentStorage> AgentDataBuffer;
    typedef std::unordered_map<std::string, AgentDataBuffer> AgentDataBufferStateMap;
    typedef std::unordered_map<std::string, VarOffsetStruct> AgentOffsetMap;
    typedef std::unordered_map<std::string, AgentDataBufferStateMap> AgentDataMap;

 private:
    /**
     * Provides the offset data for variable storage
     * Used by host agent creation
     */
    AgentOffsetMap agentOffsets;
    /**
     * Storage used by host agent creation before copying data to device at end of each step()
     */
    AgentDataMap agentData;
    void initOffsetsAndMap();
#ifdef VISUALISATION
    /**
     * Empty if getVisualisation() hasn't been called
     */
    std::unique_ptr<ModelVis> visualisation;
#endif
    /**
     * This tracks the current number of alive CUDASimulation instances
     * When the last is destructed, cudaDeviceReset is triggered();
     */
    static std::atomic<int> active_instances;
    /**
     * Active instances, but linked to the device each instance has been initialised on
     */
    static std::map<int, std::atomic<int>> active_device_instances;
    /**
     * These exist to prevent us doing device reset in the short period between checking last sim of device, and reset
     */
    static std::map<int, std::shared_timed_mutex> active_device_mutex;
    /**
     * This controls access to active_device_instances, active_device_mutex
     */
    static std::shared_timed_mutex active_device_maps_mutex;

 public:
    /**
     * If changed to false, will not auto cudaDeviceReset when final CUDASimulation instance is destructed
     */
    static bool AUTO_CUDA_DEVICE_RESET;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDASIMULATION_H_
