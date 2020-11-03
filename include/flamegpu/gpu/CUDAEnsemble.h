#ifndef INCLUDE_FLAMEGPU_GPU_CUDAENSEMBLE_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAENSEMBLE_H_
#include <atomic>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>


#include "flamegpu/exception/FGPUDeviceException.h"
#include "flamegpu/sim/Simulation.h"

#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/CUDAScatter.h"
#include "flamegpu/runtime/utility/RandomManager.cuh"
#include "flamegpu/runtime/flamegpu_host_new_agent_api.h"
#include "flamegpu/visualiser/ModelVis.h"


/**
 * Experimental class for managing multiple simultaneously executing copies of a simulation
 * Based on CUDASimulation, however as it holds multiple populations it has various major differences to the user interface and model execution
 */
class CUDAEnsemble {
    /**
     * Requires internal access to scan/scatter singletons
     */
    friend class HostAgentInstance;
    /**
     * Map of a number of CUDA agents by name.
     * The CUDA agents are responsible for allocating and managing all the device memory
     * Each vector holds a CUDAAgent/CUDAMessage per active instance executed by the ensemble.
     * When an active instance completes, it's CUDAAgent/CUDAMessage structures are reused
     * This saves additional memory allocations etc
     */
    typedef std::unordered_map<std::string, std::vector<std::shared_ptr<CUDAAgent>>> CUDAEnsembleAgentMap;
    typedef std::unordered_map<std::string, std::vector<std::reference_wrapper<AgentPopulation>>> CUDAEnsembleAgentPop;
    /**
     * Map of a number of CUDA messages by name.
     * The CUDA messages are responsible for allocating and managing all the device memory
     */
    typedef std::unordered_map<std::string, std::vector<std::shared_ptr<CUDAMessage>>> CUDAEnsembleMessageMap;
    /**
     * Map of a number of CUDA sub models by name.
     * The CUDA submodels are responsible for allocating and managing all the device memory of non mapped agent vars
     * Ordered is used, so that random seed mutation always occurs same order.
     */
    typedef std::map<std::string, std::unique_ptr<CUDAEnsemble>> CUDASubModelMap;

 public:
    typedef std::vector<NewAgentStorage> AgentDataBuffer;
    typedef std::unordered_map<std::string, AgentDataBuffer> AgentDataBufferStateMap;
    typedef std::unordered_map<std::string, VarOffsetStruct> AgentOffsetMap;
    typedef std::unordered_map<std::string, AgentDataBufferStateMap> AgentDataMap;
     /**
      * CUDA specific config
      */
    struct CConfig {
        /**
         * GPU to execute model on
         * Defaults to device 0, this is most performant device as detected by CUDA
         */
        int device_id = 0;
    };
    /**
     * Ensemble specific config
     */
    struct EConfig {
        /**
         * Write more status information to stdout during model execution
         */
        bool verbose = false;
        /**
         * Print timing info after ensemble execution has completed
         */
        bool timing = false;
        /**
         * Steps to run each run for
         * If 0 is provided, the model must have atleast 1 exit condition
         */
        unsigned int steps = 100;
        /**
         * Max concurrent simulations
         */
        unsigned int max_concurrent_runs = 2;
        /**
         * Total runs
         */
        unsigned int total_runs = 100;
        /**
         * Directory to store output data (each run will get it's own output directory within this)
         * @note Do we also want to allow format selection?
         */
        std::string out_directory;
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
    explicit CUDAEnsemble(const ModelDescription& model, int argc = 0, const char** argv = nullptr);

 private:
    /**
     * Private constructor, used to initialise submodels
     * Allocates CUDASubAgents, and handles mappings
     * @param submodel_desc The submodel description of the submodel (this should be from the already cloned model hierarchy)
     * @param master_model The CUDAEnsemble of the master model
     * @todo Move common components (init list and initOffsetsAndMap()) into a common/shared constructor
     */
    CUDAEnsemble(const std::shared_ptr<SubModelData>& submodel_desc, CUDAEnsemble *master_model);

 public:
    /**
     * Inverse operation of contructor
     */
    virtual ~CUDAEnsemble();
    /**
     * Execute the simulation until config.steps have been executed, or an exit condition trips
     */
    void simulate();
    /**
     * Replaces internal population data for the specified agent
     * @param population The agent type and data to replace agents with
     * @return The index of population
     * @throw InvalidCudaAgent If the agent type is not recognised
     */
    unsigned int addPopulationData(AgentPopulation& population);
    void setPopulationData(const unsigned int &index,AgentPopulation& population);
    /**
     * Returns the internal population data for the specified agent
     * @param index Index of the population to return
     * @param population The agent type and data to fetch
     * @throw InvalidCudaAgent If the agent type is not recognised
     */
    void getPopulationData(const unsigned int &index, AgentPopulation& population);
    /**
     * Returns the number of populations within the ensemble
     */
    unsigned int getPopulationCount() const;
    /**
     * @return A mutable reference to the cuda model specific configuration struct
     * @see Simulation::applyConfig() Should be called afterwards to apply changes
     */
    CConfig &CUDAConfig();
    EConfig &EnsembleConfig();
    /**
     * @return An immutable reference to the cuda model specific configuration struct
     */
    const CConfig &getCUDAConfig() const;
    const EConfig &getEnsembleConfig() const;
//#ifdef VISUALISATION
//    /**
//     * Creates (on first call) and returns the visualisation configuration options for this model instance
//     */
//    ModelVis &getVisualisation();
//#endif

    /**
     * Get the duration of the last call to simulate() in milliseconds. 
     * With a resolution of around 0.5 microseconds (cudaEventElapsedtime)
     * @note Individual sim times makes no sense here, so this time should refer to execution of ensemble. E.g. you request 100, how long does running of 100 take.
     */
    float getElapsedTime() const;

 private:
    /**
     * Steps the simulation once
     * @return False if an exit condition was triggered
     */
    bool step();
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
     * Returns the unique instance id of this CUDAEnsemble instance
     * @note This value is used internally for environment property storage
     */
    //using Simulation::getInstanceID;
    struct CUDAEnsembleInstance : public SimInterface {
        CUDAEnsembleInstance(std::shared_ptr<const ModelData> modelddata,
        std::map<std::string, std::unique_ptr<CUDAEnsemble>> &submodels,
        RandomManager &rng,
        const AgentOffsetMap &agentOffsets, const unsigned int &instance_id);
        ~CUDAEnsembleInstance() override {};
        const ModelData& getModelDescription() const override {
            return *model;
        }
        AgentInterface& getAgent(const std::string& name) override {
            const auto it = agent_map.find(name);

            if (it == agent_map.end()) {
                THROW InvalidCudaAgent("CUDA agent ('%s') not found, in CUDAEnsembleInstance::getAgent().",
                    name.c_str());
            }
            return *(it->second.get());
        }
        CUDAAgent& getCUDAAgent(const std::string& agent_name) const {
            const auto it = agent_map.find(agent_name);

            if (it == agent_map.end()) {
                THROW InvalidCudaAgent("CUDA agent ('%s') not found, in CUDASimulation::getCUDAAgent().",
                    agent_name.c_str());
            }

            return *(it->second);
        }
        CUDAMessage& getCUDAMessage(const std::string& message_name) const {
            const auto it = message_map.find(message_name);

            if (it == message_map.end()) {
                THROW InvalidCudaMessage("CUDA message ('%s') not found, in CUDASimulation::getCUDAMessage().",
                    message_name.c_str());
            }

            return *(it->second);
        }
        unsigned int getInstanceID() const override {
            return instance_id;
        }
        unsigned int getStepCounter() override {
            return 0;// TODO: how will this work?
        }
        const std::shared_ptr<const ModelData> model;
        std::unordered_map<std::string, std::unique_ptr<CUDAAgent>> agent_map;
        std::unordered_map<std::string, std::unique_ptr<CUDAMessage>> message_map;
        std::map<std::string, std::unique_ptr<CUDAEnsemble>> &submodel_map;
        typedef std::vector<NewAgentStorage> AgentDataBuffer;//Hate duplicating these typedefs
        typedef std::unordered_map<std::string, AgentDataBuffer> AgentDataBufferStateMap;
        typedef std::unordered_map<std::string, AgentDataBufferStateMap> AgentDataMap;
        AgentDataMap agentData;
        const unsigned int instance_id;
        std::unique_ptr<FLAMEGPU_HOST_API> host_api = nullptr;
    };

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
    float ensemble_elapsed_time;
    /**
     * Update the step counter for host and device.
     */
    void incrementStepCounter();
    /**
     * Internal model config
     */
    CConfig cuda_config;
    EConfig ensemble_config;
    const std::shared_ptr<const ModelData> model;    
    /**
     * This is only used when model is a submodel, otherwise it is empty
     * If it is set, it causes simulate() to additionally reset/cull unmapped populations
     */
    const std::shared_ptr<const SubModelData> submodel;
    /**
     * Only used by submodels, only required to fetch the name of master model when initialising environment (as this occurs after constructor)
     */
    CUDAEnsemble const * mastermodel;
    /**
     * Map of submodel storage
     */
    CUDASubModelMap submodel_map;
    /**
     * We hold agent population data here before/after use
     */
    CUDAEnsembleAgentPop agent_data;
    /**
     * Streams created within this cuda context for executing functions within layers in parallel
     */
    std::vector<cudaStream_t> streams;
    std::vector<CUDAEnsembleInstance> instance_vector;
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
     * Initialise the instances singletons.
     */
    void initialiseSingletons();
    /**
     * Initialise the rtc by building any RTC functions.
     * This must be done at the start of step to ensure that any device selection has taken place and to preserve the context between runtime and RTC.
     */
    void initialiseRTC();
    /**
     * This must be allocated after initialise singletons
     * This must be reset after cudaDeviceReset()
     */
    jitify::JitCache *rtc_kernel_cache;
    /**
     * Adds any agents stored in agentData to the device
     * Clears agent storage in agentData
     * @param streamId Stream index to perform scatter on
     * @note called at the end of step() and after all init/hostLayer functions and exit conditions have finished
     */
    void processHostAgentCreation(const unsigned int &streamId);

 public:

    void initialise(int argc, const char** argv);
    void applyConfig();
 private:
    // 1 Of these structs exists per stream
    struct InstanceOffsetData {
        InstanceOffsetData(const unsigned int &activeInstances);
        ~InstanceOffsetData();
        unsigned int * const d_instance_offsets;
        Curve::NamespaceHash * const d_instance_id_hashes;
    private:
        static void * const cmalloc(const size_t &len);
    };
    std::vector<InstanceOffsetData> instance_offset_vector;
    // Setup all of the CUDAAgent, CUDAMessage objects for the active instances
    void initCUDAInstances(const unsigned int &active_instances);
    // Purge all of the CUDAAgent, CUDAMessage instances
    void releaseCUDAInstances();
    /**
     * Provides the offset data for variable storage
     * Used by host agent creation
     */
    AgentOffsetMap agentOffsets;
    void initOffsetsAndMap();
    /**
     * This tracks the current number of alive CUDAEnsemble instances
     * When the last is destructed, cudaDeviceReset is triggered();
     */
    static std::atomic<int> active_instances;
    void printHelp(const char *executable);
    int checkArgs(int argc, const char** argv);

 public:
    /**
     * If changed to false, will not auto cudaDeviceReset when final CUDAEnsemble instance is destructed
     */
    static bool AUTO_CUDA_DEVICE_RESET;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAENSEMBLE_H_
