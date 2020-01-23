#ifndef INCLUDE_FLAMEGPU_GPU_CUDAAGENTMODEL_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAAGENTMODEL_H_

#include <map>
#include <memory>
#include <string>

#include "flamegpu/sim/Simulation.h"

#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/runtime/utility/RandomManager.cuh"
#include "CUDAScatter.h"

/**
 * CUDA runner for Simulation interface
 * Executes a FGPU2 model using GPU
 */
class CUDAAgentModel : public Simulation {
    /**
     * Map of a number of CUDA agents by name.
     * The CUDA agents are responsible for allocating and managing all the device memory
     */
    typedef std::map<std::string, std::unique_ptr<CUDAAgent>> CUDAAgentMap;
    /**
     * Map of a number of CUDA messages by name.
     * The CUDA messages are responsible for allocating and managing all the device memory
     */
    typedef std::map<std::string, std::unique_ptr<CUDAMessage>> CUDAMessageMap;

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
     * @param model The model description to initialise the runner to execute
     */
    explicit CUDAAgentModel(const ModelDescription& model);
    /**
     * Inverse operation of contructor
     */
    virtual ~CUDAAgentModel();
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

 protected:
     /**
      * Called by Simulation::applyConfig() to trigger any runner specific configs
      * @see CUDAAgentModel::CUDAConfig()
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
     * Number of times step() has been called since sim was last reset/init
     */
    unsigned int step_count;
    /**
     * Map of agent storage 
     */
    CUDAAgentMap agent_map;
    /**
     * Curve instance used for variable mapping
     * @todo Is this necessary? CUDAAgent/CUDAMessage have their own copy
     */
    Curve &curve;
    /**
     * Internal model config
     */
    Config config;
    /**
     * Map of message storage 
     */
    CUDAMessageMap message_map;
    /**
    * Resizes device random array during step()
    */
    RandomManager &rng;
    /**
     * Held here for tracking when to release cuda memory
     */
    CUDAScatter &scatter;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAAGENTMODEL_H_
