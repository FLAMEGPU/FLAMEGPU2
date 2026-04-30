#ifndef INCLUDE_FLAMEGPU_STOCKAGENT_SUBMODELS_SINGLEAGENTDISCRETEMOVEMENT_H_
#define INCLUDE_FLAMEGPU_STOCKAGENT_SUBMODELS_SINGLEAGENTDISCRETEMOVEMENT_H_

#include <map>
#include <string>
#include <memory>

#include "AbstractSubmodels.h"


namespace flamegpu {
namespace stockAgent {
namespace submodels {

/**
 * Submodel for handling agent movement logic on a discrete 2D grid.
 * This submodel now supports two distinct agent types:
 * 1. A moving agent (e.g., Bee, Person)
 * 2. An environment agent (e.g., Flower, Grid Cell)
 */
class SingleAgentDiscreteMovement : public AbstractSubmodel {
 public:
    /**
     * Empty constructor. Object must be initialized via addSingleAgentDiscreteMovementSubmodel().
     */
    SingleAgentDiscreteMovement() = default;

    /**
     * Defines the movement submodel and adds it to the provided parent model.
     * @param model The parent ModelDescription
     * @param ENV_SIZE_X Width of the environment
     * @param ENV_SIZE_Y Height of the environment
     * @return The created SubModelDescription
     */
    flamegpu::SubModelDescription addSingleAgentDiscreteMovementSubmodel(flamegpu::ModelDescription &model, int ENV_SIZE_X, int ENV_SIZE_Y);

    /**
     * Binds a parent agent to the submodel's internal moving agent.
     */
    void setMovingAgent(const std::string& parent_name,
                  const std::map<std::string, std::string>& var_map = {},
                  const std::map<std::string, std::string>& state_map = {},
                  bool auto_map = false);

    /**
     * Binds a parent agent to the submodel's internal environment agent (grid cell).
     */
    void setEnvironmentAgent(const std::string& parent_name,
                  const std::map<std::string, std::string>& var_map = {},
                  const std::map<std::string, std::string>& state_map = {},
                  bool auto_map = false);

    void validate() override;

 private:
    // Internal pointer allows SingleAgentDiscreteMovement to be default-constructible
    std::unique_ptr<flamegpu::SubModelDescription> smm;
    bool is_initialized = false;

    // Internal setup methods
    void setMessages(flamegpu::AgentDescription &movingAgent, flamegpu::AgentDescription &envAgent, int ENV_SIZE_X, int ENV_SIZE_Y);
    void defineMessageSubmodule(flamegpu::ModelDescription &smm, int ENV_SIZE_X, int ENV_SIZE_Y);
    void defineLayer(flamegpu::ModelDescription &smm);
};

}  // namespace submodels
}  // namespace stockAgent
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_STOCKAGENT_SUBMODELS_SINGLEAGENTDISCRETEMOVEMENT_H_
