#ifndef INCLUDE_FLAMEGPU_VISUALISER_AGENTSTATEVIS_H_
#define INCLUDE_FLAMEGPU_VISUALISER_AGENTSTATEVIS_H_
#ifdef VISUALISATION

#include <string>

#include "config/AgentStateConfig.h"
#include "config/Stock.h"

namespace flamegpu {
namespace visualiser {

struct Color;
class ColorFunction;
class AgentVis;

/**
 * This provides an interface for managing the render options for all agents within a specific state
 * Options default to values set within their parent
 * Even if the default is changed after the agent state is constructed
 * TODO: Block everything non-const from being called whilst VIS is active
 */
class AgentStateVis {
    /**
     * Parent has friend access
     */
    friend class AgentVis;
 public:
    /**
     * Creates a new AgentStateVis to configure visualisation options for a particular agent-state
     * @param parent Visualisation options for the agent
     * @param state_name State of the agent for which this options should represent
     */
    AgentStateVis(const AgentVis &parent, const std::string &state_name);

    /**
     * Use a model from file
     * @param modelPath File path of model
     * @param texturePath Optional path to a texture for the model
     * @note Model must be .obj format
     */
    void setModel(const std::string &modelPath, const std::string &texturePath = "");
    /**
     * Use a stock model
     * @param model Model from internal resources
     * @see Stock::Models::Model
     */
    void setModel(const Stock::Models::Model &model);
    /**
     * Scale each dimension of the model to the corresponding world scales
     * @param xLen World scale of the model's on the x axis
     * @param yLen World scale of the model's on the y axis
     * @param zLen World scale of the model's on the z axis
     * @note Y is considered the vertical axis
     */
    void setModelScale(float xLen, float yLen, float zLen);
    /**
     * Uniformly scale model so that max dimension equals this
     * @param maxLen World scale of the model's relative to the axis which it is
     * largest
     */
    void setModelScale(float maxLen);
    /**
     * Set a custom colour function
     */
    void setColor(const ColorFunction& cf);
    /**
     * Disable custom color, e.g. if you're using a textured model
     */
    void clearColor();

 private:
    /**
     * The parent visualisation options, these hold the default configuration
     */
    const AgentVis &parent;
    /**
     * Name of the state from agent description hierarchy
     */
    const std::string state_name;
    /**
     * Holds the config used to render agents in this state
     */
    AgentStateConfig config;
    /**
     * Holds a boolean for each option (or group of options), to decide whether they should be updated if the default is changed
     */
    AgentStateConfigFlags configFlags;
};

}  // namespace visualiser
}  // namespace flamegpu

#endif  // VISUALISATION
#endif  // INCLUDE_FLAMEGPU_VISUALISER_AGENTSTATEVIS_H_
