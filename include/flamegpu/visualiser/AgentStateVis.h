#ifndef INCLUDE_FLAMEGPU_VISUALISER_AGENTSTATEVIS_H_
#define INCLUDE_FLAMEGPU_VISUALISER_AGENTSTATEVIS_H_
#ifdef VISUALISATION

#include <string>

#include "config/AgentStateConfig.h"
#include "config/Stock.h"

class AgentVis;

/**
 * This provides an interface for managing the render options for all agents within a specific state
 * Options default to values set within their parent
 * TODO: Block everything non-const from being called whilst VIS is active
 */
class AgentStateVis {
    friend class AgentVis;
 public:
    /**
     *
     * @param _parent Visualisation options for the agent
     * @param state_name State of the agent for which this options should represent
     */
    AgentStateVis(const AgentVis &_parent, const std::string &state_name);

    /**
     * Use a model from file
     */
    void setModel(const std::string &modelPath);
    /**
     * Use a stock model
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

 private:
    const AgentVis &parent;
    const std::string state_name;

    AgentStateConfig config;
    AgentStateConfigFlags configFlags;
};

#endif  // VISUALISATION
#endif  // INCLUDE_FLAMEGPU_VISUALISER_AGENTSTATEVIS_H_
