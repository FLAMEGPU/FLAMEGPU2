#ifndef INCLUDE_FLAMEGPU_VISUALISER_AGENTVIS_H_
#define INCLUDE_FLAMEGPU_VISUALISER_AGENTVIS_H_
#ifdef VISUALISATION

// #include <glm/mat4x4.hpp>
#include <unordered_map>
#include <memory>
#include <string>

#include "flamegpu/visualiser/AgentStateVis.h"
#include "config/AgentStateConfig.h"
#include "config/Stock.h"

struct AgentData;
class CUDAAgent;
class FLAMEGPU_Visualisation;

/**
 * This provides an interface for managing the render options for all agents of a specific type
 * State() can be called to specialise options for agents within a specific state
 * TODO: Block everything non-const from being called whilst VIS is active
 */
class AgentVis {
    friend class ModelVis;
    friend class AgentStateVis;
 public:
    explicit AgentVis(CUDAAgent &agent);
    /**
     * Returns the configuration handler for the named state
     */
    AgentStateVis &State(const std::string &state_name);

    void setXVariable(const std::string &var_name);
    void setYVariable(const std::string &var_name);
    void setZVariable(const std::string &var_name);
    void clearZVariables();

    std::string getXVariable() const;
    std::string getYVariable() const;
    std::string getZVariable() const;

    /**
     * Use a model from file
     */
    void setModel(const std::string &modelPath, const std::string &texturePath = "");
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
     * @param maxLen World scale of the model's relative to the axis which it is largest
     */
    void setModelScale(float maxLen);

    /**
     * Some shader modes wont want this though, e.g. tex
     */
    // void setColor(const float &r, const float &g, const float &b) { setColor(glm::vec4(r, g, b, 1.0f)); }
    // void setColor(const float &r, const float &g, const float &b, const float &a) { setColor(glm::vec4(r, g, b, a)); }
    // void setColor(const glm::vec3 &rgb) { setColor(glm::vec4(rgb, 1.0f)); }
    // void setColor(const glm::vec4 &rgba);

 private:
    /**
     * Pass vis configs for each agent state to visualiser
     */
    void initBindings(std::unique_ptr<FLAMEGPU_Visualisation> &vis);
    /**
     * Update agent count and data within visualiation buffers for each agent state
     */
    void requestBufferResizes(std::unique_ptr<FLAMEGPU_Visualisation> &vis);
    void updateBuffers(std::unique_ptr<FLAMEGPU_Visualisation> &vis);
    AgentStateConfig defaultConfig;
    std::unordered_map<std::string, AgentStateVis> states;
    CUDAAgent &agent;
    const AgentData &agentData;

    std::string x_var, y_var, z_var;
};

#endif  // VISUALISATION
#endif  // INCLUDE_FLAMEGPU_VISUALISER_AGENTVIS_H_
