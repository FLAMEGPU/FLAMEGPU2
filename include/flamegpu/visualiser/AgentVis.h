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
    /**
     * @param agent The CUDAAgent this class is configuring the visualisation for
     */
    explicit AgentVis(CUDAAgent &agent);
    /**
     * Returns the configuration handler for the named state
     */
    AgentStateVis &State(const std::string &state_name);

    /**
     * Set the name of the variable representing the agents x location
     * @param var_name Name of the agent variable
     * @note unnecessary if the variable is "x"
     */
    void setXVariable(const std::string &var_name);
    /**
     * Set the name of the variable representing the agents y location
     * @param var_name Name of the agent variable
     * @note unnecessary if the variable is "y"
     */
    void setYVariable(const std::string &var_name);
    /**
     * Set the name of the variable representing the agents z location
     * @param var_name Name of the agent variable
     * @note unnecessary if the variable is "z", or the model is 2D
     */
    void setZVariable(const std::string &var_name);
    /**
     * Clears the agent's z variable binding
     * @see setZVariable(const std::string &)
     */
    void clearZVariables();
    /**
     * Returns the variable used for the agent's location's x coordinate
     */
    std::string getXVariable() const;
    /**
     * Returns the variable used for the agent's location's y coordinate
     */
    std::string getYVariable() const;
    /**
     * Returns the variable used for the agent's location's z coordinate
     */
    std::string getZVariable() const;

    /**
     * Use a model from file
     * @param modelPath The path to the model's file (must be .obj)
     * @param texturePath Optional path to the texture used by the model
     */
    void setModel(const std::string &modelPath, const std::string &texturePath = "");
    /**
     * Use a stock model
     * @param model Model from the libraries internal resources
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
     * This requests that the visualisation resizes buffers
     * @param vis The affected visualisation
     * Used when agent population has grown
     */
    void requestBufferResizes(std::unique_ptr<FLAMEGPU_Visualisation> &vis);
    /**
     * This passes the correct device pointers to the visualisation and forces it to update the data used for rendering
     * @param vis The affected visualisation
     * @note This should only be called when visualisation muted is held
     */
    void updateBuffers(std::unique_ptr<FLAMEGPU_Visualisation> &vis);
    /**
     * This is the default configuration options for states of this agent
     * These values will be used for any state configuration options which have not been set independently
     */
    AgentStateConfig defaultConfig;
    /**
     * Map of configurations for individual agent states
     */
    std::unordered_map<std::string, AgentStateVis> states;
    /**
     * CUDAAgent being rendered
     */
    CUDAAgent &agent;
    /**
     * Agent description hierarchy being rendered
     */
    const AgentData &agentData;
    /**
     * Names of the agent variables holding the agent's location
     * @see setXVariable(const std::string &)
     * @see setYVariable(const std::string &)
     * @see setZVariable(const std::string &)
     */
    std::string x_var, y_var, z_var;
};

#endif  // VISUALISATION
#endif  // INCLUDE_FLAMEGPU_VISUALISER_AGENTVIS_H_
