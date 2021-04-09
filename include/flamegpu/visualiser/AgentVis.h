#ifndef INCLUDE_FLAMEGPU_VISUALISER_AGENTVIS_H_
#define INCLUDE_FLAMEGPU_VISUALISER_AGENTVIS_H_
#ifdef VISUALISATION

// #include <glm/mat4x4.hpp>
#include <unordered_map>
#include <memory>
#include <string>
#include <map>

#include "flamegpu/visualiser/AgentStateVis.h"
#include "config/AgentStateConfig.h"
#include "config/Stock.h"
#include "config/TexBufferConfig.h"

struct Palette;
struct AgentData;
class CUDAAgent;
class FLAMEGPU_Visualisation;
class ColorFunction;
struct Color;
class AutoPalette;

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
     * @param autopalette Automatic source of colors for individual agent states
     * @note Agent states only receive colors from the autopalette when AgentVis::State() is called for each state
     * @note By default, all states share the same color from the autopalette
     */
    explicit AgentVis(CUDAAgent &agent, const std::shared_ptr<AutoPalette> &autopalette = nullptr);
    /**
     * Returns the configuration handler for the named state
     * On first use for each state this will assign the state a color from the AutoPalette if available
     * Clear the autopalette first if you wish for it to use the default color
     */
    AgentStateVis &State(const std::string &state_name);

    /**
     * Set the name of the variable representing the agents x/y/z location coordinates
     * @param var_name Name of the agent variable
     * @note unnecessary if the variables are named "x", "y", "z" respectively
     */
    void setXVariable(const std::string &var_name);
    void setYVariable(const std::string &var_name);
    void setZVariable(const std::string &var_name);
    /**
     * Set the name of the variable representing the agents x/y/z direction vector components
     * Single axis rotation only requires x/z components
     * Double axis rotation requires all 3 components
     * Triple axis rotation requires all 3 components and additionally all 3 Up components
     * @param var_name Name of the agent variable
     */
    void setForwardXVariable(const std::string& var_name);
    void setForwardYVariable(const std::string& var_name);
    void setForwardZVariable(const std::string& var_name);
    /**
     * Set the name of the variable representing the agents x/y/z UP vector
     * This should be 90 degrees perpendicular to the direction vector
     * @param var_name Name of the agent variable
     */
    void setUpXVariable(const std::string& var_name);
    void setUpYVariable(const std::string& var_name);
    void setUpZVariable(const std::string& var_name);
    /**
     * Set the name of the variable representing the agents yaw rotation angle (radians)
     * This is an alternate to providing a direction vector, setting this will erase direction x/z if bound
     *
     * @param var_name Name of the agent variable
     * @note setRollVariable() can be used in place of the UP vector if preferred
     */
    void setYawVariable(const std::string& var_name);
    /**
     * Set the name of the variable representing the agents pitch rotation angle (radians)
     * This is an alternate to providing a direction vector, setting this will erase direction y if bound
     *
     * @param var_name Name of the agent variable
     */
    void setPitchVariable(const std::string& var_name);
    /**
     * Set the name of the variable representing the agents yaw rotation angle (radians)
     * This is an alternate to providing an UP vector, setting this will erase up x/y/z if bound
     *
     * @param var_name Name of the agent variable
     * @note setRollVariable() can be used in place of the UP vector if preferred
     */
    void setRollVariable(const std::string& var_name);
    /**
     * Clears the agent's x/y/z location variable bindings
     * @see setXVariable(conCst std::string &)
     * @see setYVariable(conCst std::string &)
     * @see setZVariable(conCst std::string &)
     */
    void clearXVariable();
    void clearYVariable();
    void clearZVariable();
    /**
     * Clears the agent's x/y/z direction variable bindings
     * @see setForwardXVariable(const std::string &)
     * @see setForwardYVariable(const std::string &)
     * @see setForwardZVariable(const std::string &)
     */
    void clearForwardXVariable();
    void clearForwardYVariable();
    void clearForwardZVariable();
    /**
     * Clears the agent's x/y/z UP variable bindings
     * @see setUpXVariable(const std::string &)
     * @see setUpYVariable(const std::string &)
     * @see setUpZVariable(const std::string &)
     */
    void clearUpXVariable();
    void clearUpYVariable();
    void clearUpZVariable();
    /**
     * Clears the agent's yaw angle variable bindings
     * @see setYawVariable(const std::string &)
     */
    void clearYawVariable();
    /**
     * Clears the agent's pitch angle variable bindings
     * @see setPitchVariable(const std::string &)
     */
    void clearPitchVariable();
    /**
     * Clears the agent's roll angle variable bindings
     * @see setRollVariable(const std::string &)
     */
    void clearRollVariable();
    /**
     * Returns the variable used for the agent's x/y/z location coordinates
     */
    std::string getXVariable() const;
    std::string getYVariable() const;
    std::string getZVariable() const;
    /**
     * Returns the variable used for the agent's x/y/z direction vector components
     */
    std::string getForwardXVariable() const;
    std::string getForwardYVariable() const;
    std::string getForwardZVariable() const;
    /**
     * Returns the variable used for the agent's x/y/z direction vector components
     */
    std::string getUpXVariable() const;
    std::string getUpYVariable() const;
    std::string getUpZVariable() const;
    /**
     * Returns the variable used for the agent's yaw angle
     */
    std::string getYawVariable() const;
    /**
     * Returns the variable used for the agent's pitch angle
     */
    std::string getPitchVariable() const;
    /**
     * Returns the variable used for the agent's roll angle
     */
    std::string getRollVariable() const;

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
     * Set the auto-palette used to assign agent-state's colors
     * @note The color is assigned the first time State() is called, otherwise agents use the default color
     */
    void setAutoPalette(const Palette& ap);
    /**
     * Set a custom color function
     * @note Disables the auto-palette
     */
    void setColor(const ColorFunction &cf);
    /**
     * Disable custom color and/or auto-palette, e.g. if you're using a textured model
     */
    void clearColor();

 private:
    /**
     * Pass vis configs for each agent state to visualiser
     */
    void initBindings(std::unique_ptr<FLAMEGPU_Visualisation> &vis);
    /**
     * This requests that the visualisation resizes buffers
     * @param vis The affected visualisation
     * @param force When true is passed, vis will delay closing the splash screen until this update has been processed
     * Used when agent population has grown
     * @return Returns true if a non-0 buffer was requested
     */
    bool requestBufferResizes(std::unique_ptr<FLAMEGPU_Visualisation> &vis, bool force);
    /**
     * This passes the correct device pointers to the visualisation and forces it to update the data used for rendering
     * @param vis The affected visualisation
     * @note This should only be called when visualisation muted is held
     */
    void updateBuffers(std::unique_ptr<FLAMEGPU_Visualisation> &vis);
    /**
     * Link to the currently active auto_palette
     */
    std::weak_ptr<AutoPalette> auto_palette;
    /**
     * If setAutoPalette() is called, the created AutoPalette is stored here
     */
    std::shared_ptr<AutoPalette> owned_auto_palette;
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
     * Holds information on core agent-wide texture buffers
     * e.g. location/direction
     */
    std::map<TexBufferConfig::Function, TexBufferConfig> core_tex_buffers;
};

#endif  // VISUALISATION
#endif  // INCLUDE_FLAMEGPU_VISUALISER_AGENTVIS_H_
