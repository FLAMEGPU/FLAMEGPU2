#ifndef INCLUDE_FLAMEGPU_VISUALISER_AGENTVIS_H_
#define INCLUDE_FLAMEGPU_VISUALISER_AGENTVIS_H_
#ifdef FLAMEGPU_VISUALISATION

// #include <glm/mat4x4.hpp>
#include <unordered_map>
#include <memory>
#include <string>
#include <map>

// @todo - All vis headers should live in the vis repo.
#include "flamegpu/visualiser/AgentStateVis.h"
#include "flamegpu/visualiser/config/AgentStateConfig.h"
#include "flamegpu/visualiser/config/Stock.h"
#include "flamegpu/visualiser/config/TexBufferConfig.h"

namespace flamegpu {
struct AgentData;
class CUDAAgent;

namespace visualiser {
struct Palette;
class FLAMEGPU_Visualisation;
class ColorFunction;
struct Color;
class AutoPalette;

struct AgentVisData {
    /**
     * @param agent The CUDAAgent this class is configuring the visualisation for
     * @param autopalette Automatic source of colors for individual agent states
     * @note Agent states only receive colors from the autopalette when AgentVis::State() is called for each state
     * @note By default, all states share the same color from the autopalette
     */
    explicit AgentVisData(CUDAAgent& agent, const std::shared_ptr<AutoPalette>& autopalette = nullptr);
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
    std::unordered_map<std::string, std::shared_ptr<AgentStateVisData>> states;
    /**
     * CUDAAgent being rendered
     */
    CUDAAgent& agent;
    /**
     * Agent description hierarchy being rendered
     */
    std::shared_ptr<const AgentData> agentData;
    /**
     * Holds information on core agent-wide texture buffers
     * e.g. location/direction
     */
    std::map<TexBufferConfig::Function, TexBufferConfig> core_tex_buffers;

    /**
     * Pass vis configs for each agent state to visualiser
     */
    void initBindings(std::unique_ptr<FLAMEGPU_Visualisation>& vis);
    /**
     * This requests that the visualisation resizes buffers
     * @param vis The affected visualisation
     * @param force When true is passed, vis will delay closing the splash screen until this update has been processed
     * Used when agent population has grown
     * @return Returns true if a non-0 buffer was requested
     */
    bool requestBufferResizes(std::unique_ptr<FLAMEGPU_Visualisation>& vis, bool force);
    /**
     * This passes the correct device pointers to the visualisation and forces it to update the data used for rendering
     * @param vis The affected visualisation
     * @note This should only be called when visualisation muted is held
     */
    void updateBuffers(std::unique_ptr<FLAMEGPU_Visualisation>& vis);
};

/**
 * This provides an interface for managing the render options for all agents of a specific type
 * State() can be called to specialise options for agents within a specific state
 * TODO: Block everything non-const from being called whilst VIS is active
 */
class AgentVis {
    friend class AgentStateVis;

 public:
    explicit AgentVis(std::shared_ptr<AgentVisData> data);
    /**
     * Returns the configuration handler for the named state
     * On first use for each state this will assign the state a color from the AutoPalette if available
     * Clear the autopalette first if you wish for it to use the default color
     */
    AgentStateVis State(const std::string &state_name);

    /**
     * Set the name of the variable representing the agents x/y/z location coordinates
     * @param var_name Name of the agent variable
     * @note unnecessary if the variables are named "x", "y", "z" respectively
     * @note Implicitly calls clearXYVariable(), clearXYZVariable()
     * @throws InvalidAgentVar If the variable is not type float[1]
     */
    void setXVariable(const std::string &var_name);
    void setYVariable(const std::string &var_name);
    void setZVariable(const std::string &var_name);
    /**
     * Set the name of the array variable (length 2) representing the agents x/y location coordinates
     * @param var_name Name of the agent variable
     * @note Implicitly calls clearXVariable(),  clearYVariable(), clearZVariable(),clearXYZVariable()
     * @throws InvalidAgentVar If the variable is not type float[2]
     */
    void setXYVariable(const std::string &var_name);
    /**
     * Set the name of the array variable (length 3) representing the agents x/y/z location coordinates
     * @param var_name Name of the agent variable
     * @note Implicitly calls clearXVariable(),  clearYVariable(), clearZVariable(),clearXYVariable()
     * @throws InvalidAgentVar If the variable is not type float[3]
     */
    void setXYZVariable(const std::string &var_name);
    /**
     * Set the name of the variable representing the agents x direction vector components
     * Single axis rotation only requires x/z components
     * Double axis rotation requires all 3 components
     * Triple axis rotation requires all 3 components and additionally all 3 Up components
     * @param var_name Name of the agent variable
     * @note setForwardXVariable() and setForwardZVariable() are an alternate to providing a yaw angle
     * @see setYawVariable(const std::string&)
     * @note Forward is a synonym for Direction
     * @note Implicitly calls clearHeadingVariable(), clearForwardXZVariable(), clearForwardXYZVariable(),
     * clearDirectionYPVariable(), clearDirectionYPRVariable()
     * @throws InvalidAgentVar If the variable is not type float[1]
     */
    void setForwardXVariable(const std::string& var_name);
    /**
     * Set the name of the variable representing the agents y direction vector components
     * Single axis rotation only requires x/z components
     * Double axis rotation requires all 3 components
     * Triple axis rotation requires all 3 components and additionally all 3 Up components
     * @param var_name Name of the agent variable
     * @note setForwardYVariable() is an alternate to providing a pitch angle
     * @see setPitchVariable(const std::string&)
     * @note Forward is a synonym for Direction
     * @note Implicitly calls clearPitchVariable(), clearForwardXYZVariable(), clearDirectionYPVariable(),
     * clearDirectionYPRVariable()
     * @throws InvalidAgentVar If the variable is not type float[1]
     */
    void setForwardYVariable(const std::string& var_name);
    /**
     * Set the name of the variable representing the agents z direction vector components
     * Single axis rotation only requires x/z components
     * Double axis rotation requires all 3 components
     * Triple axis rotation requires all 3 components and additionally all 3 Up components
     * @param var_name Name of the agent variable
     * @note setForwardXVariable() and setForwardZVariable() are an alternate to providing a yaw angle
     * @see setYawVariable(const std::string&)
     * @note Forward is a synonym for Direction
     * @note Implicitly calls clearHeadingVariable(), clearForwardXZVariable(), clearForwardXYZVariable(),
     * clearDirectionYPVariable(), clearDirectionYPRVariable()
     * @throws InvalidAgentVar If the variable is not type float[1]
     */
    void setForwardZVariable(const std::string& var_name);
    /**
     * Set the name of the array variable (length 2) representing the agents x/z direction vector components
     * Single axis rotation only requires x/z components
     * @param var_name Name of the agent variable
     * @note setForwardXVariable() and setForwardZVariable() are an alternate to providing a yaw angle, setting either of these will erase yaw if bound
     * @see setYawVariable(const std::string&)
     * @note setForwardYVariable() is an alternate to providing a pitch angle, setting this will erase pitch if bound
     * @see setPitchVariable(const std::string&)
     * @note Forward is a synonym for Direction
     * @note Implicitly calls clearHeadingVariable(), clearForwardXVariable(), clearForwardYVariable(),
     * clearForwardZVariable(), clearForwardXYZVariable(),clearDirectionYPVariable(), clearDirectionYPRVariable()
     * @throws InvalidAgentVar If the variable is not type float[2]
     */
    void setForwardXZVariable(const std::string& var_name);
    /**
     * Set the name of the array variable (length 3) representing the agents x/y/z direction vector components
     * Single axis rotation only requires x/z components
     * Double axis rotation requires all 3 components
     * Triple axis rotation requires all 3 components and additionally all 3 Up components
     * @param var_name Name of the agent variable
     * @note setForwardXVariable() and setForwardZVariable() are an alternate to providing a yaw angle, setting either of these will erase yaw if bound
     * @see setYawVariable(const std::string&)
     * @note setForwardYVariable() is an alternate to providing a pitch angle, setting this will erase pitch if bound
     * @see setPitchVariable(const std::string&)
     * @note Forward is a synonym for Direction
     * @note Implicitly calls clearHeadingVariable(), clearForwardXVariable(), clearForwardYVariable(),
     * clearForwardZVariable(), clearForwardXZVariable(),clearDirectionYPVariable(), clearDirectionYPRVariable()
     * @throws InvalidAgentVar If the variable is not type float[3]
     */
    void setForwardXYZVariable(const std::string& var_name);
    /**
     * Set the name of the variable representing the agents x/y/z UP vector
     * This should be 90 degrees perpendicular to the direction vector
     * @param var_name Name of the agent variable
     * @note setUpXVariable(), setUpYVariable() and setUpZVariable() are an alternate to providing a roll angle, setting any of these will erase roll if bound
     * @see setRollVariable(const std::string&)
     * @note Implicitly calls clearRollVariable(), clearUpXYZVariable(), clearDirectionYPRVariable()
     * @note Up can only be used in combination with Forward x/y/z (and not Yaw, Pitch or directionYP)
     * @throws InvalidAgentVar If the variable is not type float[1]
     */
    void setUpXVariable(const std::string& var_name);
    void setUpYVariable(const std::string& var_name);
    void setUpZVariable(const std::string& var_name);
    /**
     * Set the name of the array variable (length 3) representing the agents x/y/z UP vector
     * This should be 90 degrees perpendicular to the direction vector
     * @param var_name Name of the agent variable
     * @note setUpXVariable(), setUpYVariable() and setUpZVariable() are an alternate to providing a roll angle, setting any of these will erase roll if bound
     * @see setRollVariable(const std::string&)
     * @note Implicitly calls clearRollVariable(), clearUpXVariable(), clearUpYVariable(), clearUpZVariable(),
     * clearDirectionYPRVariable()
     * @note Up can only be used in combination with Forward x/y/z (and not Yaw, Pitch or directionYP)
     * @throws InvalidAgentVar If the variable is not type float[3]
     */
    void setUpXYZVariable(const std::string& var_name);
    /**
     * Set the name of the variable representing the agents yaw rotation angle (radians)
     *
     * @param var_name Name of the agent variable
     * @note This is an alternate to providing a direction vector, setting this will erase forward x/z if bound
     * @see setForwardXVariable(const std::string&)
     * @see setForwardZVariable(const std::string&)
     * @see setDirectionYPVariable(const std::string&)
     * @see setDirectionYPRVariable(const std::string&)
     * @note Heading is a synonym for Yaw
     * @note Implicitly calls clearForwardXVariable(), clearForwardYVariable(), clearForwardZVariable(),
     * clearForwardXYZVariable(), clearDirectionYPVariable(), clearDirectionYPRVariable()
     * @throws InvalidAgentVar If the variable is not type float[1]
     */
    void setYawVariable(const std::string& var_name);
    /**
     * Set the name of the variable representing the agents pitch rotation angle (radians)
     *
     * @param var_name Name of the agent variable
     * @note This is an alternate to providing a direction vector, setting this will erase forward y if bound
     * @see setForwardYVariable(const std::string&)
     * @see setDirectionYPVariable(const std::string&)
     * @see setDirectionYPRVariable(const std::string&)
     * @note Implicitly calls clearForwardYVariable(), clearForwardXYZVariable(), clearDirectionYPVariable(),
     * clearDirectionYPRVariable()
     */
    void setPitchVariable(const std::string& var_name);
    /**
     * Set the name of the variable representing the agents yaw rotation angle (radians)
     *
     * @param var_name Name of the agent variable
     * @note This is an alternate to providing an UP vector, setting this will erase up x/y/z if bound
     * @see setUpXVariable(const std::string&)
     * @see setUpYVariable(const std::string&)
     * @see setUpZVariable(const std::string&)
     * @see setDirectionYPRVariable(const std::string&)
     * @note Bank is a synonym for Roll
     * @note Implicitly calls clearUpXVariable(), clearUpYVariable(), clearUpZVariable(), clearUpXYZVariable(),
     * clearDirectionYPRVariable()
     * @throws InvalidAgentVar If the variable is not type float[1]
     */
    void setRollVariable(const std::string& var_name);
    /**
     * Set the name of the array variable (length 2) representing the agents yaw/pitch rotation angles (radians)
     *
     * @param var_name Name of the agent variable
     * @note This is an alternate to providing a direction vector, setting this will erase forward x/z if bound
     * @see setForwardXVariable(const std::string&)
     * @see setForwardZVariable(const std::string&)
     * @note Heading is a synonym for Yaw
     * @note Implicitly calls clearForwardXVariable(), clearForwardYVariable(), clearForwardZVariable(),
     * clearForwardXZVariable(), clearForwardXYZVariable(), clearHeadingVariable(), clearPitchVariable(),
     * clearDirectionYPRVariable()
     * @throws InvalidAgentVar If the variable is not type float[2]
     */
    void setDirectionYPVariable(const std::string& var_name);
    /**
     * Set the name of the array variable (length 3) representing the agents yaw/pitch/roll rotation angles (radians)
     *
     * @param var_name Name of the agent variable
     * @note This is an alternate to providing a direction vector, setting this will erase forward x/z if bound
     * @see setForwardXVariable(const std::string&)
     * @see setForwardZVariable(const std::string&)
     * @note Heading is a synonym for Yaw
     * @note Bank is a synonym for Roll
     * @note Implicitly calls clearForwardXVariable(), clearForwardYVariable(), clearForwardZVariable(),
     * clearForwardXZVariable(), clearForwardXYZVariable(), clearUpXVariable(), clearUpYVariable(), clearUpZVariable(),
     * clearUpXYZVariable(), clearHeadingVariable(), clearPitchVariable(), clearRollVariable(), clearDirectionYPVariable()
     * @throws InvalidAgentVar If the variable is not type float[3]
     */
    void setDirectionYPRVariable(const std::string& var_name);
    /**
     * Set the name of the variable representing the agents uniform scale multiplier
     *
     * The scale multiplier is multiplied by the model scale
     *
     * @param var_name Name of the agent variable
     * @note This is an alternate to providing individual scale components, setting this will erase scale x/y/z if bound
     * @see setScaleXVariable(const std::string&)
     * @see setScaleYVariable(const std::string&)
     * @see setScaleZVariable(const std::string&)
     * @see setModelScale(float)
     * @see setModelScale(float, float, float)
     */
    void setUniformScaleVariable(const std::string& var_name);
    /**
     * Set the name of the variable representing the agents x/y/z scale multiplier components
     * It is not necessary to set all 3 components if only 1 or 2 are required. Unset values will be treated as a 1.0 multiplier
     *
     * The scale multiplier is multiplied by the model scale
     *
     * @param var_name Name of the agent variable
     * @note This is an alternate to providing a single uniform scale multiplier, setting this will erase uniform scale if bound
     * @see setUniformScaleVariable(const std::string&)
     * @see setModelScale(float)
     * @see setModelScale(float, float, float)
     * @throws InvalidAgentVar If the variable is not type float[1]
     */
    void setScaleXVariable(const std::string& var_name);
    void setScaleYVariable(const std::string& var_name);
    void setScaleZVariable(const std::string& var_name);
    /**
     * Set the name of the array variable (length 2) representing the agents x/y scale multiplier components
     * It is not necessary to set all 3 components if only 1 or 2 are required. Unset values will be treated as a 1.0 multiplier
     *
     * The scale multiplier is multiplied by the model scale
     *
     * @param var_name Name of the agent variable
     * @note This is an alternate to providing a single uniform scale multiplier, setting this will erase uniform scale or individual scale components if bound
     * @see setUniformScaleVariable(const std::string&)
     * @see setModelScale(float)
     * @see setModelScale(float, float, float)
     * @throws InvalidAgentVar If the variable is not type float[2]
     */
    void setScaleXYVariable(const std::string& var_name);
    /**
     * Set the name of the array variable (length 3) representing the agents x/y/z scale multiplier components
     * It is not necessary to set all 3 components if only 1 or 2 are required. Unset values will be treated as a 1.0 multiplier
     *
     * The scale multiplier is multiplied by the model scale
     *
     * @param var_name Name of the agent variable
     * @note This is an alternate to providing a single uniform scale multiplier, setting this will erase uniform scale or individual scale components if bound
     * @see setUniformScaleVariable(const std::string&)
     * @see setModelScale(float)
     * @see setModelScale(float, float, float)
     * @throws InvalidAgentVar If the variable is not type float[3]
     */
    void setScaleXYZVariable(const std::string& var_name);
    /**
     * Clears the agent's x/y/z location variable bindings
     * @see setXVariable(const std::string &)
     * @see setYVariable(const std::string &)
     * @see setZVariable(const std::string &)
     */
    void clearXVariable();
    void clearYVariable();
    void clearZVariable();
    /**
     * Clears the agent's xy location variable bindings
     * @see setXYVariable(const std::string &)
     */
    void clearXYVariable();
    /**
     * Clears the agent's xyz location variable bindings
     * @see setXYZVariable(const std::string &)
     */
    void clearXYZVariable();
    /**
     * Clears the agent's x/y/z forward variable bindings
     * @see setForwardXVariable(const std::string &)
     * @see setForwardYVariable(const std::string &)
     * @see setForwardZVariable(const std::string &)
     */
    void clearForwardXVariable();
    void clearForwardYVariable();
    void clearForwardZVariable();
    /**
     * Clears the agent's xz forward variable bindings
     * @see setForwardXZVariable(const std::string &)
     */
    void clearForwardXZVariable();
    /**
     * Clears the agent's xyz forward variable bindings
     * @see setForwardXYZVariable(const std::string &)
     */
    void clearForwardXYZVariable();
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
     * Clears the agent's xyz UP variable bindings
     * @see setUpXYZVariable(const std::string &)
     */
    void clearUpXYZVariable();
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
     * Clears the agent's yaw angle variable bindings
     * @see setDirectionYPVariable(const std::string &)
     */
    void clearDirectionYPVariable();
    /**
     * Clears the agent's yaw angle variable bindings
     * @see setDirectionYPRVariable(const std::string &)
     */
    void clearDirectionYPRVariable();
    /**
     * Clears the agent's uniform scale multiplier variable bindings
     * @see setUniformScaleVariable(const std::string &)
     */
    void clearUniformScaleVariable();
    /**
     * Clears the agent's x/y/z scale multiplier variable bindings
     * @see setScaleXVariable(const std::string &)
     * @see setScaleYVariable(const std::string &)
     * @see setScaleZVariable(const std::string &)
     */
    void clearScaleXVariable();
    void clearScaleYVariable();
    void clearScaleZVariable();
    /**
     * Clears the agent's xy scale multiplier variable bindings
     * @see setScaleXYVariable(const std::string &)
     */
    void clearScaleXYVariable();
    /**
     * Clears the agent's xyz scale multiplier variable bindings
     * @see setScaleXYZVariable(const std::string &)
     */
    void clearScaleXYZVariable();
    /**
     * Returns the variable used for the agent's x/y/z location coordinates
     */
    std::string getXVariable() const;
    std::string getYVariable() const;
    std::string getZVariable() const;
    /**
     * Returns the variable used for the agent's xy location coordinates
     */
    std::string getXYVariable() const;
    /**
     * Returns the variable used for the agent's xyz location coordinates
     */
    std::string getXYZVariable() const;
    /**
     * Returns the variable used for the agent's x/y/z forward vector components
     */
    std::string getForwardXVariable() const;
    std::string getForwardYVariable() const;
    std::string getForwardZVariable() const;
    /**
     * Returns the variable used for the agent's xz forward vector components
     */
    std::string getForwardXZVariable() const;
    /**
     * Returns the variable used for the agent's xyz forward vector components
     */
    std::string getForwardXYZVariable() const;
    /**
     * Returns the variable used for the agent's x/y/z up vector components
     */
    std::string getUpXVariable() const;
    std::string getUpYVariable() const;
    std::string getUpZVariable() const;
    /**
     * Returns the variable used for the agent's xyz up vector components
     */
    std::string getUpXYZVariable() const;
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
     * Returns the variable used for the agent's roll angle
     */
    std::string getDirectionYPVariable() const;
    /**
     * Returns the variable used for the agent's roll angle
     */
    std::string getDirectionYPRVariable() const;
    /**
     * Returns the variable used for the agent's uniform scaling multiplier
     */
    std::string getUniformScaleVariable() const;
    /**
     * Returns the variable used for the agent's x/y/z scale multiplier components
     */
    std::string getScaleXVariable() const;
    std::string getScaleYVariable() const;
    std::string getScaleZVariable() const;
    /**
     * Returns the variable used for the agent's xy scale multiplier components
     */
    std::string getScaleXYVariable() const;
    /**
     * Returns the variable used for the agent's xyz scale multiplier components
     */
    std::string getScaleXYZVariable() const;

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
     * Use a keyframe animated model from file
     * @param modelPathA The path to the model's first file (must be .obj)
     * @param modelPathB The path to the model's second file (must be .obj, have the same number of vertices/polygons as the first file)
     * @param lerpVariableName Name of the agent variable used for controlling linear interpolation between the two frames. This must be a `float` type variable with a value in the inclusive-inclusive range [0, 1]
     * @param texturePath Optional path to the texture used by the two models
     * @see setModel(const std::string &, const std::string &) This version can be used to provide agents a static model
     */
    void setKeyFrameModel(const std::string& modelPathA, const std::string& modelPathB, const std::string& lerpVariableName, const std::string& texturePath = "");
    /**
     * Use a stock keyframe animated model
     * @param model Model from the libraries internal resources
     * @param lerpVariableName Name of the agent variable used for controlling linear interpolation between the two frames. This must be a `float` type variable with a value in the inclusive-inclusive range [0, 1]
     * @see setModel(const Stock::Models::Model &) This version can be used to provide agents a static model
     */
    void setKeyFrameModel(const Stock::Models::KeyFrameModel& model, const std::string& lerpVariableName);
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
     * Pointer to data struct
     */
    std::shared_ptr<AgentVisData> data;
};

}  // namespace visualiser
}  // namespace flamegpu

#endif  // FLAMEGPU_VISUALISATION
#endif  // INCLUDE_FLAMEGPU_VISUALISER_AGENTVIS_H_
