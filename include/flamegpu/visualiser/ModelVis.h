#ifndef INCLUDE_FLAMEGPU_VISUALISER_MODELVIS_H_
#define INCLUDE_FLAMEGPU_VISUALISER_MODELVIS_H_
#ifdef VISUALISATION

#include <string>
#include <unordered_map>
#include <thread>
#include <memory>
#include <climits>

#include "flamegpu/visualiser/AgentVis.h"
#include "flamegpu/visualiser/StaticModelVis.h"
#include "flamegpu/visualiser/LineVis.h"
#include "flamegpu/visualiser/color/AutoPalette.h"
#include "config/ModelConfig.h"

namespace flamegpu {

struct ModelData;
class CUDASimulation;
class FLAMEGPU_Visualisation;

namespace visualiser {

/**
 * This provides an interface for managing the render options for a specific CUDASimulation
 */
class ModelVis {
 public:
    /**
     * This class is constructed by/with a CUDASimulation
     * Constructor will be clarified later, once requirements are clearer
     * Requires:
     * > texturise agent variable pointers
     * > On resize, also update textures
     */
    explicit ModelVis(const flamegpu::CUDASimulation &model/*TBD*/);
    /**
     * Default destructor behaviour
     * Defined explicitly, so that header include does not require including FLAMEGPU_Visualisation for std::unique_ptr destruction.
     */
    ~ModelVis();
    /**
     * Sets the palette to automatically give color to agent added to the model
     * This can be overriden at an agent level or disabled
     * Similarly, individual agent-states can have their colour overriden
     */
    void setAutoPalette(const Palette &palette);
    /**
     * Disables the auto-palette, subsequently created AgentVis/AgentStateVis will not take colors from it
     * @note AgentVis/AgentStateVis which have already sampled a color will not lose their existing color
     */
    void clearAutoPalette();
    /**
     * Enables visualisation of the named agent and returns the configuration handler
     * @see Agent(const std::string&)
     * @todo Block this from being called whilst visualiser is allocated
     */
    AgentVis &addAgent(const std::string &agent_name);
    /**
     * Returns the configuration handler if the agent has been marked for visualisation
     * @see addAgent(const std::string&)
     */
    AgentVis &Agent(const std::string &agent_name);
    /**
     * Set the title for the visualisation window
     * This value defaults to the model's name
     * @param title The title for the viusalisation window
     */
    void setWindowTitle(const std::string &title);
    /**
     * Set the dimensions of the visualisation window
     * This value defaults to 1280x720 (720p)
     * @param width Window width
     * @param height Window height
     */
    void setWindowDimensions(const unsigned int& width, const unsigned int& height);
    /**
     * Set the clear color (the background color) of the visualisation
     * This value defaults to black (0,0,0)
     * @param red Red color value 0.0f-1.0f
     * @param green Green color value 0.0f-1.0f
     * @param blue Blue color value 0.0f-1.0f
     */
    void setClearColor(const float& red, const float& green, const float& blue);
	/**
	 * Sets the FPS overlay as visible or not
	 * This value defaults to true
	 * @param showFPS True if the FPS should be shown
	 * @note The visualisation executes in an independent thread to the simulation,
	 * so the FPS does not correspond to the speed of the simulation's execution
	 * @see setFPSColor(const float &, const float &, const float &)
	 */
    void setFPSVisible(const bool& showFPS);
	/**
	 * Sets the color of the FPS overlay's text
	 * This value defaults to white (1,1,1)
	 * This  may be useful if you have changed the clear color
     * @param red Red color value 0.0f-1.0f
     * @param green Green color value 0.0f-1.0f
     * @param blue Blue color value 0.0f-1.0f
	 */
    void setFPSColor(const float& red, const float& green, const float& blue);
    /**
     * The location at which the camera of the visualisation 'camera' begins
     * This value defaults to (1.5, 1.5, 1.5)
     * @param x The x coordinate
     * @param y The y coordinate
     * @param z The z coordinate
     */
    void setInitialCameraLocation(const float &x, const float &y, const float &z);
    /**
     * The location at which the camera of the visualisation initially looks towards
     * This is used with the camera location to derive the direction
     * This value defaults to (0,0,0)
     * @param x The x coordinate
     * @param y The y coordinate
     * @param z The z coordinate
     */
    void setInitialCameraTarget(const float &x, const float &y, const float &z);
    /**
     * The speed of camera movement, in units travelled per millisecond
     * This value defaults to (0.05, 5.0)
     * @param speed The camera speed
     * @param shiftMultiplier The multiplier applied to the speed when shift is pressed
     */
    void setCameraSpeed(const float &speed, const float &shiftMultiplier = 5.0f);

    /**
     * Sets the near and far clipping planes of the view frustum
     * This value defaults to (0.05 5000.0)
     * @note This is for advanced configuration of the visualisation and the default values likely suffice
     */
    void setViewClips(const float &nearClip, const float &farClip);
    /**
	 * Sets the Step count overlay as visible or not
	 * This value defaults to true
	 * @param showStep True if the count should be shown
	 * @note This uses the FPSColor
	 * @see setFPSColor(const float &, const float &, const float &)
	 */
    void setStepVisible(const bool& showStep);
    /**
     * Sets a limit on the rate of simulation
     * A value of 0 leaves the rate unlimited
     * This value defaults to 0
     * @param stepsPerSecond The number of simulation steps to execute per second
     */
    void setSimulationSpeed(const unsigned int& stepsPerSecond);
    /**
     * Sets whether the simulation should begin in a paused state or not
     * This value defaults to false
     * The simulation can be resumed (or re-paused) by pressing 'p'
	 * @param beginPaused True if the simulation should begin paused
     */
    void setBeginPaused(const bool& beginPaused);
    /**
     * Adds a static model to the visualisation
     * @param modelPath Path of the model on disk
     * @param texturePath Optional path to a texture fore the model on disk
     */
    StaticModelVis addStaticModel(const std::string &modelPath, const std::string &texturePath = "");
    /**
     * Create a new sketch constructed from individual line segments to the visualisation
     * @param r Initial color's red component
     * @param g Initial color's green component
     * @param b Initial color's blue component
     * @param a Initial color's alpha component
     */
    LineVis newLineSketch(float r, float g, float b, float a = 1.0f);
    /**
     * Create a new sketch constructed from a single line of connected vertices to the visualisation
     * @param r Initial color's red component
     * @param g Initial color's green component
     * @param b Initial color's blue component
     * @param a Initial color's alpha component
     */
    LineVis newPolylineSketch(float r, float g, float b, float a = 1.0f);
    /**
     * Sets the visualisation running in a background thread
     */
    void activate() {
#ifdef SWIG
        modelCfg.isPython = true;
#endif
        _activate();
    }
    /**
     * Kills the background thread
     * Does nothing visualisation is not running
     */
    void deactivate();
    /**
     * Blocks the main thread until the background visualisation thread has returned
     * Does nothing visualisation is not running
     * @note It is expected that you will close the visualiser window with the cross in the corner if join is called
     */
    void join();
    /**
     * Returns whether the background thread is active or not
     */
    bool isRunning() const;
    /**
     * Updates all agent renders from corresponding
     * @param sc Step count, the step count value shown in visualisation HUD
     */
    void updateBuffers(const unsigned int &sc = UINT_MAX);

 private:
    /**
     * Private version called by public version
     * Public version is defined inline so swig can get a different version
     */
    void _activate();
    /**
     * Main struct of visualisation configuration options for the model
     */
    ModelConfig modelCfg;
    /**
     * Autopalette which provides default colors to all agents
     * By default this uses Stock::Palettes:DARK2
     */
    std::shared_ptr<AutoPalette> autoPalette;
    /**
     * Per agent, visualisation configuration options
     */
    std::unordered_map<std::string, AgentVis> agents;
    /**
     * Reference back to the model to be visualised
     */
    const CUDASimulation &model;
    /**
     * Reference back to the model description hierarchy to be visualised
     */
    const ModelData &modelData;
    /**
     * Pointer to the visualisation
     */
    std::unique_ptr<FLAMEGPU_Visualisation> visualiser;
};

}  // namespace visualiser
}  // namespace flamegpu

#endif  // VISUALISATION
#endif  // INCLUDE_FLAMEGPU_VISUALISER_MODELVIS_H_
