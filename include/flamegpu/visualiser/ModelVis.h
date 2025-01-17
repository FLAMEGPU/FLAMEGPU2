#ifndef INCLUDE_FLAMEGPU_VISUALISER_MODELVIS_H_
#define INCLUDE_FLAMEGPU_VISUALISER_MODELVIS_H_
#ifdef FLAMEGPU_VISUALISATION

#include <string>
#include <unordered_map>
#include <thread>
#include <memory>
#include <climits>

// @todo - All vis headers should live in the vis repo.
#include "flamegpu/visualiser/AgentVis.h"
#include "flamegpu/visualiser/EnvironmentGraphVis.h"
#include "flamegpu/visualiser/StaticModelVis.h"
#include "flamegpu/visualiser/LineVis.h"
#include "flamegpu/visualiser/PanelVis.h"
#include "flamegpu/visualiser/color/AutoPalette.h"
#include "flamegpu/visualiser/config/ModelConfig.h"

namespace flamegpu {

struct ModelData;
class CUDASimulation;
class FLAMEGPU_Visualisation;

namespace visualiser {

struct ModelVisData {
    /**
     * This class is constructed by/with a CUDASimulation
     * Constructor will be clarified later, once requirements are clearer
     * Requires:
     * > texturise agent variable pointers
     * > On resize, also update textures
     */
    explicit ModelVisData(const CUDASimulation& model/*TBD*/);
    /**
     * Pass the vis shared pointer to directed graphs being visualised so they can trigger updates
     * @param vis pointer to this
     * @param map the map of environment directed graph cuda buffers
     */
    void hookVis(std::shared_ptr<visualiser::ModelVisData> &vis, std::unordered_map<std::string, std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers>> &map);
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
    std::unordered_map<std::string, std::shared_ptr<AgentVisData>> agents;
    /**
     * Per environment direct graph, visualisation configuration options
     */
    std::unordered_map<std::string, std::shared_ptr<EnvironmentGraphVisData>> graphs;
    /**
     * Reference back to the model to be visualised
     */
    const CUDASimulation& model;
    /**
     * Reference back to the model description hierarchy to be visualised
     */
    const ModelData& modelData;
    /**
     * Pointer to the visualisation
     */
    std::unique_ptr<FLAMEGPU_Visualisation> visualiser;
    /**
     * Only need to register env properties once
     */
    bool env_registered = false;
    /**
     * Updates all agent renders from corresponding
     * @param sc Step count, the step count value shown in visualisation HUD
     */
    void updateBuffers(const unsigned int& sc = UINT_MAX);
    /**
     * Singletons have init, so env props are ready to grab
     */
    void registerEnvProperties();
    /**
     * Random seed has changed
     */
    void updateRandomSeed();
    /**
     * Rebuild all environment graph sketches
     */
    void buildEnvGraphs();
    /**
     * Rebuild a specific environment graph sketch
     */
    void rebuildEnvGraph(const std::string& graph_name);
};

/**
 * This provides an interface for managing the render options for a specific CUDASimulation
 */
class ModelVis {
 public:
    explicit ModelVis(std::shared_ptr<ModelVisData> data, bool _isSWIG);
    /**
     * Default destructor behaviour
     * Defined explicitly, so that header include does not require including FLAMEGPU_Visualisation for std::unique_ptr destruction.
     */
    ~ModelVis() = default;
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
    AgentVis addAgent(const std::string &agent_name);
    /**
     * Returns the configuration handler if the agent has been marked for visualisation
     * @see addAgent(const std::string&)
     */
    AgentVis Agent(const std::string &agent_name);
    /**
     * Select a graph to be rendered
     * @param graph_name The name of the environment directed graph to visualise
     * @return A handle to configure the visualisation of the specified graph
     */
    EnvironmentGraphVis addGraph(const std::string &graph_name);
    /**
     * Returns the configuration handler if the environment directed graph has been marked for visualisation
     * @see addGraph(const std::string&)
     */
    EnvironmentGraphVis Graph(const std::string& graph_name);
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
     * Set the initial camera roll in radians
     * This value defaults to 0
     * @param roll The roll angle in radians
     */
    void setInitialCameraRoll(const float &roll);
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
     * Sets whether the visualisation should use an orthographic (or perspective) projection
     * This value defaults to false
     *
     * Orthographic projection can be toggled during the visualisation by pressing 'F9'
     * Orthographic projection is used for 2D models, whereby depth should not affect scale
     * @param isOrtho True if the visualisation should use an orthographic projection
     */
    void setOrthographic(const bool& isOrtho);
    /**
     * Sets initial zoom modifier for the orthographic projection
     * This value defaults to 1.0
     * This setting has no impact on perspective projection mode
     * This value must be greater than 0.001, which is the maximum/closest zoom supported.
     * @param zoomMod The initial zoom modifier
     */
    void setOrthographicZoomModifier(const float& zoomMod);
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
    StaticModelVis newStaticModel(const std::string &modelPath, const std::string &texturePath = "");
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
     * Add a customisable user interface panel to the visualisation
     *
     * Each panel can be moved around the screen/minimised with custom elements representing environment properties added
     * @param panel_title The string that will be visible in the title of the panel
     */
    PanelVis newUIPanel(const std::string& panel_title);
    /**
     * Sets the visualisation running in a background thread
     */
    void activate();
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

 private:
    const bool isSWIG;
    std::shared_ptr<ModelVisData> data;
};

}  // namespace visualiser
}  // namespace flamegpu

#endif  // FLAMEGPU_VISUALISATION
#endif  // INCLUDE_FLAMEGPU_VISUALISER_MODELVIS_H_
