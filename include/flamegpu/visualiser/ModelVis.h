#ifndef INCLUDE_FLAMEGPU_VISUALISER_MODELVIS_H_
#define INCLUDE_FLAMEGPU_VISUALISER_MODELVIS_H_
#ifdef VISUALISATION

#include <string>
#include <unordered_map>
#include <thread>
#include <memory>
#include "flamegpu/visualiser/AgentVis.h"
#include "FLAMEGPU_Visualisation.h"
#include "config/ModelConfig.h"

struct ModelData;
class CUDAAgentModel;

/**
 * This provides an interface for managing the render options for a specific CUDAAgentModel
 */
class ModelVis {
 public:
    /**
     * This class is constructed by/with a CUDAAgentModel
     * Constructor will be clarified later, once requirements are clearer
     * Requires:
     * > texturise agent variable pointers
     * > On resize, also update textures
     */
    explicit ModelVis(const CUDAAgentModel &model/*TBD*/);

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
    /**
     * Updates all agent renders from corresponding
     */
    void updateBuffers();

 private:
    ModelConfig modelCfg;
    std::unordered_map<std::string, AgentVis> agents;
    const CUDAAgentModel &model;
    const ModelData &modelData;

    std::unique_ptr<FLAMEGPU_Visualisation> visualiser;
};

#endif  // VISUALISATION
#endif  // INCLUDE_FLAMEGPU_VISUALISER_MODELVIS_H_
