#ifndef INCLUDE_FLAMEGPU_VISUALISER_LINEVIS_H_
#define INCLUDE_FLAMEGPU_VISUALISER_LINEVIS_H_
#ifdef VISUALISATION

#include <memory>

#include "config/LineConfig.h"

/**
 * Interface for managing a LineConfig
 */
class LineVis {
 public:
    /**
     * Create a new interface for managing a LineConfig
     * @param l The line config  being constructed
     * @param r Initial color's red component
     * @param g Initial color's green component
     * @param b Initial color's blue component
     * @param a Initial color's alpha component
     */
    LineVis(std::shared_ptr<LineConfig> l, float r, float g, float b, float a);
    /**
     * Update the color for following vertices
     * @param r Color's red component
     * @param g Color's green component
     * @param b Color's blue component
     * @param a Color's alpha component
     */
    void setColor(float r, float g, float b, float a = 1.0f);
    /**
     * Adds a new vertex to the drawing
     * @param x Vertex's x coord
     * @param y Vertex's y coord
     * @param z Vertex's z coord
     * @note Y is considered the vertical axis
     */
    void addVertex(float x, float y, float z = 0.0f);

 private:
    /**
     * The color used for any new vertices
     * @see setColor(float, float, float, float)
     */
    float currentColor[4];
    /**
     * The line data which this class acts as an interface for managing
     */
    std::shared_ptr<LineConfig> l;
};

#endif  // VISUALISATION
#endif  // INCLUDE_FLAMEGPU_VISUALISER_LINEVIS_H_
