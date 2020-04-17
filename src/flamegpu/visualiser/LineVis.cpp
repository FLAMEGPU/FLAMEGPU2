#include "flamegpu/visualiser/LineVis.h"

LineVis::LineVis(std::shared_ptr<LineConfig> _l, float r, float g, float b, float a)
    : currentColor{r, g, b, a}
    , l(std::move(_l)) { }

void LineVis::setColor(float r, float g, float b, float a) {
    currentColor[0] = r;
    currentColor[1] = g;
    currentColor[2] = b;
    currentColor[3] = a;
}

void LineVis::addVertex(float x, float y, float z) {
    // New vertex info
    l->vertices.push_back(x);
    l->vertices.push_back(y);
    l->vertices.push_back(z);
    // New color info
    l->colors.push_back(currentColor[0]);
    l->colors.push_back(currentColor[1]);
    l->colors.push_back(currentColor[2]);
    l->colors.push_back(currentColor[3]);
}
