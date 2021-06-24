// @todo - ifdef visualisation
#include "flamegpu/visualiser/StaticModelVis.h"

#include "flamegpu/exception/FGPUException.h"

namespace flamegpu {
namespace visualiser {

StaticModelVis::StaticModelVis(std::shared_ptr<ModelConfig::StaticModel> _m)
    : m(std::move(_m)) { }

void StaticModelVis::setModelScale(float xLen, float yLen, float zLen) {
    if (xLen <= 0 || yLen <= 0 || zLen <= 0) {
        THROW exception::InvalidArgument("StaticModelVis::setModelScale(): Invalid argument, lengths must all be positive.\n");
    }
    m->scale[0] = xLen;
    m->scale[1] = yLen;
    m->scale[2] = zLen;
}
void StaticModelVis::setModelScale(float maxLen) {
    if (maxLen <= 0) {
        THROW exception::InvalidArgument("StaticModelVis::setModelScale(): Invalid argument, maxLen must be positive.\n");
    }
    m->scale[0] = -maxLen;
}
void StaticModelVis::setModelLocation(float x, float y, float z) {
    m->location[0] = x;
    m->location[1] = y;
    m->location[2] = z;
}
void StaticModelVis::setModelRotation(float x, float y, float z, float radians) {
    if (x == 0 && y == 0 && z == 0) {
        THROW exception::InvalidArgument("StaticModelVis::setModelRotation(): Invalid argument, axis cannot be (0,0,0).\n");
    }
    m->rotation[0] = x;
    m->rotation[1] = y;
    m->rotation[2] = z;
    m->rotation[3] = radians;
}

}  // namespace visualiser
}  // namespace flamegpu
