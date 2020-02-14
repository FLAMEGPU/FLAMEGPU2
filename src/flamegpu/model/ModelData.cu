#include "flamegpu/model/ModelData.h"
#include "flamegpu/runtime/messaging.h"

std::unique_ptr<MsgSpecialisationHandler> MessageData::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MsgSpecialisationHandler>(new MsgBruteForce::CUDAModelHandler(owner));
}
std::unique_ptr<MsgSpecialisationHandler> Spatial3DMessageData::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MsgSpecialisationHandler>(new MsgSpatial3D::CUDAModelHandler(owner));
}

// Used for the MessageData::getType() type and derived methods
class MsgSpatial2D {};  // Haven't created this yet, but we have the description
std::type_index MessageData::getType() const { return std::type_index(typeid(MsgBruteForce)); }
std::type_index Spatial2DMessageData::getType() const { return std::type_index(typeid(MsgSpatial2D)); }
std::type_index Spatial3DMessageData::getType() const { return std::type_index(typeid(MsgSpatial3D)); }
