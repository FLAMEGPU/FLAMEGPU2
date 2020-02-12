#include "flamegpu/model/ModelData.h"
#include "flamegpu/runtime/messaging/Spatial3D.h"

std::unique_ptr<MsgSpecialisationHandler<CUDAMessage>> MessageData::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MsgSpecialisationHandler<CUDAMessage>>(new MsgBruteForce::CUDAModelHandler<CUDAMessage>(owner));
}
std::unique_ptr<MsgSpecialisationHandler<CUDAMessage>> Spatial3DMessageData::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MsgSpecialisationHandler<CUDAMessage>>(new MsgSpatial3D::CUDAModelHandler<CUDAMessage>(owner));
}