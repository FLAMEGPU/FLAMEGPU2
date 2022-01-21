#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBRUTEFORCESORTED_MESSAGEBRUTEFORCESORTEDDEVICE_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBRUTEFORCESORTED_MESSAGEBRUTEFORCESORTEDDEVICE_CUH_

#include "flamegpu/runtime/messaging/MessageBruteForceSorted.h"
#include "flamegpu/runtime/messaging/MessageSpatial2D/MessageSpatial2DDevice.cuh"
#include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceDevice.cuh"

namespace flamegpu {

/**
 * This class is accessible via DeviceAPI.message_in if MessageBruteForceSorted is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for reading spatially partitioned messages
 */
class MessageBruteForceSorted::In {
 public:

    /**
     * Constructer
     * Initialises member variables
     * @param agentfn_hash Added to message_hash to produce combined_hash
     * @param message_hash Added to agentfn_hash to produce combined_hash
     * @param _metadata Reinterpreted as type MessageBruteForceSorted::MetaData
     */
    __device__ In(detail::curve::Curve::NamespaceHash agentfn_hash, detail::curve::Curve::NamespaceHash message_hash, const void *_metadata)
        : combined_hash(agentfn_hash + message_hash)
        , metadata(reinterpret_cast<const MetaData*>(_metadata))
    { }

 private:
    /**
     * CURVE hash for accessing message data
     * agentfn_hash + message_hash
     */
    detail::curve::Curve::NamespaceHash combined_hash;
    /**
     * Device pointer to metadata required for accessing data structure
     * e.g. PBM, search origin, environment bounds
     */
    const MetaData *metadata;
};

/**
 * This class is accessible via DeviceAPI.message_out if MessageBruteForceSorted is specified in FLAMEGPU_AGENT_FUNCTION
 * It gives access to functionality for outputting spatially partitioned messages
 */
class MessageBruteForceSorted::Out : public MessageBruteForce::Out {
 public:
    /**
     * Constructer
     * Initialises member variables
     * @param agentfn_hash Added to message_hash to produce combined_hash
     * @param message_hash Added to agentfn_hash to produce combined_hash
     * @param scan_flag_messageOutput Scan flag array for optional message output
     */
    __device__ Out(detail::curve::Curve::NamespaceHash agentfn_hash, detail::curve::Curve::NamespaceHash message_hash, const void *, unsigned int *scan_flag_messageOutput)
        : MessageBruteForce::Out(agentfn_hash, message_hash, nullptr, scan_flag_messageOutput)
    { }
    /**
     * Sets the location for this agents message
     * @param x Message x coord
     * @param y Message y coord
     * @param z Message z coord
     * @note Convenience wrapper for setVariable()
     */
    __device__ void setLocation(const float &x, const float &y, const float &z) const;
};

__device__ __forceinline__ MessageBruteForceSorted::GridPos3D getGridPosition3D(const MessageBruteForceSorted::MetaData *md, float x, float y, float z) {
    // Clamp each grid coord to 0<=x<dim
    int gridPos[3] = {
        static_cast<int>(floorf(((x-md->min[0]) / md->environmentWidth[0])*md->gridDim[0])),
        static_cast<int>(floorf(((y-md->min[1]) / md->environmentWidth[1])*md->gridDim[1])),
        static_cast<int>(floorf(((z-md->min[2]) / md->environmentWidth[2])*md->gridDim[2]))
    };
    MessageBruteForceSorted::GridPos3D rtn = {
        gridPos[0] < 0 ? 0 : (gridPos[0] >= static_cast<int>(md->gridDim[0]) ? static_cast<int>(md->gridDim[0]) - 1 : gridPos[0]),
        gridPos[1] < 0 ? 0 : (gridPos[1] >= static_cast<int>(md->gridDim[1]) ? static_cast<int>(md->gridDim[1]) - 1 : gridPos[1]),
        gridPos[2] < 0 ? 0 : (gridPos[2] >= static_cast<int>(md->gridDim[2]) ? static_cast<int>(md->gridDim[2]) - 1 : gridPos[2])
    };
    return rtn;
}
__device__ __forceinline__ unsigned int getHash3D(const MessageBruteForceSorted::MetaData *md, const MessageBruteForceSorted::GridPos3D &xyz) {
    // Bound gridPos to gridDimensions
    unsigned int gridPos[3] = {
        (unsigned int)(xyz.x < 0 ? 0 : (xyz.x >= static_cast<int>(md->gridDim[0]) - 1 ? static_cast<int>(md->gridDim[0]) - 1 : xyz.x)),  // Only x should ever be out of bounds here
        (unsigned int) xyz.y,  // xyz.y < 0 ? 0 : (xyz.y >= md->gridDim[1] - 1 ? md->gridDim[1] - 1 : xyz.y),
        (unsigned int) xyz.z,  // xyz.z < 0 ? 0 : (xyz.z >= md->gridDim[2] - 1 ? md->gridDim[2] - 1 : xyz.z)
    };
    // Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
    return (unsigned int)(
        (gridPos[2] * md->gridDim[0] * md->gridDim[1]) +   // z
        (gridPos[1] * md->gridDim[0]) +                    // y
        gridPos[0]);                                      // x
}

__device__ inline void MessageBruteForceSorted::Out::setLocation(const float &x, const float &y, const float &z) const {
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // set the variables using curve
    detail::curve::Curve::setMessageVariable<float>("x", combined_hash, x, index);
    detail::curve::Curve::setMessageVariable<float>("y", combined_hash, y, index);
    detail::curve::Curve::setMessageVariable<float>("z", combined_hash, z, index);

    // Set scan flag incase the message is optional
    this->scan_flag[index] = 1;
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGEBRUTEFORCESORTED_MESSAGEBRUTEFORCESORTEDDEVICE_CUH_
