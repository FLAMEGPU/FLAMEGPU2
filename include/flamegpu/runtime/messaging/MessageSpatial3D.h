#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESPATIAL3D_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESPATIAL3D_H_

#include "flamegpu/runtime/messaging/MessageBruteForce.h"

namespace flamegpu {

/**
 * 3D Continuous spatial messaging functionality
 *
 * User specifies the environment bounds and search radius
 * When accessing messages, a search origin is specified
 * A subset of messages, including those within radius of the search origin are returned
 * The user must distance check that they fall within the search radius manually
 * Unlike FLAMEGPU1, these spatial messages do not wrap over environment bounds.
 */
class MessageSpatial3D {
 public:
    // Host
    struct Data;        // Forward declare inner classes
    class Description;  // Forward declare inner classes
    class CUDAModelHandler;
    // Device
    class In;
    class Out;

    /**
     * Basic class to group 3 dimensional bin coordinates
     * Would use glm::ivec3, but project does not currently have glm
     */
    struct GridPos3D {
        int x, y, z;
    };
    /**
     * MetaData required by spatial partitioning during message reads
     */
    struct MetaData {
        /**
         * Minimum environment bounds
         */
        float min[3];
        /**
         * Maximum environment bounds
         */
        float max[3];
        /**
         * Search radius (also used as subdividision bin width)
         */
        float radius;
        /**
         * Pointer to the partition boundary matrix in device memory
         * The PBM is never stored on the host
         */
        unsigned int *PBM;
        /**
         * The number of subdividision bins in each dimensions
         */
        unsigned int gridDim[3];
        /**
         * max-lowerBound
         */
        float environmentWidth[3];
    };
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_MESSAGESPATIAL3D_H_
