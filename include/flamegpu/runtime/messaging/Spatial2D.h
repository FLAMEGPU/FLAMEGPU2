#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL2D_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL2D_H_

#include "flamegpu/runtime/messaging/BruteForce.h"

/**
 * 2D Continuous spatial messaging functionality
 *
 * User specifies the environment bounds and search radius
 * When accessing messages, a search origin is specified
 * A subset of messages, including those within radius of the search origin are returned
 * The user must distance check that they fall within the search radius manually
 * Unlike FLAMEGPU1, these spatial messages do not wrap over environment bounds.
 */
class MsgSpatial2D {
    /**
     * Common size type
     */
    typedef MsgNone::size_type size_type;

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
    struct GridPos2D {
        int x, y;
    };

    /**
     * MetaData required by spatial partitioning during message reads
     */
    struct MetaData {
        /**
         * Minimum environment bounds
         */
        float min[2];
        /**
         * Maximum environment bounds
         */
        float max[2];
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
        unsigned int gridDim[2];
        /**
         * max-min
         */
        float environmentWidth[3];
    };
};
#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL2D_H_
