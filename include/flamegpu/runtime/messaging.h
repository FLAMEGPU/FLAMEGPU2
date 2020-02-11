#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_

#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/runtime/messaging/Spatial2D.h"

/**
 * This empty class is used when messaging is not enabled for an agent function
 */
class MsgNone
{
public:
    class In
    {
    public:
        In() {}
        __device__ In(Curve::NamespaceHash, Curve::NamespaceHash, unsigned int)
        {

        }
    };
    class Out
    {
    public:
        Out() {}
        __device__ Out(Curve::NamespaceHash, Curve::NamespaceHash)
        {

        }
    };
};
#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_