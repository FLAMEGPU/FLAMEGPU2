#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_

/**
 * This file provides the root include for messaging
 * The bulk of message specialisation is implemented within headers included at the bottom of this file
 */

#include "flamegpu/runtime/messaging/None.h"
#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/runtime/messaging/Spatial2D.h"
#include "flamegpu/runtime/messaging/Spatial3D.h"

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_
