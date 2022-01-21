#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_DEVICE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_DEVICE_H_

/**
 * This file provides the root include for messaging on the device
 */

#include "flamegpu/runtime/messaging/MessageNone/MessageNoneDevice.cuh"
#include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceDevice.cuh"
#include "flamegpu/runtime/messaging/MessageSpatial2D/MessageSpatial2DDevice.cuh"
#include "flamegpu/runtime/messaging/MessageSpatial3D/MessageSpatial3DDevice.cuh"
#include "flamegpu/runtime/messaging/MessageArray/MessageArrayDevice.cuh"
#include "flamegpu/runtime/messaging/MessageArray2D/MessageArray2DDevice.cuh"
#include "flamegpu/runtime/messaging/MessageArray3D/MessageArray3DDevice.cuh"
#include "flamegpu/runtime/messaging/MessageBucket/MessageBucketDevice.cuh"
#include "flamegpu/runtime/messaging/MessageBruteForceSorted/MessageBruteForceSortedDevice.cuh"


#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_DEVICE_H_
