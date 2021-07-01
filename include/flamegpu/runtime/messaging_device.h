#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_DEVICE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_DEVICE_H_

/**
 * This file provides the root include for messaging on the device
 */

#include "flamegpu/runtime/messaging/None/NoneDevice.cuh"
#include "flamegpu/runtime/messaging/BruteForce/BruteForceDevice.cuh"
#include "flamegpu/runtime/messaging/Spatial2D/Spatial2DDevice.cuh"
#include "flamegpu/runtime/messaging/Spatial3D/Spatial3DDevice.cuh"
#include "flamegpu/runtime/messaging/Array/ArrayDevice.cuh"
#include "flamegpu/runtime/messaging/Array2D/Array2DDevice.cuh"
#include "flamegpu/runtime/messaging/Array3D/Array3DDevice.cuh"
#include "flamegpu/runtime/messaging/Bucket/BucketDevice.cuh"


#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_DEVICE_H_
