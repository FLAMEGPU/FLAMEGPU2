#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_CUH_

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "flamegpu/runtime/detail/SharedBlock.h"
#include "flamegpu/defines.h"
#include "flamegpu/exception/FLAMEGPUDeviceException.cuh"
#include "flamegpu/runtime/AgentFunction_shim.cuh"

namespace flamegpu {

// ! FLAMEGPU function return type
enum AGENT_STATUS { ALIVE = 1, DEAD = 0 };

typedef void(AgentFunctionWrapper)(
#if !defined(SEATBELTS) || SEATBELTS
    exception::DeviceExceptionBuffer *error_buffer,
#endif
#ifndef __CUDACC_RTC__
    const detail::curve::CurveTable *d_curve_table,
    const char* d_env_buffer,
#endif
    id_t *d_agent_output_nextID,
    const unsigned int popNo,
    const void *in_messagelist_metadata,
    const void *out_messagelist_metadata,
    curandState *d_rng,
    unsigned int *scanFlag_agentDeath,
    unsigned int *scanFlag_messageOutput,
    unsigned int *scanFlag_agentOutput);  // Can't put __global__ in a typedef

/**
 * Wrapper function for launching agent functions
 * Initialises FLAMEGPU_API instance
 * @param error_buffer Buffer used for detecting and reporting exception::DeviceErrors (flamegpu must be built with SEATBELTS enabled for this to be used)
 * @param d_curve_table Pointer to curve hash table in device memory
 * @param d_env_buffer Pointer to env buffer in device memory
 * @param d_agent_output_nextID If agent output is enabled, this points to a global memory src of the next suitable agent id, this will be atomically incremented at birth
 * @param popNo Total number of agents executing the function (number of threads launched)
 * @param in_messagelist_metadata Pointer to the MessageIn metadata struct, it is interpreted by MessageIn
 * @param out_messagelist_metadata Pointer to the MessageOut metadata struct, it is interpreted by MessageOut
 * @param d_rng Array of curand states for this kernel
 * @param scanFlag_agentDeath Scanflag array for agent death
 * @param scanFlag_messageOutput Scanflag array for optional message output
 * @param scanFlag_agentOutput Scanflag array for optional agent output
 * @tparam AgentFunction The modeller defined agent function (defined as FLAMEGPU_AGENT_FUNCTION in model code)
 * @tparam MessageIn Message handler for input messages (e.g. MessageNone, MessageBruteForce, MessageSpatial3D)
 * @tparam MessageOut Message handler for output messages (e.g. MessageNone, MessageBruteForce, MessageSpatial3D)
 */
template<typename AgentFunction, typename MessageIn, typename MessageOut>
__global__ void agent_function_wrapper(
#if !defined(SEATBELTS) || SEATBELTS
    exception::DeviceExceptionBuffer *error_buffer,
#endif
#ifndef __CUDACC_RTC__
    const detail::curve::CurveTable* __restrict__ d_curve_table,
    const char* d_env_buffer,
#endif
    id_t *d_agent_output_nextID,
    const unsigned int popNo,
    const void *in_messagelist_metadata,
    const void *out_messagelist_metadata,
    curandState *d_rng,
    unsigned int *scanFlag_agentDeath,
    unsigned int *scanFlag_messageOutput,
    unsigned int *scanFlag_agentOutput) {
    // We place these at the start of shared memory, so we can locate it anywhere in device code without a reference
    using detail::sm;
    if (threadIdx.x == 0) {
#if !defined(SEATBELTS) || SEATBELTS
        sm()->device_exception = error_buffer;
#endif
#ifndef __CUDACC_RTC__
        sm()->env_buffer = d_env_buffer;
#endif
    }
#ifndef __CUDACC_RTC__
    for (int idx = threadIdx.x; idx < detail::curve::Curve::MAX_VARIABLES; idx += blockDim.x) {
        sm()->curve_variables[idx] = d_curve_table->variables[idx];
        sm()->curve_hashes[idx] = d_curve_table->hashes[idx];
#if !defined(SEATBELTS) || SEATBELTS
        sm()->curve_type_size[idx] = d_curve_table->type_size[idx];
        sm()->curve_elements[idx] = d_curve_table->elements[idx];
        sm()->curve_count[idx] = d_curve_table->count[idx];
#endif
    }
#endif

    #if defined(__CUDACC__)  // @todo - This should not be required. This template should only ever be processed by a CUDA compiler.
    // Sync the block after Thread 0 has written to shared.
    __syncthreads();
    #endif  // __CUDACC__
    // Must be terminated here, else AgentRandom has bounds issues inside DeviceAPI constructor
    if (DeviceAPI<MessageIn, MessageOut>::getThreadIndex() >= popNo)
        return;
    // create a new device FLAME_GPU instance
    DeviceAPI<MessageIn, MessageOut> api = DeviceAPI<MessageIn, MessageOut>(
        d_agent_output_nextID,
        d_rng,
        scanFlag_agentOutput,
        MessageIn::In(in_messagelist_metadata),
        MessageOut::Out(out_messagelist_metadata, scanFlag_messageOutput));

    // call the user specified device function
    AGENT_STATUS flag = AgentFunction()(&api);
    if (scanFlag_agentDeath) {
        // (scan flags will not be processed unless agent death has been requested in model definition)
        scanFlag_agentDeath[DeviceAPI<MessageIn, MessageOut>::getThreadIndex()] = flag;
#if !defined(SEATBELTS) || SEATBELTS
    } else if (flag == DEAD) {
        DTHROW("Agent death must be enabled per agent function when defining the model.\n");
#endif
    }
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_CUH_
