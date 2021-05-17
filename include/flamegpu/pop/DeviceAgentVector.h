#ifndef INCLUDE_FLAMEGPU_POP_DEVICEAGENTVECTOR_H_
#define INCLUDE_FLAMEGPU_POP_DEVICEAGENTVECTOR_H_

#include "flamegpu/pop/AgentVector.h"
#include "flamegpu/pop/DeviceAgentVector_impl.h"

namespace flamegpu {

/**
 * This acts as a reference to DeviceAgentVector_impl
 * That class cannot be copied or assigned so it is accessed via a reference wrapper
 *
 * @see DeviceAgentVector_impl
 */
typedef DeviceAgentVector_impl& DeviceAgentVector;

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_POP_DEVICEAGENTVECTOR_H_
