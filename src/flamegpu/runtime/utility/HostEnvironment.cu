#include "flamegpu/runtime/utility/HostEnvironment.cuh"

namespace flamegpu {

HostEnvironment::HostEnvironment(const unsigned int &_instance_id)
    : env_mgr(EnvironmentManager::getInstance())
    , instance_id(_instance_id) { }

}  // namespace flamegpu
