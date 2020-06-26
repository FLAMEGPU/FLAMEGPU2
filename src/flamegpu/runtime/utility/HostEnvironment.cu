#include "flamegpu/runtime/utility/HostEnvironment.cuh"

HostEnvironment::HostEnvironment(const unsigned int &_instance_id)
    : env_mgr(EnvironmentManager::getInstance())
    , instance_id(_instance_id) { }
