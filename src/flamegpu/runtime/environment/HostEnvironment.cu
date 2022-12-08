#include "flamegpu/runtime/environment/HostEnvironment.cuh"

namespace flamegpu {

HostEnvironment::HostEnvironment(const unsigned int _instance_id, const std::shared_ptr<detail::EnvironmentManager> &env, detail::CUDAMacroEnvironment& _macro_env)
    : env_mgr(env)
    , macro_env(_macro_env)
    , instance_id(_instance_id) { }

}  // namespace flamegpu
