#include "flamegpu/runtime/utility/HostEnvironment.cuh"

HostEnvironment::HostEnvironment(const std::string &_model_name)
    : env_mgr(EnvironmentManager::getInstance())
    , model_name(_model_name) { }
