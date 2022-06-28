#include "flamegpu/runtime/environment/HostEnvironment.cuh"

namespace flamegpu {

HostEnvironment::HostEnvironment(const unsigned int _instance_id, const std::shared_ptr<detail::EnvironmentManager> &env, detail::CUDAMacroEnvironment& _macro_env,
  CUDADirectedGraphMap& _directed_graph_map, detail::CUDAScatter& _scatter, const unsigned int _streamID, const cudaStream_t _stream)
    : env_mgr(env)
    , macro_env(_macro_env)
    , directed_graph_map(_directed_graph_map)
    , instance_id(_instance_id)
    , scatter(_scatter)
    , streamID(_streamID)
    , stream(_stream) { }

HostEnvironmentDirectedGraph HostEnvironment::getDirectedGraph(const std::string& name) const {
    const auto rt = directed_graph_map.find(name);
    if (rt != directed_graph_map.end())
        return HostEnvironmentDirectedGraph(rt->second, scatter, streamID, stream);
    THROW exception::InvalidGraphName("Directed Graph with name '%s' was not found, "
        "in HostEnvironment::getDirectedGraph()",
        name.c_str());
}

}  // namespace flamegpu
