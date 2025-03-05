#ifndef INCLUDE_FLAMEGPU_IO_JSONGRAPHREADER_H_
#define INCLUDE_FLAMEGPU_IO_JSONGRAPHREADER_H_

#include <memory>
#include <string>

#include "flamegpu/detail/cuda.cuh"

namespace flamegpu {
namespace detail {
class CUDAEnvironmentDirectedGraphBuffers;
}  // namespace detail
namespace io {

class JSONGraphReader {
 public:
    /**
     * Imports the provided graph from the json "adjacency like" format, supported by NetworkX/d3.js
     *
     * @param filepath The path to load the graph from
     * @param directed_graph The graph buffers to import into
     * @param stream CUDA stream (required by directed_graph for synchronising device buffers)
     *
     * @throws exception::InvalidFilePath If the file cannot be opened for reading
     * @throws exception::JSONError If JSON fails for any reason (e.g. structure does not match expectations)
     */
     static void loadAdjacencyLike(const std::string &filepath, const std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> &directed_graph, cudaStream_t stream);
};

}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_JSONGRAPHREADER_H_
