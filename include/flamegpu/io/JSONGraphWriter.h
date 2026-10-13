#ifndef INCLUDE_FLAMEGPU_IO_JSONGRAPHWRITER_H_
#define INCLUDE_FLAMEGPU_IO_JSONGRAPHWRITER_H_

#include <memory>
#include <string>

#include "flamegpu/detail/cuda.cuh"

namespace flamegpu {
namespace detail {
class CUDAEnvironmentDirectedGraphBuffers;
}  // namespace detail
namespace io {

class JSONGraphWriter {
 public:
    /**
     * Exports the provided graph in the json "adjacency like" format, supported by NetworkX/d3.js
     *
     * @param filepath The path to save the graph to
     * @param directed_graph The graph buffers to export
     * @param stream CUDA stream (required by directed_graph for synchronising device buffers)
     * @param pretty_print Whether JSON should be human readable (vs minified)
     *
     * @throws exception::InvalidFilePath If the file cannot be opened for writing
     * @throws exception::RapidJSONError If conversion to JSON fails for any reason
     */
     static void saveAdjacencyLike(const std::string &filepath, const std::shared_ptr<const detail::CUDAEnvironmentDirectedGraphBuffers> &directed_graph, cudaStream_t stream, bool pretty_print = true);
};

}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_JSONGRAPHWRITER_H_
