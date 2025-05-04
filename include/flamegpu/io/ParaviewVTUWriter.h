#ifndef INCLUDE_FLAMEGPU_IO_PARAVIEWVTUWRITER_H_
#define INCLUDE_FLAMEGPU_IO_PARAVIEWVTUWRITER_H_

#include <memory>
#include <string>

namespace flamegpu {
class AgentVector;
namespace io {

class ParaviewVTUWriter {
    std::string output_dir;

 public:
    explicit ParaviewVTUWriter(std::string _output_dir)
        : output_dir(std::move(_output_dir))
    { }
    void writeAgentState(const std::string &agent, const std::string &state, const std::shared_ptr<const AgentVector>& agents_map, unsigned int step);
};

}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_PARAVIEWVTUWRITER_H_
