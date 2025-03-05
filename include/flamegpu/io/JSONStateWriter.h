#ifndef INCLUDE_FLAMEGPU_IO_JSONSTATEWRITER_H_
#define INCLUDE_FLAMEGPU_IO_JSONSTATEWRITER_H_

#include <memory>
#include <string>
#include <fstream>

#include <nlohmann/json.hpp>


#include "flamegpu/io/StateWriter.h"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/util/StringPair.h"


namespace flamegpu {
namespace detail {
class CUDAMacroEnvironment;
}
namespace io {
/**
 * JSON format StateWriter
 */
class JSONStateWriter : public StateWriter {
 public:
    /**
     * Constructs a writer capable of writing model state to an JSON file
     */
    JSONStateWriter();
    /**
    * Returns true if beginWrite() has been called and writing is active
    */
    bool isWriting() override { return outStream.is_open(); }
    /**
    * Starts writing to the specified file in the specified mode
    * @param output_file Filename of the input file (This will be used to determine which reader to return)
    * @param pretty_print Whether the output file should be "pretty" or minified.
    * @throws Throws exception if beginWrite() is called a second time before endWrite() has been called
    * @see endWrite()
    */
    void beginWrite(const std::string &output_file, bool pretty_print) override;
    /**
    * Saves the current file and ends the writing state
    * @see beginWrite(const std::string &, bool)
    * @throws Throws exception if file IO fails
    * @throws Throws exception if beginWrite() has not been called so writing is not active
    */
    void endWrite() override;

    void writeConfig(const Simulation *sim_instance) override;
    void writeStats(unsigned int iterations) override;
    void writeEnvironment(const std::shared_ptr<const detail::EnvironmentManager>& env_manager) override;
    void writeMacroEnvironment(const std::shared_ptr<const detail::CUDAMacroEnvironment>& macro_env, std::initializer_list<std::string> filter = {}) override;
    void writeAgents(const util::StringPairUnorderedMap<std::shared_ptr<const AgentVector>>& agents_map) override;

 private:
    bool config_written = false;
    bool stats_written = false;
    bool environment_written = false;
    bool macro_environment_written = false;
    bool agents_written = false;
    // Dirty workaround for PrettyWriter overloads not being virtual
    bool newline_purge_required = false;
    std::string outputPath;
    std::fstream outStream;

    nlohmann::ordered_json j;
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_JSONSTATEWRITER_H_
