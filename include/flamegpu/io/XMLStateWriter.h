#ifndef INCLUDE_FLAMEGPU_IO_XMLSTATEWRITER_H_
#define INCLUDE_FLAMEGPU_IO_XMLSTATEWRITER_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "flamegpu/io/StateWriter.h"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/util/StringPair.h"

namespace flamegpu {
namespace io {
/**
 * XML format StateWriter
 */
class XMLStateWriter : public StateWriter {
 public:
    /**
     * Returns a writer capable of writing model state to an XML file
     * Agent data will be read from 'model_state'
     * @param model_name Name from the model description hierarchy of the model to be exported
     * @param env_manager Environment manager containing env property data for this sim instance
     * @param model_state Map of AgentVector to read the agent data from per agent, key should be agent name
     * @param iterations The value from the step counter at the time of export.
     * @param output_file Filename of the input file (This will be used to determine which reader to return)
     * @param sim_instance Instance of the Simulation object (This is used for setting/getting config)
     */
    XMLStateWriter(
        const std::string &model_name,
        const std::shared_ptr<detail::EnvironmentManager>& env_manager,
        const util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> &model_state,
        unsigned int iterations,
        const std::string &output_file,
        const Simulation *sim_instance);
    /**
     * Actually perform the writing to file
     * @param prettyPrint Whether to include indentation and line breaks to aide human reading
     * @return Always tinyxml2::XML_SUCCESS
     * @throws exception::TinyXMLError If export of the model state fails
     */
    int writeStates(bool prettyPrint) override;
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_XMLSTATEWRITER_H_
