#ifndef INCLUDE_FLAMEGPU_IO_FACTORY_H_
#define INCLUDE_FLAMEGPU_IO_FACTORY_H_

/**
 * @file
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#include <memory>
#include <string>
#include <unordered_map>

#include "flamegpu/io/statereader.h"
#include "flamegpu/io/statewriter.h"
#include "flamegpu/io/xmlReader.h"
#include "flamegpu/io/xmlWriter.h"
#include "flamegpu/io/jsonReader.h"
#include "flamegpu/io/jsonWriter.h"

//  move later
inline std::string getFileExt(const std::string& s) {
    // Find the last position of '.' in given string
    size_t i = s.rfind('.', s.length());
    if (i != std::string::npos) {
        return(s.substr(i + 1, s.length() - i));
    }
    // In case of no extension return empty string
    return("");
}

/**
* Concrete factory creates concrete products, but
* returns them as abstract.
*/
class ReaderFactory {
 public:
    /**
     * Returns a reader capable of reading 'input'
     * Environment properties will be read into the Simulation instance pointed to by 'sim_instance_id'
     * Agent data will be read into 'model_state'
     * @param model_name Name from the model description hierarchy of the model to be loaded
     * @param sim_instance_id Instance is from the Simulation instance to load the environment properties into
     * @param model_state Map of AgentPopulation to load the agent data into per agent, key should be agent name
     * @param input Filename of the input file (This will be used to determine which reader to return)
     * @param sim_instance Instance of the Simulation object (This is used for setting/getting config)
     * @throws UnsupportedFileType If the file extension does not match an appropriate reader
     */
    static StateReader *createReader(
        const std::string &model_name,
        const unsigned int &sim_instance_id,
        const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model_state,
        const std::string &input,
        Simulation *sim_instance) {
        const std::string extension = getFileExt(input);

        if (extension == "xml") {
            return new xmlReader(model_name, sim_instance_id, model_state, input, sim_instance);
        } else if (extension == "json") {
            return new jsonReader(model_name, sim_instance_id, model_state, input, sim_instance);
        }
        /*
        if (extension == "bin") {
            return new xmlReader(model_state, input);
        }
        */
        THROW UnsupportedFileType("File '%s' is not a type which can be read "
            "by ReaderFactory::createReader().",
            input.c_str());
    }
};

class WriterFactory {
 public:
    /**
     * Returns a writer capable of writing model state to 'output_file'
     * Environment properties from the Simulation instance pointed to by 'sim_instance_id' will be used
     * Agent data will be read from 'model_state'
     * @param model_name Name from the model description hierarchy of the model to be exported
     * @param sim_instance_id Instance is from the Simulation instance to export the environment properties from
     * @param model_state Map of AgentPopulation to read the agent data from per agent, key should be agent name
     * @param iterations The value from the step counter at the time of export.
     * @param output_file Filename of the input file (This will be used to determine which reader to return)
     * @param sim_instance Instance of the Simulation object (This is used for setting/getting config)
     * @throws UnsupportedFileType If the file extension does not match an appropriate reader
     */
    static StateWriter *createWriter(
        const std::string &model_name,
        const unsigned int &sim_instance_id,
        const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model_state,
        const unsigned int &iterations,
        const std::string &output_file,
        const Simulation *sim_instance) {
        const std::string extension = getFileExt(output_file);

        if (extension == "xml") {
            return new xmlWriter(model_name, sim_instance_id, model_state, iterations, output_file, sim_instance);
        } else if (extension == "json") {
            return new jsonWriter(model_name, sim_instance_id, model_state, iterations, output_file, sim_instance);
        }
        THROW UnsupportedFileType("File '%s' is not a type which can be written "
            "by WriterFactory::createWriter().",
            output_file.c_str());
     }
};

#endif  // INCLUDE_FLAMEGPU_IO_FACTORY_H_
