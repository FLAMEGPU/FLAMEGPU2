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
#include <utility>
#include <algorithm>

#include "flamegpu/io/statereader.h"
#include "flamegpu/io/statewriter.h"
#include "flamegpu/io/xmlReader.h"
#include "flamegpu/io/xmlWriter.h"
#include "flamegpu/io/jsonReader.h"
#include "flamegpu/io/jsonWriter.h"
#include "flamegpu/io/jsonLogger.h"
#include "flamegpu/io/xmlLogger.h"
#include "flamegpu/util/StringPair.h"
#include "flamegpu/model/EnvironmentDescription.h"

class AgentVector;

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
     * @param env_desc Environment description for validating property data on load
     * @param env_init Dictionary of loaded values map:<{name, index}, value>
     * @param model_state Map of AgentVector to load the agent data into per agent, key should be agent name
     * @param input Filename of the input file (This will be used to determine which reader to return)
     * @param sim_instance Instance of the Simulation object (This is used for setting/getting config)
     * @throws UnsupportedFileType If the file extension does not match an appropriate reader
     */
    static StateReader* createReader(
        const std::string& model_name,
        const std::unordered_map<std::string, EnvironmentDescription::PropData>& env_desc,
        std::unordered_map<std::pair<std::string, unsigned int>, Any>& env_init,
        StringPairUnorderedMap<std::shared_ptr<AgentVector>>& model_state,
        const std::string& input,
        Simulation* sim_instance) {
        const std::string extension = getFileExt(input);

        if (extension == "xml") {
            return new xmlReader(model_name, env_desc, env_init, model_state, input, sim_instance);
        } else if (extension == "json") {
            return new jsonReader(model_name, env_desc, env_init, model_state, input, sim_instance);
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
     * @param model_state Map of AgentVector to read the agent data from per agent, key should be agent name
     * @param iterations The value from the step counter at the time of export.
     * @param output_file Filename of the input file (This will be used to determine which reader to return)
     * @param sim_instance Instance of the Simulation object (This is used for setting/getting config)
     * @throws UnsupportedFileType If the file extension does not match an appropriate reader
     */
    static StateWriter* createWriter(
        const std::string& model_name,
        const unsigned int& sim_instance_id,
        const StringPairUnorderedMap<std::shared_ptr<AgentVector>>& model_state,
        const unsigned int& iterations,
        const std::string& output_file,
        const Simulation* sim_instance) {
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
    /**
     * Return a clean file extension from the provided string
     * If the file extension is not supported empty string is returned instead
     */
    static std::string detectSupportedFileExt(const std::string &user_file_ext) {
        std::string rtn = user_file_ext;
        // Move entire string to lower case
        std::transform(rtn.begin(), rtn.end(), rtn.begin(), [](unsigned char c) { return std::use_facet< std::ctype<char>>(std::locale()).tolower(c); });
        // Strip first character if it is '.'
        if (rtn[0] == '.')
          rtn = rtn.substr(1);
        // Compare against supported formats
        if (rtn == "xml" ||
            rtn == "json") {
            return rtn;
        }
        return "";
    }
    /**
     * @param output_path File for the log to be output to, this will be used to determine the logger type
     */
    static std::unique_ptr<Logger> createLogger(const std::string &output_path, bool prettyPrint, bool truncateFile = true) {
        const std::string extension = getFileExt(output_path);

        if (extension == "xml") {
            return std::make_unique<xmlLogger>(output_path, prettyPrint, truncateFile);
        } else if (extension == "json") {
            return std::make_unique<jsonLogger>(output_path, prettyPrint, truncateFile);
        }
        THROW UnsupportedFileType("File '%s' is not a type which can be written "
            "by WriterFactory::createLogger().",
            output_path.c_str());
    }
};

#endif  // INCLUDE_FLAMEGPU_IO_FACTORY_H_
