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
    static StateReader *createReader(const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model_state, const char *input) {
        std::string extension = getFileExt(input);

        if (extension == "xml") {
            return new xmlReader(model_state, input);
        }
        /*
        if (extension == "bin") {
            return new xmlReader(model_state, input);
        }
        */
        THROW UnsupportedFileType("File '%s' is not a type which can be read "
            "by ReaderFactory::createReader().\n",
            input);
    }
};

class WriterFactory {
 public:
     static StateWriter *createWriter(const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model_state, const unsigned int &iterations, const char *output_file) {
         return new xmlWriter(model_state, iterations, output_file);
     }
};

#endif  // INCLUDE_FLAMEGPU_IO_FACTORY_H_
