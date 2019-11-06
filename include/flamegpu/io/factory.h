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

#include <string>
#include "flamegpu/io/statereader.h"
#include "flamegpu/io/statewriter.h"
#include "flamegpu/io/xmlReader.h"
#include "flamegpu/io/xmlWriter.h"

//  move later
std::string getFileExt(const std::string& s) {
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
    static StateReader *createReader(const ModelDescription &model, const char *input) {
        std::string extension = getFileExt(input);

        if (extension == "xml") {
            return new xmlReader(model, input);
        }
        /*
        if (extension == "bin") {
            return new xmlReader(model, input);
        }
        */
        return nullptr;
    }
};

class WriterFactory {
 public:
    static StateWriter *createWriter(const ModelDescription &model, const char *input) {
        return new xmlWriter(model, input);
    }
};

#endif  // INCLUDE_FLAMEGPU_IO_FACTORY_H_
