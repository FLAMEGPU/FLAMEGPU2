#ifndef INCLUDE_FLAMEGPU_IO_XMLREADER_H_
#define INCLUDE_FLAMEGPU_IO_XMLREADER_H_

/**
 * @file xmlreader.h
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#include <string>

#include  <flamegpu/io/statereader.h>
#include "flamegpu/model/ModelDescription.h"



// Derived classes
class xmlReader : public StateReader {
 public:
    xmlReader(const ModelDescription &model, const char* input);
    int parse();
};


#endif // INCLUDE_FLAMEGPU_IO_XMLREADER_H_
