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

#include <memory>
#include <string>
#include <unordered_map>

#include "flamegpu/io/statereader.h"
#include "flamegpu/model/ModelDescription.h"

// Derived classes
class xmlReader : public StateReader {
 public:
    xmlReader(const std::string &model_name, const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model_state, const std::string &input_file);
    int parse();
};

#endif  // INCLUDE_FLAMEGPU_IO_XMLREADER_H_
