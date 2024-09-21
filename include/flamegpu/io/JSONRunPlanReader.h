#ifndef INCLUDE_FLAMEGPU_IO_JSONRUNPLANREADER_H_
#define INCLUDE_FLAMEGPU_IO_JSONRUNPLANREADER_H_

#include <string>

#include "flamegpu/simulation/RunPlanVector.h"

namespace flamegpu {
class ModelDescription;
namespace io {

/**
 * JSON format reader of RunPlanVector
 */
class JSONRunPlanReader {
 public:
    /**
    * Loads and returns the specified JSON file if contains a RunPlanVector
    * @param input_filepath Path on disk to read the file from
    * @param model The model used to initialise the RunPlanVector
    */
    static RunPlanVector load(const std::string &input_filepath, const ModelDescription& model);
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_JSONRUNPLANREADER_H_
