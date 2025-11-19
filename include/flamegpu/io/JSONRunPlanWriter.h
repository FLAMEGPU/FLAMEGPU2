#ifndef INCLUDE_FLAMEGPU_IO_JSONRUNPLANWRITER_H_
#define INCLUDE_FLAMEGPU_IO_JSONRUNPLANWRITER_H_

#include <memory>
#include <string>

#include "flamegpu/simulation/RunPlanVector.h"

namespace flamegpu {
namespace io {
/**
 * JSON format writer of RunPlanVector
 */
class JSONRunPlanWriter {
    /**
     * Utility method for writing out the outer structure of a RunPlanVector
     * @param writer An initialised RapidJSON writer.
     * @param rpv RunPlanVector to be written
     * @tparam T Should be rapidjson::Writer or rapidjson::PrettyWriter (one does not inherit the other)
     */
    template <typename T>
    static void writeCommon(std::unique_ptr<T> &writer, const RunPlanVector &rpv);
    /**
     * Utility method for writing out a single RunPlan
     * @param writer An initialised RapidJSON writer.
     * @param rp RunPlan to be writer
     * @tparam T Should be rapidjson::Writer or rapidjson::PrettyWriter (one does not inherit the other)
     */
    template <typename T>
    static void writeRunPlan(std::unique_ptr<T> &writer, const RunPlan &rp);

 public:
    /**
    * Exports the provided RunPlanVector in JSON format to the specified output_filepath
    * @param rpv The RunPlanVector to be exported
    * @param output_filepath Location on disk to export the file
    * @param pretty Whether the exported JSON is "prettified" (true) or "minified" (false), defaults true.
    */
    static void save(const RunPlanVector &rpv, const std::string &output_filepath, bool pretty = true);
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_JSONRUNPLANWRITER_H_
