#ifndef INCLUDE_FLAMEGPU_IO_JSONRUNPLANWRITER_H_
#define INCLUDE_FLAMEGPU_IO_JSONRUNPLANWRITER_H_

#include <rapidjson/writer.h>

#include <memory>
#include <string>


#include "flamegpu/simulation/RunPlanVector.h"

namespace flamegpu {
namespace io {
/**
 * JSON format writer of RunPlanVector
 */
class JSONRunPlanWriter {
    // Typedef for the writer used, as the full template specification is way too long
    typedef rapidjson::Writer<rapidjson::StringBuffer, rapidjson::UTF8<>, rapidjson::UTF8<>, rapidjson::CrtAllocator, rapidjson::kWriteNanAndInfFlag> GenericJSONWriter;
    /**
     * Utility method for writing out a single RunPlan
     * @param writer An initialised RapidJSON writer.
     * @param rp RunPlan to be writer
     */
    static void writeRunPlan(std::unique_ptr<GenericJSONWriter>& writer, const RunPlan& rp);

 public:
    /**
    * Exports the provided RunPlanVector in JSON format to the specified output_filepath
    * @param rpv The RunPlanVector to be exported
    * @param output_filepath Location on disk to export the file
    * @param pretty Whether the exported JSON is "prettified" or "minified"
    */
    static void save(const RunPlanVector &rpv, const std::string &output_filepath, bool pretty = true);
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_JSONRUNPLANWRITER_H_
