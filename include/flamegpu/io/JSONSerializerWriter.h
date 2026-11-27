#ifndef INCLUDE_FLAMEGPU_IO_JSONSERIALIZERWRITER_H_
#define INCLUDE_FLAMEGPU_IO_JSONSERIALIZERWRITER_H_

#include <memory>
#include <string>
#include <fstream>

#include <nlohmann/json.hpp>

#include "flamegpu/model/ModelDescription.h"


namespace flamegpu {
namespace io {
/**
 * JSON format SerializerWriter
 */
class JSONSerializerWriter {
 public:
    /**
     * Constructs a writer capable of writing (serializing) model object to an JSON file
	 * @param outPath The file path to serialize the model to (must end in '.json')
     */
    JSONSerializerWriter(const std::string &outPath);

	/**
	 * Write the model file to serialized json
	 @param model A model description object which will be serialized
	 */
    void writeModel(ModelDescription model);

 private:
    std::string out_path;
    bool prettyPrint;
    bool truncateFile;
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_JSONSERIALIZERWRITER_H_
