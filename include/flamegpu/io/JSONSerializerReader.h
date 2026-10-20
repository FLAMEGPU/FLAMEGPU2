#ifndef INCLUDE_FLAMEGPU_IO_JSONSERIALIZERREADER_H_
#define INCLUDE_FLAMEGPU_IO_JSONSERIALIZERREADER_H_

#include <memory>
#include <string>
#include <fstream>

#include <nlohmann/json.hpp>

#include "flamegpu/model/ModelDescription.h"


namespace flamegpu {
namespace io {
/**
 * JSON format SerializerReader
 */
class JSONSerializerReader {
 public:
	/**
	 * Write the model file to serialized json
	 @param filePath The file path to a serialized FLAME GPU model (must end in '.json')
	 */
    static ModelDescription parse(const std::string &filePath);
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_JSONSERIALIZERREADER_H_
