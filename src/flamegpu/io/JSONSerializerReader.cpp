#include "flamegpu/io/JSONSerializerReader.h"

#include <stack>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <cstdio>
#include <memory>

#include <nlohmann/json.hpp>

#include "flamegpu/exception/FLAMEGPUException.h"


namespace flamegpu {
namespace io {

namespace {

}  // namespace

ModelDescription JSONSerializerReader::parse(const std::string& filepath) {
	THROW exception::JSONError("Not implemented\n");
}

}  // namespace io
}  // namespace flamegpu
