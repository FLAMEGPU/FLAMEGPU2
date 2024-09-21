#include "flamegpu/io/JSONRunPlanReader.h"

#include <fstream>
#include <stack>

#include <rapidjson/stream.h>
#include <rapidjson/reader.h>
#include <rapidjson/error/en.h>

#include "flamegpu/model/ModelDescription.h"

namespace flamegpu {
namespace io {
class JSONRunPlanReader_impl : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, JSONRunPlanReader_impl> {
    enum Mode { Root, Plan, Core, Properties, PropertyArray, Nop };
    std::stack<Mode> mode;
    std::string lastKey;
    /**
     * Tracks current position reading environment property arrays
     */
    unsigned int current_array_index;
    std::string filename;
    RunPlanVector &rpv;

 public:
    JSONRunPlanReader_impl(const std::string& _filename, RunPlanVector& _rpv)
        : filename(_filename)
        , rpv(_rpv) { }
    template<typename T>
    bool processValue(const T val) {
        Mode isArray = Nop;
        if (mode.top() == PropertyArray) {
            isArray = mode.top();
            mode.pop();
        }
        if (mode.top() == Properties) {
            const auto it = rpv.environment->find(lastKey);
            if (it == rpv.environment->end()) {
                THROW exception::RapidJSONError("Input file contains unrecognised environment property '%s',"
                    "in JSONRunPlanReader::load()\n", lastKey.c_str());
            }
            if (current_array_index >= it->second.data.elements) {
                THROW exception::RapidJSONError("Input file contains environment property '%s' with %u elements expected %u,"
                    "in JSONRunPlanReader::load()\n", lastKey.c_str(), current_array_index, it->second.data.elements);
            }
            // Retrieve the linked any and replace the value
            const auto rp = rpv.end();
            const std::type_index val_type = it->second.data.type;
            if (it->second.data.elements ==0) {
                // Properties don't exist by default, so must be created
                if (val_type == std::type_index(typeid(float))) {
                    rp->setProperty(lastKey, static_cast<float>(val));
                } else if (val_type == std::type_index(typeid(double))) {
                    rp->setProperty(lastKey, static_cast<double>(val));
                } else if (val_type == std::type_index(typeid(int64_t))) {
                    rp->setProperty(lastKey, static_cast<int64_t>(val));
                } else if (val_type == std::type_index(typeid(uint64_t))) {
                    rp->setProperty(lastKey, static_cast<uint64_t>(val));
                } else if (val_type == std::type_index(typeid(int32_t))) {
                    rp->setProperty(lastKey, static_cast<int32_t>(val));
                } else if (val_type == std::type_index(typeid(uint32_t))) {
                    rp->setProperty(lastKey, static_cast<uint32_t>(val));
                } else if (val_type == std::type_index(typeid(int16_t))) {
                    rp->setProperty(lastKey, static_cast<int16_t>(val));
                } else if (val_type == std::type_index(typeid(uint16_t))) {
                    rp->setProperty(lastKey, static_cast<uint16_t>(val));
                } else if (val_type == std::type_index(typeid(int8_t))) {
                    rp->setProperty(lastKey, static_cast<int8_t>(val));
                } else if (val_type == std::type_index(typeid(uint8_t))) {
                    rp->setProperty(lastKey, static_cast<uint8_t>(val));
                } else {
                    THROW exception::RapidJSONError("RunPlan contains property '%s' of unsupported type '%s', "
                        "in JSONRunPlanReader::load()\n", lastKey.c_str(), val_type.name());
                }
            } else {
                // Arrays require more fiddly handling
                // Create the array if this is the first item
                if (current_array_index == 0) {
                    rp->property_overrides.emplace(lastKey, detail::Any(it->second));
                }
                // Copy in the specific value
                const auto prop_it = rp->property_overrides.at(lastKey);
                if (val_type == std::type_index(typeid(float))) {
                    static_cast<double*>(const_cast<void*>(prop_it.ptr))[current_array_index++] = static_cast<float>(val);
                } else if (val_type == std::type_index(typeid(double))) {
                    static_cast<double*>(const_cast<void*>(prop_it.ptr))[current_array_index++] = static_cast<double>(val);
                } else if (val_type == std::type_index(typeid(int64_t))) {
                    static_cast<int64_t*>(const_cast<void*>(prop_it.ptr))[current_array_index++] = static_cast<int64_t>(val);
                } else if (val_type == std::type_index(typeid(uint64_t))) {
                    static_cast<uint64_t*>(const_cast<void*>(prop_it.ptr))[current_array_index++] = static_cast<uint64_t>(val);
                } else if (val_type == std::type_index(typeid(int32_t))) {
                    static_cast<int32_t*>(const_cast<void*>(prop_it.ptr))[current_array_index++] = static_cast<int32_t>(val);
                } else if (val_type == std::type_index(typeid(uint32_t))) {
                    static_cast<uint32_t*>(const_cast<void*>(prop_it.ptr))[current_array_index++] = static_cast<uint32_t>(val);
                } else if (val_type == std::type_index(typeid(int16_t))) {
                    static_cast<int16_t*>(const_cast<void*>(prop_it.ptr))[current_array_index++] = static_cast<int16_t>(val);
                } else if (val_type == std::type_index(typeid(uint16_t))) {
                    static_cast<uint16_t*>(const_cast<void*>(prop_it.ptr))[current_array_index++] = static_cast<uint16_t>(val);
                } else if (val_type == std::type_index(typeid(int8_t))) {
                    static_cast<int8_t*>(const_cast<void*>(prop_it.ptr))[current_array_index++] = static_cast<int8_t>(val);
                } else if (val_type == std::type_index(typeid(uint8_t))) {
                    static_cast<uint8_t*>(const_cast<void*>(prop_it.ptr))[current_array_index++] = static_cast<uint8_t>(val);
                } else {
                    THROW exception::RapidJSONError("RunPlan contains property '%s' of unsupported type '%s', "
                        "in JSONRunPlanReader::load()\n", lastKey.c_str(), val_type.name());
                }
            }
        }  else {
            THROW exception::RapidJSONError("Unexpected value whilst parsing input file '%s'.\n", filename.c_str());
        }
        if (isArray == PropertyArray) {
            mode.push(isArray);
        }
        return true;
    }
    bool Null() { return true; }
    bool Bool(bool b) { return processValue<bool>(b); }
    bool Int(int i) { return processValue<int32_t>(i); }
    bool Uint(unsigned u) {
        if (mode.top() == Plan) {
            if (lastKey == "steps") {
                rpv.end()->setSteps(u);
                return true;
            }
            return false;
        }
        return processValue<uint32_t>(u);
     }
    bool Int64(int64_t i) { return processValue<int64_t>(i); }
    bool Uint64(uint64_t u) {
         if (mode.top() == Plan) {
             if (lastKey == "random_seed") {
                 rpv.end()->setRandomSimulationSeed(u);
                 return true;
             }
             return false;
         }
         return processValue<uint64_t>(u);
     }
    bool Double(double d) { return processValue<double>(d); }
    bool String(const char*s, rapidjson::SizeType, bool) {
        if (mode.top() == Plan) {
            if (lastKey == "output_subdirectory") {
                rpv.end()->setOutputSubdirectory(s);
                return true;
            }
        }
         // Properties never contain strings
        THROW exception::RapidJSONError("Unexpected string whilst parsing input file '%s'.\n", filename.c_str());
    }
    bool StartObject() {
        if (mode.empty()) {
            mode.push(Root);
        } else if (mode.top() == Plan) {
            if (lastKey == "RunPlanVector") {
                mode.push(Core);
            } else {
                THROW exception::RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        } else if (mode.top() == Core) {
            if (lastKey == "properties") {
                mode.push(Properties);
            } else {
                THROW exception::RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
            }
        } else if (mode.top() == PropertyArray) {
            rpv.push_back(RunPlan(rpv.environment, rpv.allow_0_steps));
        } else {
            THROW exception::RapidJSONError("Unexpected object start whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
    bool Key(const char* str, rapidjson::SizeType, bool) {
        lastKey = str;
        return true;
    }
    bool EndObject(rapidjson::SizeType) {
        mode.pop();
        return true;
    }
    bool StartArray() {
        if (current_array_index != 0) {
            THROW exception::RapidJSONError("Array start when current_array_index !=0, in file '%s'. This should never happen.\n", filename.c_str());
        }
        if (mode.top() == Plan && lastKey == "properties") {
            mode.push(Properties);
        } else if (mode.top() == Properties) {
            mode.push(PropertyArray);
        } else {
            THROW exception::RapidJSONError("Unexpected array start whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
    bool EndArray(rapidjson::SizeType) {
        if (mode.top() == PropertyArray) {
            mode.pop();
            if (mode.top() == Properties) {
                // Confirm env array had correct number of elements
                const auto &prop = rpv.environment->at(lastKey);
                if (current_array_index != prop.data.elements) {
                    THROW exception::RapidJSONError("Input file contains property '%s' with %u elements expected %u,"
                        "in JSONRunPlanReader::load()\n", lastKey.c_str(), current_array_index, prop.data.elements);
                }
            }
            current_array_index = 0;
        } else if (mode.top() == Properties) {
            mode.pop();
        } else {
            THROW exception::RapidJSONError("Unexpected array end whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
};
RunPlanVector JSONRunPlanReader::load(const std::string &input_filepath, const ModelDescription& model) {
    // Read the input file into a stringstream
    std::ifstream in(input_filepath, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        THROW exception::InvalidFilePath("Unable to open file '%s' for reading, in JSONRunPlanReader::load().", input_filepath.c_str());
    }
    const std::string filestring = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    rapidjson::StringStream filess = rapidjson::StringStream(filestring.c_str());
    in.close();
    // Attempt to parse the JSON into a RunPlanVector
    RunPlanVector result(model.model, 0);
    rapidjson::Reader reader;
    JSONRunPlanReader_impl handler(input_filepath, result);
    rapidjson::ParseResult pr = reader.Parse<rapidjson::kParseNanAndInfFlag, rapidjson::StringStream, flamegpu::io::JSONRunPlanReader_impl>(filess, handler);
    if (pr.Code() != rapidjson::ParseErrorCode::kParseErrorNone) {
        THROW exception::RapidJSONError("Whilst parsing input file '%s', RapidJSON returned error: %s\n", input_filepath.c_str(), rapidjson::GetParseError_En(pr.Code()));
    }
    // Return the result
    return result;
}
}
}
