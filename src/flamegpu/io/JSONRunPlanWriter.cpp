#include "flamegpu/io/JSONRunPlanWriter.h"

#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <fstream>

namespace flamegpu {
namespace io {
void JSONRunPlanWriter::save(const RunPlanVector& rpv, const std::string& output_filepath, const bool pretty_print) {
    // Init writer
    auto buffer = rapidjson::StringBuffer();
    std::unique_ptr<GenericJSONWriter> writer;
    if (pretty_print) {
        auto t_writer = std::make_unique<rapidjson::PrettyWriter<rapidjson::StringBuffer, rapidjson::UTF8<>, rapidjson::UTF8<>, rapidjson::CrtAllocator, rapidjson::kWriteNanAndInfFlag>>(buffer);
        t_writer->SetIndent('\t', 1);
        writer = std::move(t_writer);
    } else {
        writer = std::make_unique<rapidjson::Writer<rapidjson::StringBuffer, rapidjson::UTF8<>, rapidjson::UTF8<>, rapidjson::CrtAllocator, rapidjson::kWriteNanAndInfFlag>>(buffer);
    }
    writer->StartObject();
    writer->Key("RunPlanVector");
    writer->StartArray();
    // Write out RunPlan records
    for (const auto &rp : rpv) {
        writeRunPlan(writer, rp);
    }
    // Finalise and dump to file
    writer->EndArray();
    writer->EndObject();
    std::ofstream out(output_filepath, std::ofstream::trunc);
    if (!out.is_open()) {
        THROW exception::InvalidFilePath("Unable to open '%s' for writing, in JSONRunPlanWriter::save().", output_filepath.c_str());
    }
    out << buffer.GetString();
    out.close();
    // Cleanup (redundant in a static method)
    writer.reset();
    buffer.Clear();
}
/**
 * Utility method for writing out a single RunPlan
 * @param writer An initialised RapidJSON writer.
 * @param rp RunPlan to be writer
 */
void JSONRunPlanWriter::writeRunPlan(std::unique_ptr<GenericJSONWriter> &writer, const RunPlan &rp) {
    // Core
    writer->Key("random_seed");
    writer->Uint64(rp.random_seed);
    writer->Key("steps");
    writer->Uint(rp.steps);
    writer->Key("output_subdirectory");
    writer->String(rp.output_subdirectory.c_str());
    // This value is internal and is based on whether the corresponding ModelDescription has an exit condition
    // writer->Key("allow_0_steps");
    // writer->Bool(rp.allow_0_steps);
    // Properties
    writer->Key("properties");
    writer->StartObject();
    for (const auto &[name, p] : rp.property_overrides) {
        writer->Key(name.c_str());
        // Locate the environment property's metadata
        const auto p_meta = rp.environment->at(name);
        // Output value
        if (p_meta.data.elements > 1) {
            // Value is an array
            writer->StartArray();
        }
        // Loop through elements, to construct array
        for (unsigned int el = 0; el < p_meta.data.elements; ++el) {
            if (p_meta.data.type == std::type_index(typeid(float))) {
                writer->Double(*(reinterpret_cast<const float*>(p_meta.data.ptr) + el));
            } else if (p_meta.data.type == std::type_index(typeid(double))) {
                writer->Double(*(reinterpret_cast<const double*>(p_meta.data.ptr) + el));
            } else if (p_meta.data.type == std::type_index(typeid(int64_t))) {
                writer->Int64(*(reinterpret_cast<const int64_t*>(p_meta.data.ptr) + el));
            } else if (p_meta.data.type == std::type_index(typeid(uint64_t))) {
                writer->Uint64(*(reinterpret_cast<const uint64_t*>(p_meta.data.ptr) + el));
            } else if (p_meta.data.type == std::type_index(typeid(int32_t))) {
                writer->Int(*(reinterpret_cast<const int32_t*>(p_meta.data.ptr) + el));
            } else if (p_meta.data.type == std::type_index(typeid(uint32_t))) {
                writer->Uint(*(reinterpret_cast<const uint32_t*>(p_meta.data.ptr) + el));
            } else if (p_meta.data.type == std::type_index(typeid(int16_t))) {
                writer->Int(*(reinterpret_cast<const int16_t*>(p_meta.data.ptr) + el));
            } else if (p_meta.data.type == std::type_index(typeid(uint16_t))) {
                writer->Uint(*(reinterpret_cast<const uint16_t*>(p_meta.data.ptr) + el));
            } else if (p_meta.data.type == std::type_index(typeid(int8_t))) {
                writer->Int(static_cast<int32_t>(*(reinterpret_cast<const int8_t*>(p_meta.data.ptr) + el)));  // Char outputs weird if being used as an integer
            } else if (p_meta.data.type == std::type_index(typeid(uint8_t))) {
                writer->Uint(static_cast<uint32_t>(*(reinterpret_cast<const uint8_t*>(p_meta.data.ptr) + el)));  // Char outputs weird if being used as an integer
            } else {
                THROW exception::RapidJSONError("RunPlan contains environment property '%s' of unsupported type '%s', "
                    "in JSONRunPlanWriter::writeRunPlan()\n", name.c_str(), p_meta.data.type.name());
            }
        }
        if (p_meta.data.elements > 1) {
            // Value is an array
            writer->EndArray();
        }
    }
    writer->EndObject();
}
}  // namespace io
}  // namespace flamegpu
