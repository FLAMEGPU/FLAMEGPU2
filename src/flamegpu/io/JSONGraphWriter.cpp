#include "flamegpu/io/JSONGraphWriter.h"

#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <fstream>
#include <cstdio>
#include <utility>
#include <string>
#include <memory>

#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/simulation/detail/CUDAEnvironmentDirectedGraphBuffers.cuh"

namespace flamegpu {
namespace io {

namespace {
template<typename T>
void writeAnyVertex(T &writer, const std::pair<std::string, Variable> &var, const unsigned int index, const std::shared_ptr<const detail::CUDAEnvironmentDirectedGraphBuffers>& directed_graph, cudaStream_t stream) {
    // Output value
    if (var.second.elements > 1) {
        writer->StartArray();
    }
    // Loop through elements, to construct array
    size_type foo = 0;
    for (unsigned int el = 0; el < var.second.elements; ++el) {
        if (var.second.type == std::type_index(typeid(float))) {
            writer->Double(directed_graph->getVertexPropertyBuffer<float>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(double))) {
            writer->Double(directed_graph->getVertexPropertyBuffer<double>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int64_t))) {
            writer->Int64(directed_graph->getVertexPropertyBuffer<int64_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(uint64_t))) {
            writer->Uint64(directed_graph->getVertexPropertyBuffer<uint64_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int32_t))) {
            writer->Int(directed_graph->getVertexPropertyBuffer<int32_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(uint32_t))) {
            writer->Uint(directed_graph->getVertexPropertyBuffer<uint32_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int16_t))) {
            writer->Int(directed_graph->getVertexPropertyBuffer<int16_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(uint16_t))) {
            writer->Uint(directed_graph->getVertexPropertyBuffer<uint16_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int8_t))) {
            writer->Int(static_cast<int32_t>(directed_graph->getVertexPropertyBuffer<int8_t>(var.first, foo, stream)[var.second.elements * index + el]));  // Char outputs weird if being used as an integer
        } else if (var.second.type == std::type_index(typeid(uint8_t))) {
            writer->Uint(static_cast<uint32_t>(directed_graph->getVertexPropertyBuffer<uint8_t>(var.first, foo, stream)[var.second.elements * index + el]));  // Char outputs weird if being used as an integer
        } else if (var.second.type == std::type_index(typeid(char))) {
            writer->Int(static_cast<int32_t>(directed_graph->getVertexPropertyBuffer<char>(var.first, foo, stream)[var.second.elements * index + el]));  // Char outputs weird if being used as an integer
        } else {
            THROW exception::RapidJSONError("Attempting to export value of unsupported type '%s', "
                "in JsonGraphWriter::writeAny()\n", var.second.type.name());
        }
    }
    if (var.second.elements > 1) {
        writer->EndArray();
    }
}
template<typename T>
void writeAnyEdge(T &writer, const std::pair<std::string, Variable> &var, const unsigned int index, const std::shared_ptr<const detail::CUDAEnvironmentDirectedGraphBuffers>& directed_graph, cudaStream_t stream) {
    // Output value
    if (var.second.elements > 1) {
        writer->StartArray();
    }
    // Loop through elements, to construct array
    size_type foo = 0;
    for (unsigned int el = 0; el < var.second.elements; ++el) {
        if (var.second.type == std::type_index(typeid(float))) {
            writer->Double(directed_graph->getEdgePropertyBuffer<float>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(double))) {
            writer->Double(directed_graph->getEdgePropertyBuffer<double>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int64_t))) {
            writer->Int64(directed_graph->getEdgePropertyBuffer<int64_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(uint64_t))) {
            writer->Uint64(directed_graph->getEdgePropertyBuffer<uint64_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int32_t))) {
            writer->Int(directed_graph->getEdgePropertyBuffer<int32_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(uint32_t))) {
            writer->Uint(directed_graph->getEdgePropertyBuffer<uint32_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int16_t))) {
            writer->Int(directed_graph->getEdgePropertyBuffer<int16_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(uint16_t))) {
            writer->Uint(directed_graph->getEdgePropertyBuffer<uint16_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int8_t))) {
            writer->Int(static_cast<int32_t>(directed_graph->getEdgePropertyBuffer<int8_t>(var.first, foo, stream)[var.second.elements * index + el]));  // Char outputs weird if being used as an integer
        } else if (var.second.type == std::type_index(typeid(uint8_t))) {
            writer->Uint(static_cast<uint32_t>(directed_graph->getEdgePropertyBuffer<uint8_t>(var.first, foo, stream)[var.second.elements * index + el]));  // Char outputs weird if being used as an integer
        } else if (var.second.type == std::type_index(typeid(char))) {
            writer->Int(static_cast<int32_t>(directed_graph->getEdgePropertyBuffer<char>(var.first, foo, stream)[var.second.elements * index + el]));  // Char outputs weird if being used as an integer
        } else {
            THROW exception::RapidJSONError("Attempting to export value of unsupported type '%s', "
                "in JsonGraphWriter::writeAny()\n", var.second.type.name());
        }
    }
    if (var.second.elements > 1) {
        writer->EndArray();
    }
}
template<typename T>
void toAdjancencyLike(T& writer, const std::shared_ptr<const detail::CUDAEnvironmentDirectedGraphBuffers>& directed_graph, cudaStream_t stream) {
    const EnvironmentDirectedGraphData &data = directed_graph->getDescription();
    // Begin json output object
    writer->StartObject();
    {
        // Vertices
        writer->Key("nodes");
        writer->StartArray();
        for (unsigned int i = 0; i < directed_graph->getVertexCount(); ++i) {
            writer->StartObject();
            // Reserved members first
            writer->Key("id");
            size_type foo = 0;
            writer->String(std::to_string(directed_graph->getVertexPropertyBuffer<id_t>(ID_VARIABLE_NAME, foo, stream)[i]).c_str());
            // Custom members after
            for (const auto &j : data.vertexProperties) {
                if (j.first[0] != '_') {
                    writer->Key(j.first.c_str());
                    writeAnyVertex(writer, j, i, directed_graph, stream);
                    if (i == 0 && j.first == "id" && j.first != ID_VARIABLE_NAME) {
                        fprintf(stderr, "Warning: Graph vertices contain both 'id' and '%s' properties, export may be invalid.\n", ID_VARIABLE_NAME);
                    }
                }
            }
            writer->EndObject();
        }
        writer->EndArray();
        // Edges
        writer->Key("links");
        writer->StartArray();
        for (unsigned int i = 0; i < directed_graph->getEdgeCount(); ++i) {
            writer->StartObject();
            // Reserved members first
            size_type foo = 0;
            const id_t *src_dest_buffer = directed_graph->getEdgePropertyBuffer<id_t>(GRAPH_SOURCE_DEST_VARIABLE_NAME, foo, stream);
            writer->Key("source");
            writer->String(std::to_string(src_dest_buffer[i * 2 + 1]).c_str());
            writer->Key("target");
            writer->String(std::to_string(src_dest_buffer[i * 2 + 0]).c_str());
            // Custom members after
            for (const auto& j : data.edgeProperties) {
                if (j.first[0] != '_') {
                    writer->Key(j.first.c_str());
                    writeAnyEdge(writer, j, i, directed_graph, stream);
                    if (i == 0 && j.first == "source") {
                        fprintf(stderr, "Warning: Graph edges contain both 'source' and '%s' properties, export may be invalid.\n", GRAPH_SOURCE_DEST_VARIABLE_NAME);
                    } else if (i == 0 && j.first == "target") {
                        fprintf(stderr, "Warning: Graph edges contain both 'target' and '%s' properties, export may be invalid.\n", GRAPH_SOURCE_DEST_VARIABLE_NAME);
                    }
                }
            }
            writer->EndObject();
        }
        writer->EndArray();
    }
    // End Json file
    writer->EndObject();
}
}  // namespace

void JSONGraphWriter::saveAdjacencyLike(const std::string& filepath,
    const std::shared_ptr<const detail::CUDAEnvironmentDirectedGraphBuffers>& directed_graph, cudaStream_t stream, bool pretty_print) {
    // Init writer
    rapidjson::StringBuffer s;
    if (pretty_print) {
        // rapidjson::Writer doesn't have virtual methods, so can't pass rapidjson::PrettyWriter around as ptr to rapidjson::writer
        auto writer = new rapidjson::PrettyWriter<rapidjson::StringBuffer, rapidjson::UTF8<>, rapidjson::UTF8<>, rapidjson::CrtAllocator, rapidjson::kWriteNanAndInfFlag>(s);
        writer->SetIndent('\t', 1);
        toAdjancencyLike(writer, directed_graph, stream);
        delete writer;
    } else {
        auto* writer = new rapidjson::Writer<rapidjson::StringBuffer, rapidjson::UTF8<>, rapidjson::UTF8<>, rapidjson::CrtAllocator, rapidjson::kWriteNanAndInfFlag>(s);
        toAdjancencyLike(writer, directed_graph, stream);
        delete writer;
    }
    // Perform output
    std::ofstream out(filepath, std::ofstream::trunc);
    if (!out.is_open()) {
        THROW exception::InvalidFilePath("Unable to open file '%s' for writing\n", filepath.c_str());
    }

    out << s.GetString();
    out << "\n";
    out.close();
}

}  // namespace io
}  // namespace flamegpu
