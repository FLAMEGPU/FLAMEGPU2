#include "flamegpu/io/JSONGraphWriter.h"

#include <fstream>
#include <cstdio>
#include <utility>
#include <string>
#include <memory>

#include <nlohmann/json.hpp>

#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/simulation/detail/CUDAEnvironmentDirectedGraphBuffers.cuh"

namespace flamegpu {
namespace io {

namespace {
void writeAnyVertex(nlohmann::ordered_json& j, const std::pair<std::string, Variable> &var, const unsigned int index, const std::shared_ptr<const detail::CUDAEnvironmentDirectedGraphBuffers>& directed_graph, cudaStream_t stream) {
    size_type foo = 0;
    // Output value
    if (var.second.elements == 1) {
        if (var.second.type == std::type_index(typeid(float))) {
            j = directed_graph->getVertexPropertyBuffer<float>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(double))) {
            j = directed_graph->getVertexPropertyBuffer<double>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(int64_t))) {
            j = directed_graph->getVertexPropertyBuffer<int64_t>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(uint64_t))) {
            j = directed_graph->getVertexPropertyBuffer<uint64_t>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(int32_t))) {
            j = directed_graph->getVertexPropertyBuffer<int32_t>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(uint32_t))) {
            j = directed_graph->getVertexPropertyBuffer<uint32_t>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(int16_t))) {
            j = directed_graph->getVertexPropertyBuffer<int16_t>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(uint16_t))) {
            j = directed_graph->getVertexPropertyBuffer<uint16_t>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(int8_t))) {
            j = static_cast<int32_t>(directed_graph->getVertexPropertyBuffer<int8_t>(var.first, foo, stream)[var.second.elements * index]);  // Char outputs weird if being used as an integer
        } else if (var.second.type == std::type_index(typeid(uint8_t))) {
            j = static_cast<uint32_t>(directed_graph->getVertexPropertyBuffer<uint8_t>(var.first, foo, stream)[var.second.elements * index]);  // Char outputs weird if being used as an integer
        } else if (var.second.type == std::type_index(typeid(char))) {
            j = static_cast<int32_t>(directed_graph->getVertexPropertyBuffer<char>(var.first, foo, stream)[var.second.elements * index]);  // Char outputs weird if being used as an integer
        } else {
            THROW exception::JSONError("Attempting to export value of unsupported type '%s', "
                "in JsonGraphWriter::writeAny()\n", var.second.type.name());
        }
        return;
    }
    // Loop through elements, to construct array
    j = {};
    for (unsigned int el = 0; el < var.second.elements; ++el) {
        if (var.second.type == std::type_index(typeid(float))) {
            j.push_back(directed_graph->getVertexPropertyBuffer<float>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(double))) {
            j.push_back(directed_graph->getVertexPropertyBuffer<double>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int64_t))) {
            j.push_back(directed_graph->getVertexPropertyBuffer<int64_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(uint64_t))) {
            j.push_back(directed_graph->getVertexPropertyBuffer<uint64_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int32_t))) {
            j.push_back(directed_graph->getVertexPropertyBuffer<int32_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(uint32_t))) {
            j.push_back(directed_graph->getVertexPropertyBuffer<uint32_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int16_t))) {
            j.push_back(directed_graph->getVertexPropertyBuffer<int16_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(uint16_t))) {
            j.push_back(directed_graph->getVertexPropertyBuffer<uint16_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int8_t))) {
            j.push_back(static_cast<int32_t>(directed_graph->getVertexPropertyBuffer<int8_t>(var.first, foo, stream)[var.second.elements * index + el]));  // Char outputs weird if being used as an integer
        } else if (var.second.type == std::type_index(typeid(uint8_t))) {
            j.push_back(static_cast<uint32_t>(directed_graph->getVertexPropertyBuffer<uint8_t>(var.first, foo, stream)[var.second.elements * index + el]));  // Char outputs weird if being used as an integer
        } else if (var.second.type == std::type_index(typeid(char))) {
            j.push_back(static_cast<int32_t>(directed_graph->getVertexPropertyBuffer<char>(var.first, foo, stream)[var.second.elements * index + el]));  // Char outputs weird if being used as an integer
        } else {
            THROW exception::JSONError("Attempting to export value of unsupported type '%s', "
                "in JsonGraphWriter::writeAny()\n", var.second.type.name());
        }
    }
}
void writeAnyEdge(nlohmann::ordered_json& j, const std::pair<std::string, Variable> &var, const unsigned int index, const std::shared_ptr<const detail::CUDAEnvironmentDirectedGraphBuffers>& directed_graph, cudaStream_t stream) {
    size_type foo = 0;
    // Output value
    if (var.second.elements == 1) {
        if (var.second.type == std::type_index(typeid(float))) {
            j = directed_graph->getEdgePropertyBuffer<float>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(double))) {
            j = directed_graph->getEdgePropertyBuffer<double>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(int64_t))) {
            j = directed_graph->getEdgePropertyBuffer<int64_t>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(uint64_t))) {
            j = directed_graph->getEdgePropertyBuffer<uint64_t>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(int32_t))) {
            j = directed_graph->getEdgePropertyBuffer<int32_t>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(uint32_t))) {
            j = directed_graph->getEdgePropertyBuffer<uint32_t>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(int16_t))) {
            j = directed_graph->getEdgePropertyBuffer<int16_t>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(uint16_t))) {
            j = directed_graph->getEdgePropertyBuffer<uint16_t>(var.first, foo, stream)[var.second.elements * index];
        } else if (var.second.type == std::type_index(typeid(int8_t))) {
            j = static_cast<int32_t>(directed_graph->getEdgePropertyBuffer<int8_t>(var.first, foo, stream)[var.second.elements * index]);  // Char outputs weird if being used as an integer
        } else if (var.second.type == std::type_index(typeid(uint8_t))) {
            j = static_cast<uint32_t>(directed_graph->getEdgePropertyBuffer<uint8_t>(var.first, foo, stream)[var.second.elements * index]);  // Char outputs weird if being used as an integer
        } else if (var.second.type == std::type_index(typeid(char))) {
            j = static_cast<int32_t>(directed_graph->getEdgePropertyBuffer<char>(var.first, foo, stream)[var.second.elements * index]);  // Char outputs weird if being used as an integer
        } else {
            THROW exception::JSONError("Attempting to export value of unsupported type '%s', "
                "in JsonGraphWriter::writeAny()\n", var.second.type.name());
        }
        return;
    }
    // Loop through elements, to construct array
    j = {};
    for (unsigned int el = 0; el < var.second.elements; ++el) {
        if (var.second.type == std::type_index(typeid(float))) {
            j.push_back(directed_graph->getEdgePropertyBuffer<float>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(double))) {
            j.push_back(directed_graph->getEdgePropertyBuffer<double>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int64_t))) {
            j.push_back(directed_graph->getEdgePropertyBuffer<int64_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(uint64_t))) {
            j.push_back(directed_graph->getEdgePropertyBuffer<uint64_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int32_t))) {
            j.push_back(directed_graph->getEdgePropertyBuffer<int32_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(uint32_t))) {
            j.push_back(directed_graph->getEdgePropertyBuffer<uint32_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int16_t))) {
            j.push_back(directed_graph->getEdgePropertyBuffer<int16_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(uint16_t))) {
            j.push_back(directed_graph->getEdgePropertyBuffer<uint16_t>(var.first, foo, stream)[var.second.elements * index + el]);
        } else if (var.second.type == std::type_index(typeid(int8_t))) {
            j.push_back(static_cast<int32_t>(directed_graph->getEdgePropertyBuffer<int8_t>(var.first, foo, stream)[var.second.elements * index + el]));  // Char outputs weird if being used as an integer
        } else if (var.second.type == std::type_index(typeid(uint8_t))) {
            j.push_back(static_cast<uint32_t>(directed_graph->getEdgePropertyBuffer<uint8_t>(var.first, foo, stream)[var.second.elements * index + el]));  // Char outputs weird if being used as an integer
        } else if (var.second.type == std::type_index(typeid(char))) {
            j.push_back(static_cast<int32_t>(directed_graph->getEdgePropertyBuffer<char>(var.first, foo, stream)[var.second.elements * index + el]));  // Char outputs weird if being used as an integer
        } else {
            THROW exception::JSONError("Attempting to export value of unsupported type '%s', "
                "in JsonGraphWriter::writeAny()\n", var.second.type.name());
        }
    }
}
void toAdjancencyLike(nlohmann::ordered_json &j, const std::shared_ptr<const detail::CUDAEnvironmentDirectedGraphBuffers>& directed_graph, cudaStream_t stream) {
    const EnvironmentDirectedGraphData &data = directed_graph->getDescription();
    // Vertices
    j["nodes"] = {};
    for (unsigned int i = 0; i < directed_graph->getVertexCount(); ++i) {
        nlohmann::ordered_json j_v;
        // Reserved members first
        size_type foo = 0;
        j_v["id"] = std::to_string(directed_graph->getVertexPropertyBuffer<id_t>(ID_VARIABLE_NAME, foo, stream)[i]);
        // Custom members after
        for (const auto &v : data.vertexProperties) {
            if (v.first[0] != '_') {
                writeAnyVertex(j_v[v.first], v, i, directed_graph, stream);
                if (i == 0 && v.first == "id" && v.first != ID_VARIABLE_NAME) {
                    fprintf(stderr, "Warning: Graph vertices contain both 'id' and '%s' properties, export may be invalid.\n", ID_VARIABLE_NAME);
                }
            }
        }
        j["nodes"].push_back(j_v);
    }
    // Edges
    j["links"] = {};
    for (unsigned int i = 0; i < directed_graph->getEdgeCount(); ++i) {
        nlohmann::ordered_json j_e;
        // Reserved members first
        size_type foo = 0;
        const id_t *src_dest_buffer = directed_graph->getEdgePropertyBuffer<id_t>(GRAPH_SOURCE_DEST_VARIABLE_NAME, foo, stream);
        j_e["source"] = std::to_string(src_dest_buffer[i * 2 + 1]);
        j_e["target"] = std::to_string(src_dest_buffer[i * 2 + 0]);
        // Custom members after
        for (const auto& e : data.edgeProperties) {
            if (e.first[0] != '_') {
                writeAnyEdge(j_e[e.first], e, i, directed_graph, stream);
                if (i == 0 && e.first == "source") {
                    fprintf(stderr, "Warning: Graph edges contain both 'source' and '%s' properties, export may be invalid.\n", GRAPH_SOURCE_DEST_VARIABLE_NAME);
                } else if (i == 0 && e.first == "target") {
                    fprintf(stderr, "Warning: Graph edges contain both 'target' and '%s' properties, export may be invalid.\n", GRAPH_SOURCE_DEST_VARIABLE_NAME);
                }
            }
        }
        j["links"].push_back(j_e);
    }
}
}  // namespace

void JSONGraphWriter::saveAdjacencyLike(const std::string& filepath,
    const std::shared_ptr<const detail::CUDAEnvironmentDirectedGraphBuffers>& directed_graph, cudaStream_t stream, bool pretty_print) {
    // Init writer
    nlohmann::ordered_json j;
    toAdjancencyLike(j, directed_graph, stream);
    // Perform output
    std::ofstream out(filepath, std::ios::trunc | std::ios::binary);
    if (!out.is_open()) {
        THROW exception::InvalidFilePath("Unable to open file '%s' for writing, in JSONGraphWriter::saveAdjacencyLike()\n", filepath.c_str());
    }
    if (pretty_print) {
        out << std::setw(4);
    }
    out << j;
    out.close();
}

}  // namespace io
}  // namespace flamegpu
