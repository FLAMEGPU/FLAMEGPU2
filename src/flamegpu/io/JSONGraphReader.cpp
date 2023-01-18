#include "flamegpu/io/JSONGraphReader.h"

#include <rapidjson/stream.h>
#include <rapidjson/reader.h>
#include <rapidjson/error/en.h>
#include <stack>
#include <fstream>
#include <string>
#include <map>
#include <set>

#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/simulation/detail/CUDAEnvironmentDirectedGraphBuffers.cuh"

namespace flamegpu {
namespace io {

namespace {

/**
 * This is a trivial parser, it counts the number of vertices and edges
 * This allows the graph's buffers to be preallocated to the correct size
 */
class JSONAdjacencyGraphSizeReader : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, JSONAdjacencyGraphSizeReader>  {
    enum Mode{ Nop, Root, Nodes, Links };
    std::stack<Mode> mode;
    unsigned int vertex_count = 0;
    unsigned int edge_count = 0;
    std::string lastKey;

 public:
    unsigned int getVertexCount() const {
        return vertex_count;
    }
    unsigned int getEdgeCount() const {
        return edge_count;
    }

    bool Null() { return true; }
    bool Bool(bool b) { return true; }
    bool Int(int i) { return true; }
    bool Uint(unsigned u) { return true; }
    bool Int64(int64_t i) { return true; }
    bool Uint64(uint64_t u) { return true; }
    bool Double(double d) { return true; }
    bool String(const char*str, rapidjson::SizeType, bool) { return true; }
    bool StartObject() {
        if (mode.empty()) {
            mode.push(Root);
        } else if (mode.top() == Nodes) {
            ++vertex_count;
        } else if (mode.top() == Links) {
            ++edge_count;
        }
        return true;
    }
    bool Key(const char* str, rapidjson::SizeType, bool) {
        lastKey = str;
        return true;
    }
    bool EndObject(rapidjson::SizeType) {
        return true;
    }
    bool StartArray() {
        if (mode.top() == Root && lastKey == "nodes") {
            mode.push(Nodes);
        } else if (mode.top() == Root && lastKey == "links") {
            mode.push(Links);
        } else {
            mode.push(Nop);
        }
        return true;
    }
    bool EndArray(rapidjson::SizeType) {
        mode.pop();
        return true;
    }
};
    /**
 * This is the main sax style parser for the json state
 * It stores it's current position within the hierarchy with mode, lastKey and current_variable_array_index
 */
class JSONAdjacencyGraphReader : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, JSONAdjacencyGraphReader>  {
    enum Mode{ Nop, Root, Nodes, Links, Node, Link, VariableArray };
    std::stack<Mode> mode;
    std::string lastKey;
    std::string filename;
    /**
     * The graph to update during parsing
     */
    const std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers>& graph;
    /**
     * CUDA stream required for synchronising graph buffers
     */
    cudaStream_t stream;
    /**
     * Access to graph metadata
     */
    const EnvironmentDirectedGraphData &metagraph;
    /**
     * Used for mapping vertex string IDs to numeric IDs/indices
     */
    std::map<std::string, unsigned int> vertex_id_map;
    /**
     * Tracks the current vertex/edge index in the array
     */
    unsigned int current_index = 0;
    /**
    * Tracks the next auto vertex id to be assigned
    */
    unsigned int next_id = 1;
    /**
     * Set to track assigned IDs
     */
    std::set<unsigned int> used_vertex_ids;
    /**
     * Tracks current position reading variable array
     */
    unsigned int current_variable_array_index = 0;
    /**
     * Source and target are stored separately, but must be set within graph simultaneously.
     * So these are used to cache the first value temporarily
     */
    unsigned int last_source = ID_NOT_SET, last_target = ID_NOT_SET;

 public:
     JSONAdjacencyGraphReader(const std::string &_filename,
        const std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers>& _graph, cudaStream_t _stream)
        : filename(_filename)
        , graph(_graph)
        , stream(_stream)
        , metagraph(_graph->getDescription()) { }
    template<typename T>
    bool processValue(const T val) {
        Mode isArray = Nop;
        if (mode.top() == VariableArray) {
            isArray = mode.top();
            mode.pop();
        }
        if (mode.top() == Node) {
            const auto f = metagraph.vertexProperties.find(lastKey);
            if (f == metagraph.vertexProperties.end()) {
                if (current_index == 0) {
                    fprintf(stderr, "Input file '%s' contains unexpected vector property '%s', skipped during parse.\n", filename.c_str(), lastKey.c_str());
                }
                return true;
            }
            const auto &var_data = f->second;
            size_type elements = var_data.elements;
            const std::type_index val_type = var_data.type;
            if (val_type == std::type_index(typeid(float))) {
                const float t = static_cast<float>(val);
                graph->getVertexPropertyBuffer<float>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(double))) {
                const double t = static_cast<double>(val);
                graph->getVertexPropertyBuffer<double>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(int64_t))) {
                const int64_t t = static_cast<int64_t>(val);
                graph->getVertexPropertyBuffer<int64_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(uint64_t))) {
                const uint64_t t = static_cast<uint64_t>(val);
                graph->getVertexPropertyBuffer<uint64_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(int32_t))) {
                const int32_t t = static_cast<int32_t>(val);
                graph->getVertexPropertyBuffer<int32_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(uint32_t))) {
                const uint32_t t = static_cast<uint32_t>(val);
                graph->getVertexPropertyBuffer<uint32_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(int16_t))) {
                const int16_t t = static_cast<int16_t>(val);
                graph->getVertexPropertyBuffer<int16_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(uint16_t))) {
                const uint16_t t = static_cast<uint16_t>(val);
                graph->getVertexPropertyBuffer<uint16_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(int8_t))) {
                const int8_t t = static_cast<int8_t>(val);
                graph->getVertexPropertyBuffer<int8_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(uint8_t))) {
                const uint8_t t = static_cast<uint8_t>(val);
                graph->getVertexPropertyBuffer<uint8_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(char))) {
                const char t = static_cast<char>(val);
                graph->getVertexPropertyBuffer<char>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            }  else {
                THROW exception::RapidJSONError("Input file '%s' contain vertex property '%s', of unknown type %s.\n", filename.c_str(), lastKey.c_str(), val_type.name());
            }
        } else if (mode.top() == Link) {
            const auto f = metagraph.edgeProperties.find(lastKey);
            if (f == metagraph.edgeProperties.end()) {
                if (current_index == 0) {
                    fprintf(stderr, "Input file '%s' contains unexpected edge property '%s', skipped during parse.\n", filename.c_str(), lastKey.c_str());
                }
                return true;
            }
            const auto &var_data = f->second;
            size_type elements = var_data.elements;
            const std::type_index val_type = var_data.type;
            if (val_type == std::type_index(typeid(float))) {
                const float t = static_cast<float>(val);
                graph->getEdgePropertyBuffer<float>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(double))) {
                const double t = static_cast<double>(val);
                graph->getEdgePropertyBuffer<double>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(int64_t))) {
                const int64_t t = static_cast<int64_t>(val);
                graph->getEdgePropertyBuffer<int64_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(uint64_t))) {
                const uint64_t t = static_cast<uint64_t>(val);
                graph->getEdgePropertyBuffer<uint64_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(int32_t))) {
                const int32_t t = static_cast<int32_t>(val);
                graph->getEdgePropertyBuffer<int32_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(uint32_t))) {
                const uint32_t t = static_cast<uint32_t>(val);
                graph->getEdgePropertyBuffer<uint32_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(int16_t))) {
                const int16_t t = static_cast<int16_t>(val);
                graph->getEdgePropertyBuffer<int16_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(uint16_t))) {
                const uint16_t t = static_cast<uint16_t>(val);
                graph->getEdgePropertyBuffer<uint16_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(int8_t))) {
                const int8_t t = static_cast<int8_t>(val);
                graph->getEdgePropertyBuffer<int8_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(uint8_t))) {
                const uint8_t t = static_cast<uint8_t>(val);
                graph->getEdgePropertyBuffer<uint8_t>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            } else if (val_type == std::type_index(typeid(char))) {
                const char t = static_cast<char>(val);
                graph->getEdgePropertyBuffer<char>(lastKey, elements, stream)[current_index * elements + current_variable_array_index++] = t;
            }  else {
                THROW exception::RapidJSONError("Input file '%s' contain edge property '%s', of unknown type %s.\n", filename.c_str(), lastKey.c_str(), val_type.name());
            }
        } else {
            THROW exception::RapidJSONError("Unexpected value with key '%s' whilst parsing input file '%s'.\n", lastKey.c_str(), filename.c_str());
        }
        if (isArray == VariableArray) {
            mode.push(isArray);
        } else {
            current_variable_array_index = 0;  // Didn't actually want to increment it above, because not in an array
        }
        return true;
    }
    bool Null() { return true; }
    bool Bool(bool b) { return processValue<bool>(b); }
    bool Int(int i) { return processValue<int32_t>(i); }
    bool Uint(unsigned u) { return processValue<uint32_t>(u); }
    bool Int64(int64_t i) { return processValue<int64_t>(i); }
    bool Uint64(uint64_t u) { return processValue<uint64_t>(u); }
    bool Double(double d) { return processValue<double>(d); }
    bool String(const char* str, rapidjson::SizeType, bool) {
        if (mode.top() == Node) {
            if (lastKey == "id") {
                // Attempt to convert the string to an int
                unsigned int parse_int = ID_NOT_SET;
                try {
                    parse_int = static_cast<unsigned int>(std::stoul(str));
                } catch (...) { }
                unsigned int assigned_id = parse_int;
                if (assigned_id == ID_NOT_SET) {
                    while (!used_vertex_ids.emplace(next_id++).second) { }
                }
                vertex_id_map.emplace(str, assigned_id);
                graph->setVertexID(current_index, assigned_id, stream);
                return true;
            } else {
                if (current_index) {
                    fprintf(stderr, "Input file '%s' contains vertex property '%s' of type String, this has been skipped during loading.", filename.c_str(), str);
                }
                return true;
            }
        } else if (mode.top() == Link) {
            const auto f = vertex_id_map.find(str);
            if (vertex_id_map.empty()) {
                THROW exception::RapidJSONError("'links' object occurs before 'nodes' object, unable to parse.\n", filename.c_str());
            } else if (f == vertex_id_map.end()) {
                THROW exception::RapidJSONError("Edge refers to unrecognised Vertex ID '%s', unable to load input file '%s'.\n", str, filename.c_str());
            }
            if (lastKey == "source") {
                if (last_target == ID_NOT_SET) {
                    last_source = f->second;
                } else {
                    graph->setEdgeSourceDestination(current_index, f->second, last_target);
                    last_target = ID_NOT_SET;
                }
                return true;
            } else if (lastKey == "target") {
                if (last_source == ID_NOT_SET) {
                    last_target = f->second;
                } else {
                    graph->setEdgeSourceDestination(current_index, last_source, f->second);
                    last_source = ID_NOT_SET;
                }
                return true;
            } else {
                if (current_index) {
                    fprintf(stderr, "Input file '%s' contains edge property '%s' of type String, this has been skipped during loading.", filename.c_str(), str);
                }
                return true;
            }
        }
        THROW exception::RapidJSONError("Unexpected string whilst parsing input file '%s', string properties are not supported.\n", filename.c_str());
    }
    bool StartObject() {
        if (mode.empty()) {
            mode.push(Root);
        } else if (mode.top() == Nodes) {
            mode.push(Node);
        } else if (mode.top() == Links) {
            mode.push(Link);
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
        if (mode.top() == Node || mode.top() == Link) {
            ++current_index;
        }
        mode.pop();
        return true;
    }
    bool StartArray() {
        if (current_variable_array_index != 0) {
            THROW exception::RapidJSONError("Array start when current_variable_array_index !=0, in file '%s'. This should never happen.\n", filename.c_str());
        }
        if (mode.top() == Root && lastKey == "nodes") {
            mode.push(Nodes);
        } else if (mode.top() == Root && lastKey == "links") {
            mode.push(Links);
        } else if (mode.top() == Node || mode.top() == Link) {
            mode.push(VariableArray);
        } else {
            THROW exception::RapidJSONError("Unexpected array start whilst parsing input file '%s'.\n", filename.c_str());
        }
        return true;
    }
    bool EndArray(rapidjson::SizeType) {
        if (mode.top() == VariableArray) {
            mode.pop();
            current_variable_array_index = 0;
        } else {
            mode.pop();
            current_index = 0;
        }
        return true;
    }
};
}  // namespace

void JSONGraphReader::loadAdjacencyLike(const std::string& filepath, const std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers>& directed_graph, cudaStream_t stream) {
    std::ifstream in(filepath, std::ios::in | std::ios::binary);
    if (!in) {
        THROW exception::InvalidFilePath("Unable to open file '%s' for reading.\n", filepath.c_str());
    }
    std::string filestring = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    rapidjson::StringStream filess(filestring.c_str());
    rapidjson::Reader reader;
    // First count the size of the graph
    JSONAdjacencyGraphSizeReader graphSizeCounter;
    rapidjson::ParseResult pr1 = reader.Parse<rapidjson::kParseNanAndInfFlag, rapidjson::StringStream, JSONAdjacencyGraphSizeReader>(filess, graphSizeCounter);
    if (pr1.Code() != rapidjson::ParseErrorCode::kParseErrorNone) {
        THROW exception::RapidJSONError("Whilst calculating graph size from input file '%s', RapidJSON returned error: %s\n", filepath.c_str(), rapidjson::GetParseError_En(pr1.Code()));
    }
    // Second (pre)allocate the graph's buffers
    directed_graph->setVertexCount(graphSizeCounter.getVertexCount(), stream);
    directed_graph->setEdgeCount(graphSizeCounter.getEdgeCount());
    // Third reset the string stream
    filess = rapidjson::StringStream(filestring.c_str());
    // Fourth parse the graph (and map string vertex IDs to integers)
    JSONAdjacencyGraphReader graphReader(filepath, directed_graph, stream);
    rapidjson::ParseResult pr2 = reader.Parse<rapidjson::kParseNanAndInfFlag, rapidjson::StringStream, JSONAdjacencyGraphReader>(filess, graphReader);
    if (pr2.Code() != rapidjson::ParseErrorCode::kParseErrorNone) {
        THROW exception::RapidJSONError("Whilst reading graph from input file '%s', RapidJSON returned error: %s\n", filepath.c_str(), rapidjson::GetParseError_En(pr1.Code()));
    }
}

}  // namespace io
}  // namespace flamegpu
