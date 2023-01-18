#ifndef INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDIRECTEDGRAPHDESCRIPTION_CUH_
#define INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDIRECTEDGRAPHDESCRIPTION_CUH_

#include <vector>
#include <memory>
#include <string>

#include "flamegpu/model/EnvironmentDirectedGraphData.cuh"
#include "flamegpu/model/EnvironmentDescription.h"

namespace flamegpu {
/**
 * @brief Description class for directed graphs stored within the environment
 *
 * Allows properties to be attached to the vertices and edges of a directed graph stored within the environment.
 * Properties can be any arithmetic or enum type.
 * @todo Allow user to specify location to load graph on disk as part of desc?
 */
class CEnvironmentDirectedGraphDescription {
    /**
      * Data store class for this description, constructs instances of this class
      */
    friend struct EnvironmentDirectedGraphData;

 public:
    /**
     * Constructor, creates an interface to the EnvironmentDirectedGraphData
     * @param data Data store of this directed graphs's data
     */
    explicit CEnvironmentDirectedGraphDescription(std::shared_ptr<EnvironmentDirectedGraphData> data);
    explicit CEnvironmentDirectedGraphDescription(std::shared_ptr<const EnvironmentDirectedGraphData> data);
    /**
     * Copy constructor
     * Creates a new interface to the same EnvironmentDirectedGraphData/ModelData
     */
    CEnvironmentDirectedGraphDescription(const CEnvironmentDirectedGraphDescription& other_graph) = default;
    CEnvironmentDirectedGraphDescription(CEnvironmentDirectedGraphDescription&& other_graph) = default;
    /**
     * Assignment operator
     * Assigns this interface to the same EnvironmentDirectedGraphData/ModelData
     */
    CEnvironmentDirectedGraphDescription& operator=(const CEnvironmentDirectedGraphDescription& other_graph) = default;
    CEnvironmentDirectedGraphDescription& operator=(CEnvironmentDirectedGraphDescription&& other_graph) = default;
    /**
     * Equality operator, checks whether EnvironmentDirectedGraphDescription hierarchies are functionally the same
     * @param rhs right hand side
     * @returns True when directed graphs are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const CEnvironmentDirectedGraphDescription& rhs) const;
    /**
     * Equality operator, checks whether EnvironmentDirectedGraphDescription hierarchies are functionally different
     * @param rhs right hand side
     * @returns True when directed graphs are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const CEnvironmentDirectedGraphDescription& rhs) const;

    /**
     * @return The graph's name
     */
    std::string getName() const;
    /**
     * @param property_name Name used to refer to the desired property
     * @return The type of the named property
     * @throws exception::InvalidGraphProperty If a property with the name does not exist within the graph
     */
    const std::type_index& getVertexPropertyType(const std::string& property_name) const;
    const std::type_index& getEdgePropertyType(const std::string& property_name) const;
    /**
     * @param property_name Name used to refer to the desired property
     * @return The size of the named property's type
     * @throws exception::InvalidGraphProperty If a property with the name does not exist within the graph
     */
    size_t getVertexPropertySize(const std::string& property_name) const;
    size_t getEdgePropertySize(const std::string& property_name) const;
    /**
     * @param property_name Name used to refer to the desired property
     * @return The number of elements in the name property (1 if it isn't an array)
     * @throws exception::InvalidGraphProperty If a property with the name does not exist within the graph
     */
    flamegpu::size_type getVertexPropertyLength(const std::string& property_name) const;
    flamegpu::size_type getEdgePropertyLength(const std::string& property_name) const;
    /**
     * Get the total number of propertys this graph has
     * @return The total number of properties within the graph
     * @note This count includes internal properties used to track things such as ID
     */
    flamegpu::size_type geVertexPropertiesCount() const;
    flamegpu::size_type getEdgePropertiesCount() const;
    /**
     * @param property_name Name of the property to check
     * @return True when a property with the specified name exists within the graph
     */
    bool hasVertexProperty(const std::string& property_name) const;
    bool hasEdgeProperty(const std::string& property_name) const;

 protected:
    /**
     * The class which stores all of the layer's data.
     */
    std::shared_ptr<EnvironmentDirectedGraphData> graph;
};
class EnvironmentDirectedGraphDescription : public CEnvironmentDirectedGraphDescription {
 public:
    /**
     * Constructor, creates an interface to the EnvironmentDirectedGraphData
     * @param data Data store of this directed graph's data
     */
    explicit EnvironmentDirectedGraphDescription(std::shared_ptr<EnvironmentDirectedGraphData> data);
    /**
     * Copy constructor
     * Creates a new interface to the same EnvironmentDirectedGraphData/ModelData
     */
    EnvironmentDirectedGraphDescription(const EnvironmentDirectedGraphDescription& other_graph) = default;
    EnvironmentDirectedGraphDescription(EnvironmentDirectedGraphDescription&& other_graph) = default;
    /**
     * Assignment operator
     * Assigns this interface to the same EnvironmentDirectedGraphData/ModelData
     */
    EnvironmentDirectedGraphDescription& operator=(const EnvironmentDirectedGraphDescription& other_graph) = default;
    EnvironmentDirectedGraphDescription& operator=(EnvironmentDirectedGraphDescription&& other_graph) = default;

    /**
     * Adds a new property array to the graph
     * @param property_name Name of the vertex property array
     * @param default_value Default value of vertex property for vertex if unset, defaults to each element set 0
     * @tparam T Type of the graph property, this must be an arithmetic type
     * @tparam N The length of the property array (1 if not an array, must be greater than 0)
     * @throws exception::InvalidGraphProperty If a property already exists within the graph with the same name
     * @throws exception::InvalidGraphProperty If N is <= 0
     */
    template<typename T, size_type N>
    void newVertexProperty(const std::string& property_name, const std::array<T, N>& default_value = {});
    /**
     * Adds a new property array to the graph
     * @param property_name Name of the property array
     * @param default_value Default value of edge property for edges if unset, defaults to each element set 0
     * @tparam T Type of the graph property, this must be an arithmetic type
     * @tparam N The length of the property array (1 if not an array, must be greater than 0)
     * @throws exception::InvalidGraphProperty If a property already exists within the graph with the same name
     * @throws exception::InvalidGraphProperty If N is <= 0
     */
    template<typename T, size_type N>
    void newEdgeProperty(const std::string& property_name, const std::array<T, N>& default_value = {});
#ifndef SWIG
    /**
     * Adds a new property to the graph
     * @param property_name Name of the property
     * @param default_value Default value of vertex property for vertices if unset, defaults to 0
     * @tparam T Type of the graph property, this must be an arithmetic type
     * @throws exception::InvalidGraphProperty If a property already exists within the graph with the same name
     */
    template<typename T>
    void newVertexProperty(const std::string& property_name, const T& default_value = {});
    /**
     * Adds a new property to the graph
     * @param property_name Name of the property
     * @param default_value Default value of edge property for edges if unset, defaults to 0
     * @tparam T Type of the graph property, this must be an arithmetic type
     * @throws exception::InvalidGraphProperty If a property already exists within the graph with the same name
     */
    template<typename T>
    void newEdgeProperty(const std::string& property_name, const T& default_value = {});
#else
    /**
     * Adds a new vertex property to the graph
     * @param property_name Name of the property
     * @param default_value Default value of edge property for vertices where unset, defaults to 0
     * @tparam T Type of the graph property, this must be an arithmetic type
     * @throws exception::InvalidGraphProperty If a property already exists within the graph with the same name
     * @note Swig is unable to handle {} default param, however it's required for GLM support
     * Similarly, can't just provide 2 protoypes which overload, Python doesn't support that
     * Hence, easiest to require python users to init GLM types as arrays
     */
    template<typename T>
    void newVertexProperty(const std::string& property_name, const T& default_value = 0);
    /**
     * Adds a new vertex property array to the graph
     * @param property_name Name of the edge property array
     * @param length The length of the edge property array (1 if not an array, must be greater than 0)
     * @param default_value Default value of property for vertices if unset, defaults to each element set 0
     * @tparam T Type of the graph property, this must be an arithmetic type
     * @throws exception::InvalidGraphProperty If a edge property already exists within the graph with the same name
     * @throws exception::InvalidGraphProperty If length is <= 0
     */
    template<typename T>
    void newVertexPropertyArray(const std::string& property_name, size_type length, const std::vector<T>& default_value = {});
    /**
     * Adds a new edge property to the graph
     * @param property_name Name of the property
     * @param default_value Default value of edge property for edges if unset, defaults to 0
     * @tparam T Type of the graph property, this must be an arithmetic type
     * @throws exception::InvalidGraphProperty If a property already exists within the graph with the same name
     * @note Swig is unable to handle {} default param, however it's required for GLM support
     * Similarly, can't just provide 2 protoypes which overload, Python doesn't support that
     * Hence, easiest to require python users to init GLM types as arrays
     */
    template<typename T>
    void newEdgeProperty(const std::string& property_name, const T& default_value = 0);
    /**
     * Adds a new edge_property array to the graph
     * @param property_name Name of the edge property array
     * @param length The length of the edge property array (1 if not an array, must be greater than 0)
     * @param default_value Default value of edge property for edges if unset, defaults to each element set 0
     * @tparam T Type of the graph property, this must be an arithmetic type
     * @throws exception::InvalidGraphProperty If a property already exists within the graph with the same name
     * @throws exception::InvalidGraphProperty If length is <= 0
     */
    template<typename T>
    void newEdgePropertyArray(const std::string& property_name, size_type length, const std::vector<T>& default_value = {});
#endif
};

template<typename T>
void EnvironmentDirectedGraphDescription::newVertexProperty(const std::string& property_name, const T& default_value) {
    newVertexProperty<T, 1>(property_name, { default_value });
}
template<typename T, flamegpu::size_type N>
void EnvironmentDirectedGraphDescription::newVertexProperty(const std::string& property_name, const std::array<T, N>& default_value) {
    if (!property_name.empty() && property_name[0] == '_') {
        THROW exception::ReservedName("Graph property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDirectedGraphDescription::newVertexProperty().");
    }
    // Array length 0 makes no sense
    static_assert(detail::type_decode<T>::len_t * N > 0, "A property cannot have 0 elements.");
    if (graph->vertexProperties.find(property_name) == graph->vertexProperties.end()) {
        const std::array<typename detail::type_decode<T>::type_t, detail::type_decode<T>::len_t* N>* casted_default =
            reinterpret_cast<const std::array<typename detail::type_decode<T>::type_t, detail::type_decode<T>::len_t* N>*>(&default_value);
        graph->vertexProperties.emplace(property_name, Variable(*casted_default));
        return;
    }
    THROW exception::InvalidGraphProperty("Graph ('%s') already contains vertex property '%s', "
        "in EnvironmentDirectedGraphDescription::newVertexProperty().",
        graph->name.c_str(), property_name.c_str());
}
template<typename T>
void EnvironmentDirectedGraphDescription::newEdgeProperty(const std::string& property_name, const T& default_value) {
    newEdgeProperty<T, 1>(property_name, { default_value });
}
template<typename T, flamegpu::size_type N>
void EnvironmentDirectedGraphDescription::newEdgeProperty(const std::string& property_name, const std::array<T, N>& default_value) {
    if (!property_name.empty() && property_name[0] == '_') {
        THROW exception::ReservedName("Graph property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDirectedGraphDescription::newEdgeProperty().");
    }
    // Array length 0 makes no sense
    static_assert(detail::type_decode<T>::len_t * N > 0, "A property cannot have 0 elements.");
    if (graph->edgeProperties.find(property_name) == graph->edgeProperties.end()) {
        const std::array<typename detail::type_decode<T>::type_t, detail::type_decode<T>::len_t* N>* casted_default =
            reinterpret_cast<const std::array<typename detail::type_decode<T>::type_t, detail::type_decode<T>::len_t* N>*>(&default_value);
        graph->edgeProperties.emplace(property_name, Variable(*casted_default));
        return;
    }
    THROW exception::InvalidGraphProperty("Graph ('%s') already contains edge property '%s', "
        "in EnvironmentDirectedGraphDescription::newEdgeProperty().",
        graph->name.c_str(), property_name.c_str());
}
#ifdef SWIG
template<typename T>
void EnvironmentDirectedGraphDescription::newVertexPropertyArray(const std::string& property_name, const size_type length, const std::vector<T>& default_value) {
    if (!property_name.empty() && property_name[0] == '_') {
        THROW exception::ReservedName("Graph property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDirectedGraphDescription::newVertexPropertyArray().");
    }
    if (length == 0) {
        THROW exception::InvalidGraphProperty("Graph property arrays must have a length greater than 0."
            "in EnvironmentDirectedGraphDescription::newVertexPropertyArray().");
    }
    if (default_value.size() && default_value.size() != length) {
        THROW exception::InvalidGraphProperty("Graph vertex property array length specified as %d, but default value provided with %llu elements, "
            "in EnvironmentDirectedGraphDescription::newVertexPropertyArray().",
            length, static_cast<unsigned int>(default_value.size()));
    }
    if (graph->vertexProperties.find(property_name) == graph->vertexProperties.end()) {
        std::vector<typename detail::type_decode<T>::type_t> temp(static_cast<size_t>(detail::type_decode<T>::len_t * length));
        if (default_value.size()) {
            memcpy(temp.data(), default_value.data(), sizeof(typename detail::type_decode<T>::type_t) * detail::type_decode<T>::len_t * length);
        }
        graph->vertexProperties.emplace(property_name, Variable(detail::type_decode<T>::len_t * length, temp));
        return;
    }
    THROW exception::InvalidGraphProperty("Graph ('%s') already contains vertex property '%s', "
        "in EnvironmentDirectedGraphDescription::newVertexPropertyArray().",
        graph->name.c_str(), property_name.c_str());
}
template<typename T>
void EnvironmentDirectedGraphDescription::newEdgePropertyArray(const std::string& property_name, const size_type length, const std::vector<T>& default_value) {
    if (!property_name.empty() && property_name[0] == '_') {
        THROW exception::ReservedName("Graph property names cannot begin with '_', this is reserved for internal usage, "
            "in EnvironmentDirectedGraphDescription::newEdgePropertyArray().");
    }
    if (length == 0) {
        THROW exception::InvalidGraphProperty("Graph property arrays must have a length greater than 0."
            "in EnvironmentDirectedGraphDescription::newEdgePropertyArray().");
    }
    if (default_value.size() && default_value.size() != length) {
        THROW exception::InvalidGraphProperty("Graph vertex property array length specified as %d, but default value provided with %llu elements, "
            "in EnvironmentDirectedGraphDescription::newEdgePropertyArray().",
            length, static_cast<unsigned int>(default_value.size()));
    }
    if (graph->edgeProperties.find(property_name) == graph->edgeProperties.end()) {
        std::vector<typename detail::type_decode<T>::type_t> temp(static_cast<size_t>(detail::type_decode<T>::len_t * length));
        if (default_value.size()) {
            memcpy(temp.data(), default_value.data(), sizeof(typename detail::type_decode<T>::type_t) * detail::type_decode<T>::len_t * length);
        }
        graph->edgeProperties.emplace(property_name, Variable(detail::type_decode<T>::len_t * length, temp));
        return;
    }
    THROW exception::InvalidGraphProperty("Graph ('%s') already contains edge property '%s', "
        "in EnvironmentDirectedGraphDescription::newEdgePropertyArray().",
        graph->name.c_str(), property_name.c_str());
}
#endif

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDIRECTEDGRAPHDESCRIPTION_CUH_
