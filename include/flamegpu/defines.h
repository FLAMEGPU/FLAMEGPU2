#ifndef INCLUDE_FLAMEGPU_DEFINES_H_
#define INCLUDE_FLAMEGPU_DEFINES_H_

namespace flamegpu {

// Definitions class, for macros and so on.
/**
 * Type used for generic identifiers, primarily used for Agent ids
 */
typedef unsigned int id_t;
/**
 * Internal variable name used for IDs
 */
constexpr const char* ID_VARIABLE_NAME = "_id";
/**
 * Internal variable name used for source-dest pairs
 * @note These are always stored in [dest, source] order, but shown to the user in [source, dest] order. This enables 2D sorting
 */
constexpr const char* GRAPH_SOURCE_DEST_VARIABLE_NAME = "_source_dest";
/**
 * Internal variable name used to the pointer to the PBM of edges leaving each vertex
 */
constexpr const char* GRAPH_VERTEX_PBM_VARIABLE_NAME = "_pbm";
/**
 * Internal variable name used to the pointer to the (inverted) PBM of edges joining each vertex
 */
constexpr const char* GRAPH_VERTEX_IPBM_VARIABLE_NAME = "_ipbm";
/**
 * Edges are not sorted in order of the IPBM, instead the IPBM points to indices in this list of edge indexes
 */
constexpr const char* GRAPH_VERTEX_IPBM_EDGES_VARIABLE_NAME = "_ipbm_edges";
/**
 * Accessing an ID within this buffer will return the index of the vertex
 **/
constexpr const char* GRAPH_VERTEX_INDEX_MAP_VARIABLE_NAME = "_index_map";
/**
 * Internal value used when IDs have not be set
 * If this value is changed, things may break
 */
constexpr id_t ID_NOT_SET = 0;
/**
* Typedef for verbosity level of the API
*/
enum class Verbosity {Quiet, Default, Verbose};

typedef unsigned int size_type;

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DEFINES_H_
