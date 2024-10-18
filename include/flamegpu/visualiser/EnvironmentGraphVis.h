#ifndef INCLUDE_FLAMEGPU_VISUALISER_ENVIRONMENTGRAPHVIS_H_
#define INCLUDE_FLAMEGPU_VISUALISER_ENVIRONMENTGRAPHVIS_H_
#ifdef FLAMEGPU_VISUALISATION

#include <memory>
#include <string>

#include "flamegpu/visualiser/color/Color.h"

namespace flamegpu {
struct EnvironmentDirectedGraphData;
namespace detail {
class CUDAEnvironmentDirectedGraphBuffers;
}  // namespace detail
namespace visualiser {
struct LineConfig;
struct EnvironmentGraphVisData {
    explicit EnvironmentGraphVisData(std::shared_ptr <EnvironmentDirectedGraphData> _graphData, std::shared_ptr<LineConfig>_lines);
    void constructGraph(const std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> &graph);
    std::string x_varName = "";
    std::string y_varName = "";
    std::string z_varName = "";
    std::string xy_varName = "";
    std::string xyz_varName = "";
    Color color;
    const std::shared_ptr<EnvironmentDirectedGraphData> graphData;
    const std::shared_ptr<LineConfig> lines;
};
class EnvironmentGraphVis {
 public:
     explicit EnvironmentGraphVis(std::shared_ptr<EnvironmentGraphVisData> data);

    /**
     * Set the name of the variable representing the agents x/y/z location coordinates
     * @param var_name Name of the agent variable
     * @note unnecessary if the variables are named "x", "y", "z" respectively
     * @note Implicitly calls clearXYProperty(), clearXYZProperty()
     * @throws InvalidEnvProperty If the variable is not type float[1]
     */
    void setXVertexProperty(const std::string &var_name);
    void setYVertexProperty(const std::string &var_name);
    void setZVertexProperty(const std::string &var_name);
    /**
     * Set the name of the array variable (length 2) representing the agents x/y location coordinates
     * @param var_name Name of the agent variable
     * @note Implicitly calls clearXProperty(),  clearYProperty(), clearZProperty(),clearXYZProperty()
     * @throws InvalidEnvProperty If the variable is not type float[2]
     */
    void setXYVertexProperty(const std::string &var_name);
    /**
     * Set the name of the array variable (length 3) representing the agents x/y/z location coordinates
     * @param var_name Name of the agent variable
     * @note Implicitly calls clearXProperty(),  clearYProperty(), clearZProperty(),clearXYProperty()
     * @throws InvalidEnvProperty If the variable is not type float[3]
     */
    void setXYZVertexProperty(const std::string &var_name);
    /**
     * Set the colour the graph will be rendered
     * @param cf The colour to be used, default white
     */
     void setColor(const Color& cf);

 private:
    /**
     * Pointer to data struct
     */
    std::shared_ptr<EnvironmentGraphVisData> data;
};
}  // namespace visualiser
}  // namespace flamegpu

#endif  // FLAMEGPU_VISUALISATION
#endif  // INCLUDE_FLAMEGPU_VISUALISER_ENVIRONMENTGRAPHVIS_H_
