#include <utility>
#include <vector>

#include "helpers/host_reductions_common.h"

namespace flamegpu {
namespace test_host_reductions {
float float_out = 0;
double double_out = 0;
char char_out = 0;
unsigned char uchar_out = 0;
uint16_t uint16_t_out = 0;
int16_t int16_t_out = 0;
uint32_t uint32_t_out = 0;
int32_t int32_t_out = 0;
uint64_t uint64_t_out = 0;
int64_t int64_t_out = 0;
#ifdef FLAMEGPU_USE_GLM
glm::vec3 vec3_t_out = glm::vec3(0);
#endif
std::pair<double, double> mean_sd_out;
std::vector<unsigned int> uint_vec;
std::vector<int> int_vec;

}  // namespace test_host_reductions
}  // namespace flamegpu
