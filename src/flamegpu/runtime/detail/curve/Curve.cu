#include "flamegpu/runtime/detail/curve/Curve.cuh"

#include <cuda_runtime.h>

#include <string>
#include <cstring>

namespace flamegpu {
namespace detail {
namespace curve {

__host__ Curve::VariableHash Curve::variableRuntimeHash(const std::string& _str) {
    const char* str = _str.c_str();
    const size_t length = std::strlen(str) + 1;
    unsigned int hash = 2166136261u;

    for (size_t i = 0; i < length; ++i) {
        hash ^= *str++;
        hash *= 16777619u;
    }
    return hash;
}
__host__ Curve::VariableHash Curve::variableRuntimeHash(unsigned int num) {
    return variableRuntimeHash(std::to_string(num));
}

}  // namespace curve
}  // namespace detail
}  // namespace flamegpu
