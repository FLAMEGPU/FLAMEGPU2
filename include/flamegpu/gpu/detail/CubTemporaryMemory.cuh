#ifndef INCLUDE_FLAMEGPU_GPU_DETAIL_CUBTEMPORARYMEMORY_CUH_
#define INCLUDE_FLAMEGPU_GPU_DETAIL_CUBTEMPORARYMEMORY_CUH_

#include <unordered_map>
#include <utility>

namespace flamegpu {
namespace detail {

class CubTemporaryMemory {
 public:
    CubTemporaryMemory();
    ~CubTemporaryMemory();
    void resize(size_t newSize);

    void* getPtr() const { return d_cub_temp; }
    size_t &getSize() const { d_cub_temp_size_rtn = d_cub_temp_size; return d_cub_temp_size_rtn; }

 private:
    void* d_cub_temp;
    size_t d_cub_temp_size;
    // We have this version, so it can be passed directly to cub (requires non-const reference)
    // It is simply overwritten everytime it is requested, incase it gets accidentally changed
    mutable size_t d_cub_temp_size_rtn;
};

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_GPU_DETAIL_CUBTEMPORARYMEMORY_CUH_
