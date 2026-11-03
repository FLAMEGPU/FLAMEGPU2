#ifndef INCLUDE_FLAMEGPU_DETAIL_GPU_CUB_IMPORT_CUH_
#define INCLUDE_FLAMEGPU_DETAIL_GPU_CUB_IMPORT_CUH_

// Wrapper header importing cub / hipcub with warning suppression and namespace aliasing.

#ifdef FLAMEGPU_USE_CUDA
#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable : 4706 4834)
#endif  // _MSC_VER
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress 1719
#else
#pragma diag_suppress 1719
#endif  // __NVCC_DIAG_PRAGMA_SUPPORT__
#include <cub/cub.cuh>
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_default 1719
#else
#pragma diag_default 1719
#endif  // __NVCC_DIAG_PRAGMA_SUPPORT__
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER
#endif  // FLAMEGPU_USE_CUDA

#ifdef FLAMEGPU_USE_HIP
#include <hipcub/hipcub.hpp>
// namepspace alias so cub:: can be used
namespace cub = hipcub;
#endif

#endif  // INCLUDE_FLAMEGPU_DETAIL_GPU_CUB_IMPORT_CUH_
