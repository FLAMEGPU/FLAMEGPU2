#[[[
# Deprecated module which formerly handled CMAKE_CUDA_ARCHITECTURES with sane library-provided defaults. 
# CMAKE_CUDA_ARCHITECTURES/CMAKE_HIP_ARCHITECTURES must now be handled by the user, else the cmake/compiler defaults will be used (SM_50 for CUDA 12, SM_75 for CUDA 13, 'native' for HIP (most of the time))
#]]
include_guard(GLOBAL)

message(DEPRECATION
    "  ${CMAKE_CURRENT_LIST_FILE} is deprecated and will be removed in a future release.\n"
    "  flamegpu_init_cuda_architectures and flamegpu_set_cuda_architectures are no longer required.\n"
    "  Please ensure you are setting CMAKE_CUDA_ARCHITECTURES/CMAKE_HIP_ARCHITECTURES appropriately for your GPUs.\n"
)

# This method is deprecated and no longer has an impact, but has not yet been removed to avoid causing CMake configuration errors
function(flamegpu_init_cuda_architectures)
    # Issue a deprecation warning
    message(DEPRECATION
        "  flamegpu_init_cuda_architectures is deprecated and will be removed in a future release.\n"
        "  Please remove the call to flamegpu_init_cuda_architectures and ensure you are setting CMAKE_CUDA_ARCHITECTURES/CMAKE_HIP_ARCHITECTURES appropriately for your GPUs."
    )
    # Forward all args to the generalised version of this function
    flamegpu_init_gpu_architectures(${ARGN})
endfunction()

# This method is deprecated and no longer has an impact, but has not yet been removed to avoid causing CMake configuration errors
function(flamegpu_set_cuda_architectures)
    # Issue a deprecation warning
    message(DEPRECATION
        "  flamegpu_set_cuda_architectures is deprecated and will be removed in a future release.\n"
        "  Please remove the call to flamegpu_set_cuda_architectures and ensure you are setting CMAKE_CUDA_ARCHITECTURES/CMAKE_HIP_ARCHITECTURES appropriately for your GPUs."
    )
    # Forward all args to the generalised version of this function
    flamegpu_set_gpu_architectures(${ARGN})
endfunction()
