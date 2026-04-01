#[[[
# Deprecated module for handling CMAKE_CUDA_ARCHITECTURES with sane library-provided defaults. 
# Use GPUArchitectures.cmake and flamegpu_init_gpu_architectures / flamegpu_set_gpu_architectures instead
#]]
include_guard(GLOBAL)

# Include the new version of this file
include(${CMAKE_CURRENT_LIST_DIR}/GPUArchitectures.cmake)

# This method is deprecated. Use flamegpu_set_gpu_architectures from GPUArchitectures.cmake instead
function(flamegpu_init_cuda_architectures)
    # Issue a deprecation warning
    message(AUTHOR_WARNING
        "  flamegpu_init_cuda_architectures is DEPRECATED.\n"
        "  Please use flamegpu_init_gpu_architectures from GPUArchitectures.cmake instead.\n"
        "  flamegpu_init_cuda_architectures will be removed in a future release."
    )
    # Forward all args to the generalised version of this function
    flamegpu_init_gpu_architectures(${ARGN})
endfunction()

# This method is deprecated. Use flamegpu_set_gpu_architectures from GPUArchitectures.cmake instead
function(flamegpu_set_cuda_architectures)
    # Issue a deprecation warning
    message(AUTHOR_WARNING
        "  flamegpu_set_cuda_architectures is DEPRECATED.\n"
        "  Please use flamegpu_set_gpu_architectures from GPUArchitectures.cmake instead.\n"
        "  flamegpu_set_cuda_architectures will be removed in a future release."
    )
    # Forward all args to the generalised version of this function
    flamegpu_set_gpu_architectures(${ARGN})
endfunction()
