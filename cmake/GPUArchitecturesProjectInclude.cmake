# CMake file to be injected into a project via CMAKE_PROJECT<PROJECT-NAME>_INCLUDE

# Set a locally scoped cmake variable, to alter the error message within.
set(flamegpu_IN_PROJECT_INCLUDE ON)
# Call the appropriate command to set CMAKE_CUDA/HIP_ARCHITECTURES to the user-provided value, the existing value, or a sane library-provided default
flamegpu_set_gpu_architectures()
# Unset the variable used to alter behaviour in set_gpu_architectures
unset(flamegpu_IN_PROJECT_INCLUDE)