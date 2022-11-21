# CMake file to be injected into a project via CMAKE_PROJECT<PROJECT-NAME>_INCLUDE
# If CUDA is enabled, call a CMake function which gracefully sets a library-useful

# Set a locally scoped cmake variable, to alter the error message within.
set(flamegpu_IN_PROJECT_INCLUDE ON)
# Call the appropraite command to set CMAKE_CUDA_ARCHITECTURES to the user-provided value, the exising value, or a sane libray-provided defualt
flamegpu_set_cuda_architectures()
# Unset the variable used to alter behaviour in set_cuda_architectures
unset(flamegpu_IN_PROJECT_INCLUDE)