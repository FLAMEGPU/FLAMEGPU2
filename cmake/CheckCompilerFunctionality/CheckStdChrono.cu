// Including chrono is enough to trigger the compilation error being tested for. See:
// https://github.com/FLAMEGPU/FLAMEGPU2/issues/575
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100102
#include <chrono>

// CMake try_compile performs a link step, so main is required
int main() { return 0; }