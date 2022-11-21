// We require std::filesytem, but just requireing std=c++17 does not enforce this for all compilers, so check it works. (I.e. GCC < 8 is a problem.)
// CMake doesn't appear to have knowledge of this feature.
#include <filesystem>
int main() { return 0; }