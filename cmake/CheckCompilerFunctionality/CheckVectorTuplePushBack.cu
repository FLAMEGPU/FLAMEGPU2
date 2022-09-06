// GCC 9 + CUDA 11.0 + --std=c++17 errors during complation of std::vector<std::tuple<>>::push_back. 
// Test this for the configured CUDA, incase it is OK on the given system
// See https://github.com/FLAMEGPU/FLAMEGPU2/issues/650 for more information 
#include <tuple>
#include <vector>

int main (int argc, char * argv[]) {
    std::vector<std::tuple<float>> v;
    std::tuple<float> t = {1.f};
    v.push_back(t);  // segmentation fault
}
