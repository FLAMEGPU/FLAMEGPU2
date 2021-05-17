#include "flamegpu/runtime/utility/HostRandom.cuh"

namespace flamegpu {

void HostRandom::setSeed(const unsigned int &seed) {
    rng.reseed(seed);
}
unsigned int HostRandom::getSeed() const {
    return static_cast<unsigned int>(rng.seed());
}

}  // namespace flamegpu
