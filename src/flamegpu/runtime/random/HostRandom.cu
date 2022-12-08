#include "flamegpu/runtime/random/HostRandom.cuh"

namespace flamegpu {

void HostRandom::setSeed(const uint64_t seed) {
    rng.reseed(seed);
}
uint64_t HostRandom::getSeed() const {
    return static_cast<uint64_t>(rng.seed());
}

}  // namespace flamegpu
