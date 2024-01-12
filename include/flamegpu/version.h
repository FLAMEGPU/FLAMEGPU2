#ifndef INCLUDE_FLAMEGPU_VERSION_H_
#define INCLUDE_FLAMEGPU_VERSION_H_

/**
 * FLAME GPU 2 version number macro
 * 
 * FLAMEGPU_VERSION / 1000000 is the  major version.
 * FLAMEGPU_VERSION / 1000 % 1000 is the minor version
 * FLAMEGPU_VERSION % 1000 is the Patch version.
 * Does not include pre-release or build-metadata for the semantic version as these may contain non-numeric characters
 */
#define FLAMEGPU_VERSION 2000000

namespace flamegpu {
/**
 * FLAME GPU Version number as a 7+ digit integer within the namespace
 * flamegpu::VERSION / 1000000 is the  major version.
 * flamegpu::VERSION / 1000 % 1000 is the minor version
 * flamegpu::VERSION % 1000 is the Patch version.
 */
static constexpr unsigned int VERSION = FLAMEGPU_VERSION;
/**
 * FLAME GPU Major Version number within the namespace
 * Derived from flamegpu::VERSION / 1000000 is the  major version.
 */
static constexpr unsigned int VERSION_MAJOR = flamegpu::VERSION / 1000000;
/**
 * FLAME GPU Minor Version number.
 * Derived from flamegpu::VERSION / 1000 % 1000
 */
static constexpr unsigned int VERSION_MINOR = flamegpu::VERSION / 1000 % 1000;
/**
 * FLAME GPU Patch Version number.
 * Derived from flamegpu::VERSION % 1000
 */
static constexpr unsigned int VERSION_PATCH = flamegpu::VERSION % 1000;

/**
 * Namespaced FLAME GPU version Prerelease component
 * A set of . separated pre-release components as a string, following the semver rules for comparison.
 */
static constexpr char VERSION_PRERELEASE[] = "rc.2";

/**
 * Namespaced FLAME GPU version build metadata component
 * A set of . separated pre-release components as a string, following the semver rules for comparison.
 */
extern const char VERSION_BUILDMETADATA[];

/**
 * Namespaced FLAME GPU release version
 * SemVer Release string literal
 */
extern const char VERSION_STRING[];

/**
 * Namespaced FLAME GPU full SemVer version
 */
extern const char VERSION_FULL[];

/**
 * A randomly generated string created once per CMake build directory/configuration, for use with telemetry to approximate unique users without any PII (if telemetry is enabled).
 */
extern const char TELEMETRY_RANDOM_ID[];

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_VERSION_H_
