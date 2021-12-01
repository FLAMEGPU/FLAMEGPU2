# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
## [Unreleased]

### Added

### Changed (Breaking)

### Changed

### Deprecated

### Removed

### Fixed
-->

## [2.0.0-alpha.2] - 2021-12-09

### Added
+ Environment macro properties, designed to hold large amount of data (e.g. 4 dimensional arrays) which agents can mutate via atomic operations. ([#643](https://github.com/FLAMEGPU/FLAMEGPU2/issues/643), [#738](https://github.com/FLAMEGPU/FLAMEGPU2/issues/738))
+ Support for using CUDA (11.3+) provided Thrust/CUB, if available. ([#657](https://github.com/FLAMEGPU/FLAMEGPU2/issues/657), [#692](https://github.com/FLAMEGPU/FLAMEGPU2/issues/692))
+ Agents can now be automatically sorted according to message order. ([#723](https://github.com/FLAMEGPU/FLAMEGPU2/issues/723))
+ Added Python 3.10 to CI release build matrix. ([#706](https://github.com/FLAMEGPU/FLAMEGPU2/issues/706))
+ Added contact links to new issue template ([#722](https://github.com/FLAMEGPU/FLAMEGPU2/issues/722))
+ Added a manual Windows test build CI action ([#741](https://github.com/FLAMEGPU/FLAMEGPU2/issues/741))

### Changed (Breaking)
+ Simulation times are now output in seconds and stored as double (previously millisecond, float). ([#691](https://github.com/FLAMEGPU/FLAMEGPU2/issues/691))

### Changed
+ Update Ubuntu CI to build SWIG 4.0.2 from source. ([#705](https://github.com/FLAMEGPU/FLAMEGPU2/issues/705))
+ Re-enable CMake targets MinSizeRel, RelWithDebingo. ([#698](https://github.com/FLAMEGPU/FLAMEGPU2/issues/698), [#704](https://github.com/FLAMEGPU/FLAMEGPU2/issues/704))
+ Update CMake target_link_libraries to have explicit visibility. ([#701](https://github.com/FLAMEGPU/FLAMEGPU2/issues/701), [#703](https://github.com/FLAMEGPU/FLAMEGPU2/issues/703))
+ Reduce the context creation threshold used inside test suite. ([#691](https://github.com/FLAMEGPU/FLAMEGPU2/issues/691))
+ Host functions are now stored internally with a `std::vector` to preserve order (previously order was undefined). ([#707](https://github.com/FLAMEGPU/FLAMEGPU2/issues/707), [#708](https://github.com/FLAMEGPU/FLAMEGPU2/issues/708))
+ Improve guidance in README for new visual studio/CUDA users. ([#702](https://github.com/FLAMEGPU/FLAMEGPU2/issues/702))
+ Update CI to support CUDA 11.5, and use this for 'latest' builds. ([#716](https://github.com/FLAMEGPU/FLAMEGPU2/issues/716))
+ Updated uses of diag_ pragma to nv_diag_, to be CUDA 11.5+ compatible. ([#716](https://github.com/FLAMEGPU/FLAMEGPU2/issues/716))
+ Various improvements to the Boids example models ([#739](https://github.com/FLAMEGPU/FLAMEGPU2/issues/739))

### Removed
+ Removed `Simulation::getSimulationConfig()` from the Python interface. ([#689](https://github.com/FLAMEGPU/FLAMEGPU2/issues/689), [#690](https://github.com/FLAMEGPU/FLAMEGPU2/issues/690))

### Fixed
+ Python example no longer mutates constant accessor to simulation config. ([#694](https://github.com/FLAMEGPU/FLAMEGPU2/issues/694))
+ Array message test suite would fail to build with `NO_SEATBELTS` enabled. ([#695](https://github.com/FLAMEGPU/FLAMEGPU2/issues/695))
+ Add missing `SEATBELTS` checks for reserved names within various `DeviceAPI` classes. ([#700](https://github.com/FLAMEGPU/FLAMEGPU2/issues/700))
+ Add missing `MessageBruteForce::getVariableLength()` method. ([#709](https://github.com/FLAMEGPU/FLAMEGPU2/issues/709))
+ Fixed cases where throwing of a DeviceException was not followed by a safe return. ([#718](https://github.com/FLAMEGPU/FLAMEGPU2/issues/718))
+ `SubModelDescription::(get)SubEnvironment(true)` now also automatically maps macro properties. ([#724](https://github.com/FLAMEGPU/FLAMEGPU2/issues/724))
+ `CUDAMessage` no longer loses data if a resize is performed before an append. ([#725](https://github.com/FLAMEGPU/FLAMEGPU2/issues/725), [#726](https://github.com/FLAMEGPU/FLAMEGPU2/issues/726))
+ Logging the mean of an agent variable for an empty pop, would return NaN, producing an invalid Log file. 0 is now returned. ([#734](https://github.com/FLAMEGPU/FLAMEGPU2/issues/734))
+ `CUDAEnsemble` no longer always logs both step and exit files to disk, if either is required. ([#730](https://github.com/FLAMEGPU/FLAMEGPU2/issues/730))
+ Corrected memory allocation calculations within `CUDAScatter::arrayMessageReorder()`. ([#736](https://github.com/FLAMEGPU/FLAMEGPU2/issues/736))
+ Explicitly flush CUDAEnsemble progress print statements, so they always display as expected.
+ Minor corrections to the handling of `Simulation` logging configs. ([#742](https://github.com/FLAMEGPU/FLAMEGPU2/issues/742))
+ DeviceError no longer handles %s formatter wrong in Release builds. ([#744](https://github.com/FLAMEGPU/FLAMEGPU2/issues/744), [#746](https://github.com/FLAMEGPU/FLAMEGPU2/issues/746), [#743](https://github.com/FLAMEGPU/FLAMEGPU2/issues/743))


## [2.0.0-alpha.1] - 2021-09-03

### Added
+
+ Optional support for vector types via [GLM](https://github.com/g-truc/glm) ([#217](https://github.com/FLAMEGPU/FLAMEGPU2/issues/217))
  + This is currently behind a CMake option due to significant RTC compilation time increases
+ Created `CHANGELOG.md` ([#618](https://github.com/FLAMEGPU/FLAMEGPU2/issues/#618), [#630](https://github.com/FLAMEGPU/FLAMEGPU2/pull/630))
+ Release process documentation ([#622](https://github.com/FLAMEGPU/FLAMEGPU2/issues/#622))
+ Thorough testing of `flamegpu::CUDAEnsemble`, `flamegpu::RunPlan` and `flamegpu::RunPlanVector` ([#656](https://github.com/FLAMEGPU/FLAMEGPU2/issues/656), [#665](https://github.com/FLAMEGPU/FLAMEGPU2/pull/665))
+ Added `uint64_t flamegpu::RunPlanVector::getRandomPropertySeed()` ([#656](https://github.com/FLAMEGPU/FLAMEGPU2/issues/656), [#665](https://github.com/FLAMEGPU/FLAMEGPU2/pull/665))

### Changed

+ Use `IntT` in `MessageBucketDevice` to resolve clang sign comparison warnings ([#554](https://github.com/FLAMEGPU/FLAMEGPU2/issues/554))
+ Default value of `-s/--steps` set to `1` rather than `0` ([#634](https://github.com/FLAMEGPU/FLAMEGPU2/issues/634))
+ All RNG seeds are now `uint64_t` ([#656](https://github.com/FLAMEGPU/FLAMEGPU2/issues/656), [#665](https://github.com/FLAMEGPU/FLAMEGPU2/pull/665))
+ Assorted bugfixes for `RunPlan`, `RunPlanVector` and `CUDAEnsemble` ([#656](https://github.com/FLAMEGPU/FLAMEGPU2/issues/656), [#665](https://github.com/FLAMEGPU/FLAMEGPU2/pull/665))
+ Linux python binary wheel generation now produces wheels supporting `glibc >= 2.17` ([#632](https://github.com/FLAMEGPU/FLAMEGPU2/issues/632))
+ CI configuration changes ([#641](https://github.com/FLAMEGPU/FLAMEGPU2/issues/641))
+ CMake modernisation, including use of target properties, in-source build prevention, support for patched GCC `10.3.0` and `11.1.0` ([#586](https://github.com/FLAMEGPU/FLAMEGPU2/issues/586))
+ `include/flamegpu/version.h` is no longer generated by CMake, allowing true out-of-source builds ([#600](https://github.com/FLAMEGPU/FLAMEGPU2/issues/600))
+ Performance improvements ([#564](https://github.com/FLAMEGPU/FLAMEGPU2/issues/564))
+ Compiler warning fixes and suppression ([#554](https://github.com/FLAMEGPU/FLAMEGPU2/issues/554), [#638](https://github.com/FLAMEGPU/FLAMEGPU2/pull/638), [#671](https://github.com/FLAMEGPU/FLAMEGPU2/pull/671))
+ Do not use `cudaEvent_t` based timers when using `WDDM` GPUs ([#640](https://github.com/FLAMEGPU/FLAMEGPU2/pull/640))
+ Visualiser: GLU no longer required as a dependency of the visualisation ([FLAMEGPU/FLAMEGPU2-visualiser#79](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/79))
+ Visualiser: CMake improvements ([FLAMEGPU/FLAMEGPU2-visualiser](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser) [#77](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/77), [#80](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/80), [#81](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/81))

### Removed

+ CMake versions <= 3.18 are no longer supported ([#661](https://github.com/FLAMEGPU/FLAMEGPU2/pull/661))
+ Do not suggest that Clang is a working/valid host C++ compiler at this time ([#633](https://github.com/FLAMEGPU/FLAMEGPU2/issues/633))
+ `pyflamegpu` no longer exposes `pyflamegpu.sys` and `pyflamegpu.os` ([#654](https://github.com/FLAMEGPU/FLAMEGPU2/issues/654))
+ `CUDAEnsemble::CUDAEnsemble`/`CUDAEnsemble::initialise` no longer output the FLAMEGPU version number ([#656](https://github.com/FLAMEGPU/FLAMEGPU2/issues/656), [#665](https://github.com/FLAMEGPU/FLAMEGPU2/pull/665))
+ `pyflamegpu.CUDAEnsemble().getConfig()` removed, use `pyflamegpu.CUDAEnsemble.Config()` ([#656](https://github.com/FLAMEGPU/FLAMEGPU2/issues/656), [#665](https://github.com/FLAMEGPU/FLAMEGPU2/pull/665))

### Fixed

+ Improved RTC compilation errors using `#line` directives ([#608](https://github.com/FLAMEGPU/FLAMEGPU2/issues/608))

## [2.0.0-alpha] - 2021-08-10

Initial alpha release of FLAME GPU 2.0.0, a CUDA C++ / python3 library for agent based simulations

[Unreleased]: https://github.com/FLAMEGPU/FLAMEGPU/compare/v2.0.0-alpha...HEAD
<!-- [2.0.0-alpha.2]: https://github.com/FLAMEGPU/FLAMEGPU/compare/v2.0.0-alpha.1...v2.0.0-alpha.2 -->
[2.0.0-alpha.1]: https://github.com/FLAMEGPU/FLAMEGPU/compare/v2.0.0-alpha...v2.0.0-alpha.1
[2.0.0-alpha]: https://github.com/FLAMEGPU/FLAMEGPU/releases/tag/v2.0.0-alpha