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

## [2.0.0-rc.1] - 2024-01-12

### Added

+ Support for CUDA 12.0-12.3. ([#1015](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1015), [#1056](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1056), [#1097](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1097), [#1130](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1130))
  + CUDA 12.2+ currently suffers from poor RTC compilation times due to changes in the CUDA headers ([#1118](https://github.com/FLAMEGPU/FLAMEGPU2/issues/1118)). This will be fixed in a future release.
+ Support for Python 3.12. ([#1117](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1117))
+ Visualiser: Add support for orthographic projection. ([FLAMEGPU/FLAMEGPU2-visualiser#114](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/114), [FLAMEGPU/FLAMEGPU2-visualiser#121](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/121) [#1040](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1040))
+ Visualiser: Agents can be hidden according to their state. ([#1041](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1041))
+ Declare/define Agent and Host function shims. ([#1049](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1049))
+ Poisson distribution support for random APIs. ([#1060](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1060))
+ `HostAPI` returns config structures and ensemble run index. ([#1082](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1082))
+ `pyflamegpu` installable from custom pip wheelhouse [whl.flamegpu.com](https://whl.flamegpu.com) . ([#645](https://github.com/FLAMEGPU/FLAMEGPU2/issues/645))
+ Environment macro properties can now be imported/exported. ([#1087](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1087))
+ Readme within `examples/` documenting the available examples.
+ `DeviceAPI::isAgent()`, `DeviceAPI::isState()`. ([#1116](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1116), [#1139](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1139))
+ Agent python codegen will now capture external variables with corresponding attribute. ([#1147](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1147))
+ Added `RunPlanVector::setPropertyStep()` ([#1152](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1152))
+ Added directed grraph support, via `EnvironmentDirectedGraphDescription` which can then be used with `MessageBucket` for on-graph communication. ([#1089](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1089))
+ Added optional Distributed Ensemble support (MPI). ([#1090](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1090))

### Changed (Breaking)

+ `CUDAEnsemble::getLogs` returns `std::map<unsigned int, RunLog>` rather than `std::vector<RunLog>`, required for distributed ensemble support. ([#1090](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1090))

### Changed

+ Improved dependency find logic. ([#1015](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1015))
+ Improved telemetry message when configuring CMake. ([#1030](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1030))
+ Improved robustness of context creation testing. ([#1096](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1096))
+ Improved (experimental) GLM support within python API. ([#1074](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1074))
+ Various CI changes. ([#1015](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1015), [#1036](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1036), [#1044](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1044), [#1062](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1062), [#1090](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1090), [#1097](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1097), [#1100](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1100), [#1102](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1102), [#1117](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1117), [#1130](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1130), [#1138](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1138), [#1140](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1140))

### Fixed

+ `FLAMEGPU_ENABLE_GLM` was incorrectly documented in README. ([#1033](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1033))
+ ManyLinux wheels did not have BuildNumber set. ([#1036](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1036))
+ arm/Tegra compilation errors. ([#1039](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1039), [FLAMEGPU/FLAMEGPU2-visualiser#119](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/119))
+ `AgentRandom::uniform<int>()` would never return max bound. ([#411](https://github.com/FLAMEGPU/FLAMEGPU2/pull/411))
+ `AgentRandom::uniform<float>()` would return range `(0, 1]`. ([#411](https://github.com/FLAMEGPU/FLAMEGPU2/pull/411))
+ `AgentRandom::uniform<double>()` would return range `(0, 1]`. ([#411](https://github.com/FLAMEGPU/FLAMEGPU2/pull/411))
+ Visualiser: Draw would lose data when resizing large line art. ([FLAMEGPU/FLAMEGPU2-visualiser#118](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/118))
+ Visualiser: Begin paused now pauses at the first frame with agents. ([FLAMEGPU/FLAMEGPU2-visualiser#120](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/120), [#1046](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1046))
+ Resolved crash where no optional messages were output first step. ([#1054](https://github.com/FLAMEGPU/FLAMEGPU2/issues/1054))
+ Old version of Curl would write to stdout during telemetry. ([#1027](https://github.com/FLAMEGPU/FLAMEGPU2/issues/1027))
+ Agent python codegen would fail to translate math functions inside message loops. ([#1077](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1077))
+ Resolved missing messages from python host function exceptions thrown during ensembles. ([#1067](https://github.com/FLAMEGPU/FLAMEGPU2/issues/1067))
+ Various Telemetry fixes. ([#1035](https://github.com/FLAMEGPU/FLAMEGPU2/issues/1035), [#1098](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1098), [#1099](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1099), [#1079](https://github.com/FLAMEGPU/FLAMEGPU2/issues/1079))
+ Various CMake fixes. ([#1071](https://github.com/FLAMEGPU/FLAMEGPU2/issues/1071), [#1092](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1092), [#1062](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1062), [#1113](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1113))
+ Resolved nested venvs within Windows Python Wheels. ([#998](https://github.com/FLAMEGPU/FLAMEGPU2/issues/998))
+ Agent states loaded from file could be ignored. ([#1093](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1093))
+ Agent python codegen did not support standalone message variables. ([#1110](https://github.com/FLAMEGPU/FLAMEGPU2/issues/1110), [#1143](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1143))
+ Agent python codegen did not support `int`/`float` casts. ([#1143](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1143))
+ Resolved floating point cast to enum error. ([#1148](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1148))
+ `JitifyCache` is now exposed via the python API. ([#1151](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1151))
+ Agent python codegen did not support `ID` to `id_t` conversion. ([#1153](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1153))
+ Spatial Messaging interaction radius was incorrect when the requested radius was not a factor of the environment width ([#1160](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1160))
+ `astpretty` no longer a dependency of `pyflamegpu` ([#1166](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1166))
+ Agent python codegen did not correctly account for variable scope ([#1125](https://github.com/FLAMEGPU/FLAMEGPU2/issues/1125), [#1127](https://github.com/FLAMEGPU/FLAMEGPU2/issues/1127))
+ Resolve cmake_minimum_version deprecations by updating and patching dependencies (GoogleTest, RapidJSON and CMakeRC) ([1168](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1168))
+ Fix docstrings associated with the `flamegpu` namespace ([#1169](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1169))
+ Fix unused result warning(s) issued with clang as the host compiler ([1170](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1170))
+ Include `<cstdint>` in `DiscreteColor.h` for gcc 12.3 ([#1171](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1171)). Thanks to [Daniele Baccege](https://github.com/daniele-baccega)

## [2.0.0-rc] - 2022-12-13

### Added
+ `CUDASimulation::simulate()` can now be passed a RunPlan. ([#678](https://github.com/FLAMEGPU/FLAMEGPU2/issues/678))
+ CUDA 11.7 is now included in CI builds ([#761](https://github.com/FLAMEGPU/FLAMEGPU2/issues/761), [#856](https://github.com/FLAMEGPU/FLAMEGPU2/issues/856))
+ `CUDASimulation::setEnvironmentProperty()`, `CUDASimulation::getEnvironmentProperty()` ([#760](https://github.com/FLAMEGPU/FLAMEGPU2/pull/760))
+ Added `HostAgentAPI` mean and standard deviation operators. ([#764](https://github.com/FLAMEGPU/FLAMEGPU2/pull/764), [#766](https://github.com/FLAMEGPU/FLAMEGPU2/pull/766))
+ Added file IO support for all config struct members. ([#768](https://github.com/FLAMEGPU/FLAMEGPU2/issues/768))
+ Added uniform random range support for floating-point types. ([#776](https://github.com/FLAMEGPU/FLAMEGPU2/issues/776))
+ Added getOffsetX/Y/Z() to the iterated message for array message types. ([#781](https://github.com/FLAMEGPU/FLAMEGPU2/pull/781))
+ Added VonNeumann neighbourhood iteration to `MessageArray2D` and `MessageArray3D`. ([#783](https://github.com/FLAMEGPU/FLAMEGPU2/pull/783))
+ Added wrapped iteration access to `MessageSpatial2D` and `MessageSpatial3D`. ([#185](https://github.com/FLAMEGPU/FLAMEGPU2/issues/185))
+ Log files can now be configured to include timing data. ([#799](https://github.com/FLAMEGPU/FLAMEGPU2/pull/799))
+ RTC users may now specify include paths. ([#801](https://github.com/FLAMEGPU/FLAMEGPU2/pull/801))
+ Added annotations to CI ([#480](https://github.com/FLAMEGPU/FLAMEGPU2/issues/480))
+ Added three levels of error-reporting to choose from when using `CUDAEnsemble`. ([#839](https://github.com/FLAMEGPU/FLAMEGPU2/pull/839))
+ Added `VERBOSE_PTXAS` CMake option. ([#851](https://github.com/FLAMEGPU/FLAMEGPU2/pull/851))
+ Visual Studio 2022 is now included in CI builds ([#866](https://github.com/FLAMEGPU/FLAMEGPU2/pull/866))
+ Visualiser: Agent array variables can now be used to control agent color. ([#876](https://github.com/FLAMEGPU/FLAMEGPU2/pull/876), [FLAMEGPU/FLAMEGPU2-visualiser#90](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/90))
+ Visualiser: Added two low-poly stock models (`PYRAMID`, `ARROWHEAD`). ([FLAMEGPU/FLAMEGPU2-visualiser#91](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/91))
+ Visualiser: Agents can now be represented by Keyframe pair animated models. ([#904](https://github.com/FLAMEGPU/FLAMEGPU2/pull/904), [FLAMEGPU/FLAMEGPU2-visualiser#16](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/issues/16))
+ Added Pedestrian Navigation example in a standalone repository (from FLAME GPU 1). ([Example](https://github.com/FLAMEGPU/FLAMEGPU2-pedestrian_navigation-example))
+ Added support for agent functions and function conditions to be written in a "pure python" syntax. ([#882](https://github.com/FLAMEGPU/FLAMEGPU2/pull/882), [#910](https://github.com/FLAMEGPU/FLAMEGPU2/pull/910), [#917](https://github.com/FLAMEGPU/FLAMEGPU2/pull/917))
+ Added "pure python" wrapped boids example. ([#882](https://github.com/FLAMEGPU/FLAMEGPU2/pull/882), [#940](https://github.com/FLAMEGPU/FLAMEGPU2/pull/940), [#958](https://github.com/FLAMEGPU/FLAMEGPU2/pull/958))
+ Visualiser: User interfaces can now be defined to control environment properties via visualisations. ([#911](https://github.com/FLAMEGPU/FLAMEGPU2/pull/911), [FLAMEGPU/FLAMEGPU2-visualiser#100](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/100))
+ A warning is now emitted when configuring CMake for Visual Studio 2022 if the build path contains a space. ([#934](https://github.com/FLAMEGPU/FLAMEGPU2/pull/934))
+ Python 3.11 is now included in CI builds and wheel generation. ([#944](https://github.com/FLAMEGPU/FLAMEGPU2/pull/944))
+ Сompute Capability 90 (Hopper) has been added to the list of default CUDA architectures. ([#954](https://github.com/FLAMEGPU/FLAMEGPU2/pull/954))
+ CUDAEnsemble now prevents standby during execution (on Windows), this can be disabled. ([#930](https://github.com/FLAMEGPU/FLAMEGPU2/pull/930))
+ Added `util::cleanup()` for triggering `cudaDeviceReset()`. ([974](https://github.com/FLAMEGPU/FLAMEGPU2/pull/974), also see [#950](https://github.com/FLAMEGPU/FLAMEGPU2/pull/950))
+ Message list persistence can now be configured per message type. ([#973](https://github.com/FLAMEGPU/FLAMEGPU2/pull/973))
+ `pyflamegpu_swig` build target now depends on `flamegpu` headers. ([#981](https://github.com/FLAMEGPU/FLAMEGPU2/pull/981))
+ Added `RunPlan::operator==()`, `RunPlanVector::operator==()` and `RunPlanVector::at()`. ([#983](https://github.com/FLAMEGPU/FLAMEGPU2/pull/983))
+ Added `--truncate` argument to CUDASimulation and CUDAEnsemble, allowing output files to truncate (defaults to off) ([#992](https://github.com/FLAMEGPU/FLAMEGPU2/pull/992))
+ Added CTest support for test suite execution ([#285](https://github.com/FLAMEGPU/FLAMEGPU2/pull/285))
+ Added `util::clearRTCDiskCache()` for clearing JitifyCache on-disk ([#999](https://github.com/FLAMEGPU/FLAMEGPU2/pull/999))
+ Added Telemetry allowing the collection of usage metrics, this can be disabled via several methods ([#987](https://github.com/FLAMEGPU/FLAMEGPU2/pull/987), [#991](https://github.com/FLAMEGPU/FLAMEGPU2/pull/991)), ([#1013](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1013))

### Changed (Breaking)
+ Removed redundant length argument from many Python methods. ([#831](https://github.com/FLAMEGPU/FLAMEGPU2/issues/831), [#872](https://github.com/FLAMEGPU/FLAMEGPU2/pull/872))
+ Replaced default random engine with `std::mt19937_64`. ([#754](https://github.com/FLAMEGPU/FLAMEGPU2/issues/754))
+ `CUDASimulation::initialise()` now allows you set defaults, matching the behaviour of `CUDAEnsemble`. ([#755](https://github.com/FLAMEGPU/FLAMEGPU2/issues/755))
+ Renamed `ModelVis::addStaticModel()` to `ModelVis::newStaticModel()`. ([#911](https://github.com/FLAMEGPU/FLAMEGPU2/pull/911))
+ Default CUDA random engine changed to PHILOX (from XORWOW). ([#873](https://github.com/FLAMEGPU/FLAMEGPU2/pull/873))
+ Renamed `DeviceAPI::getThreadIndex()` to `DeviceAPI::getIndex()`. ([#943](https://github.com/FLAMEGPU/FLAMEGPU2/pull/943))
+ Missing pip packages are nolonger automatically installed during CMake configure. ([#935](https://github.com/FLAMEGPU/FLAMEGPU2/pull/935))
+ `cudaDeviceReset()` is nolonger automatically triggered at `CUDASimulation`/`CUDAEnsemble` exit. ([#950](https://github.com/FLAMEGPU/FLAMEGPU2/pull/950))
+ Unrecognised runtime args will nolonger cause program exit. ([#967](https://github.com/FLAMEGPU/FLAMEGPU2/pull/967))
+ JSON output now outputs NaN/Inf values as string. ([#969](https://github.com/FLAMEGPU/FLAMEGPU2/pull/969))
+ Removed references from return values throughout model description API. ([#952](https://github.com/FLAMEGPU/FLAMEGPU2/pull/952), [#978](https://github.com/FLAMEGPU/FLAMEGPU2/pull/978), [#980](https://github.com/FLAMEGPU/FLAMEGPU2/pull/980), [#1004](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1004))
+ Message lists nolonger persist (by default) between iterations. ([#973](https://github.com/FLAMEGPU/FLAMEGPU2/pull/973))
+ Renamed `RunPlanVector::setPropertyUniformDistibution()` to `RunPlanVector::setPropertyLerpRange()` ([#983](https://github.com/FLAMEGPU/FLAMEGPU2/pull/983))
+ Replaced NVTX macros with constexpr + namespaced methods ([#990](https://github.com/FLAMEGPU/FLAMEGPU2/pull/990))
+ CUDAEnsemble now raises an exception of log files already exist (previous behaviour would append) ([#818](https://github.com/FLAMEGPU/FLAMEGPU2/issues/818), [#992](https://github.com/FLAMEGPU/FLAMEGPU2/pull/992))
+ Removed 'Callback' from Python API host function method/class names [#997](https://github.com/FLAMEGPU/FLAMEGPU2/pull/997)
+ Renamed `CUDAMessage::getMessageDescription()` to `getMessageData()` [#996](https://github.com/FLAMEGPU/FLAMEGPU2/pull/996)
+ CMake variables were updated to begin `FLAMEGPU_` ([#991](https://github.com/FLAMEGPU/FLAMEGPU2/pull/991))
+ Removed `cuda_arch` CMake variable, `CMAKE_CUDA_ARCHITECTURES` should now be used instead ([#991](https://github.com/FLAMEGPU/FLAMEGPU2/pull/991))
+ Improved organisation of files within include/src/tests ([#1007](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1007), [#1012](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1012))
+ Removed `CUDASimulation::getAgent()`, `getCUDAAgent()`, `getCUDAMessage()` from the public API. ([#1007](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1007))
+ Improved organisation/naming of examples ([#1010](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1010))
+ Thrust/CUB minimum supported version increased to 1.16.0, from 1.14.0 due for improved windows support and bugfixes. 1.17.2 is fetched via CMake if a compatible thrust/cub is not found. ([#1008](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1008))

### Changed
+ Suppress note emitted by GCC >= 10 on ppc64le about changes since GCC 5. ([#757](https://github.com/FLAMEGPU/FLAMEGPU2/issues/757))
+ Improved how input file loading errors and warnings are handled. ([#752](https://github.com/FLAMEGPU/FLAMEGPU2/issues/752), [#759](https://github.com/FLAMEGPU/FLAMEGPU2/pull/759), [#810](https://github.com/FLAMEGPU/FLAMEGPU2/pull/810))
+ Visualiser: Updated FreeType dependency, hopefully improving download stability. ([FLAMEGPU/FLAMEGPU2-visualiser#86](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/86))
+ Improve API docs for FLAMEGPU macros. ([#787](https://github.com/FLAMEGPU/FLAMEGPU2/pull/787))
+ Agent sorting has been extended to submodels and agents with coordinates in array variables. ([#805](https://github.com/FLAMEGPU/FLAMEGPU2/pull/805), [#854](https://github.com/FLAMEGPU/FLAMEGPU2/pull/854))
+ `USE_GLM` type checking is now able to convert GLM types to base type/length. ([#809](https://github.com/FLAMEGPU/FLAMEGPU2/issues/809))
+ Greatly reduced default stream usage, improving `CUDAEnsemble` performance. ([#838](https://github.com/FLAMEGPU/FLAMEGPU2/pull/838), [#843](https://github.com/FLAMEGPU/FLAMEGPU2/pull/843))
+ NVRTC is now passed the maximum supports GPU architecture flag. ([#844](https://github.com/FLAMEGPU/FLAMEGPU2/issues/844))
+ Curve is now stored in shared_memory, improving register usage in CUDA 11.3+. ([#560](https://github.com/FLAMEGPU/FLAMEGPU2/issues/560), [#571](https://github.com/FLAMEGPU/FLAMEGPU2/issues/571))
+ NVRTC is now passed the maximum supports GPU architecture flag. ([#844](https://github.com/FLAMEGPU/FLAMEGPU2/issues/844))
+ `-lineinfo` is now passed to the `MinSizeRel` and `RelWithDebInfo` build configurations. ([#798](https://github.com/FLAMEGPU/FLAMEGPU2/issues/798))
+ Various test improvements. ([#860](https://github.com/FLAMEGPU/FLAMEGPU2/pull/860), [#902](https://github.com/FLAMEGPU/FLAMEGPU2/pull/902), [#908](https://github.com/FLAMEGPU/FLAMEGPU2/pull/908), [#1002](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1002), [#1000](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1000))
+ Improved how `CUDAEnsemble` reports failure to find CUDA device to match `CUDASimulation`. ([#858](https://github.com/FLAMEGPU/FLAMEGPU2/issues/858))
+ Ubuntu CI has been updated to use Ubuntu 22.04 / GCC 11 ([#877](https://github.com/FLAMEGPU/FLAMEGPU2/pull/877))
+ Improved granularity of pyflamegpu incremental builds. ([#887](https://github.com/FLAMEGPU/FLAMEGPU2/pull/887))
+ Improved error message when multiple agents write to the same array message element. ([#895](https://github.com/FLAMEGPU/FLAMEGPU2/pull/895))
+ CI now uses CUDA 11.8 as "latest" ([#924](https://github.com/FLAMEGPU/FLAMEGPU2/pull/924))
+ Visualisation headers are now always linted, regardless of whether enabled at CMake type. ([#919](https://github.com/FLAMEGPU/FLAMEGPU2/pull/919))
+ Boids examples were updated to demonstrate visualisation UIs. ([#911](https://github.com/FLAMEGPU/FLAMEGPU2/pull/911))
+ CUDA random engine may now be selected during CMake configuration. ([#873](https://github.com/FLAMEGPU/FLAMEGPU2/pull/873))
+ Updated pinned versions of external GitHub Actions. ([#945](https://github.com/FLAMEGPU/FLAMEGPU2/pull/945))
+ Renamed Python boids examples. ([#940](https://github.com/FLAMEGPU/FLAMEGPU2/pull/940))
+ Removed redundant references from function argument throughout API. ([#946](https://github.com/FLAMEGPU/FLAMEGPU2/pull/946))
+ Unified generic `size_type` to a library-wide version. ([#948](https://github.com/FLAMEGPU/FLAMEGPU2/pull/948))
+ Improved granularity of verbosity levels. ([960](https://github.com/FLAMEGPU/FLAMEGPU2/pull/960))
+ Added `silence_unknown_args` to runtime arg parsing so that users are nolonger required to filter out bespoke args. ([#967](https://github.com/FLAMEGPU/FLAMEGPU2/pull/967))
+ Removed resolved issues from README.md. ([#994](https://github.com/FLAMEGPU/FLAMEGPU2/pull/994))
+ Removed outdated comment from version.h. ([#985](https://github.com/FLAMEGPU/FLAMEGPU2/pull/985))

### Removed
+ Removed redundant mutexes around RTC kernel launches. ([#469](https://github.com/FLAMEGPU/FLAMEGPU2/issues/469))
+ CUDA 10.x is nolonger supported ([#611](https://github.com/FLAMEGPU/FLAMEGPU2/issues/611), [FLAMEGPU/FLAMEGPU2-visualiser#89](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/89))
+ C++ 14 is nolonger supported ([#611](https://github.com/FLAMEGPU/FLAMEGPU2/issues/611), [FLAMEGPU/FLAMEGPU2-visualiser#89](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/89))
+ Remove unused RTCSafeCudaMemcpyToSymbol/Address methods. ([#878](https://github.com/FLAMEGPU/FLAMEGPU2/pull/878))
+ Python 3.6 wheels are nolonger generated by Release CI. ([#925](https://github.com/FLAMEGPU/FLAMEGPU2/pull/925))
+ Removed boids_bruteforce_dependency_graph example. ([#937](https://github.com/FLAMEGPU/FLAMEGPU2/pull/937))

### Fixed
+ Python interface support for `CUDAEnsembleConfig::devices`. ([#682](https://github.com/FLAMEGPU/FLAMEGPU2/issues/682))
+ `CUDAFatAgent` supports agents with no variables. ([#492](https://github.com/FLAMEGPU/FLAMEGPU2/issues/492))
+ `DeviceAgentVector` can nolonger be passed out of scope. ([#522](https://github.com/FLAMEGPU/FLAMEGPU2/issues/522))
+ `EnvironmentManager::setProperty()`, `EnvironmentManager::getProperty()` did not check length. ([#760](https://github.com/FLAMEGPU/FLAMEGPU2/pull/760))
+ Logging could divide by zero when calculating standard deviation on empty agent population. ([#763](https://github.com/FLAMEGPU/FLAMEGPU2/pull/763))
+ Updated Jitify dependency (fixes memory leak, improves GLM support). ([#756](https://github.com/FLAMEGPU/FLAMEGPU2/issues/756), [#813](https://github.com/FLAMEGPU/FLAMEGPU2/pull/813))
+ Corrected a sugar growback bug within the SugarScape example. ([#784](https://github.com/FLAMEGPU/FLAMEGPU2/issues/784))
+ `MessageArray3D` Moore iterator could lead to compilation failure. ([#785](https://github.com/FLAMEGPU/FLAMEGPU2/issues/785))
+ Visualiser: Did not account for agent populations shrinking. ([FLAMEGPU/FLAMEGPU2-visualiser#785](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/87))
+ `EnvironmentManager::setProperty()` had length check misplaced under `USE_GLM`. ([#791](https://github.com/FLAMEGPU/FLAMEGPU2/issues/791))
+ `CUDAEnsemble` logs did not include `RunPlan` details as intended. ([#799](https://github.com/FLAMEGPU/FLAMEGPU2/pull/799))
+ XML exit log contained redundant block. ([#799](https://github.com/FLAMEGPU/FLAMEGPU2/pull/799))
+ Final step log is nolonger double logged. ([#799](https://github.com/FLAMEGPU/FLAMEGPU2/pull/799))
+ Greatly improve RTC compile times by specifying known headers ([#811](https://github.com/FLAMEGPU/FLAMEGPU2/pull/811))
+ `RunPlan::setProperty()` would fail silently. ([#814](https://github.com/FLAMEGPU/FLAMEGPU2/pull/814))
+ Internal environment property used for tracking steps was being mapped to submodels. ([#815](https://github.com/FLAMEGPU/FLAMEGPU2/issues/815))
+ `RunPlanVector::setPropertyUniformDistribution()` was rounding floating-point values. ([#823](https://github.com/FLAMEGPU/FLAMEGPU2/pull/823))
+ `cbrt()` was incorrectly used in place of `sqrtf()` in Circles example. ([#829](https://github.com/FLAMEGPU/FLAMEGPU2/pull/829))
+ Visualiser: `AgentStateVis::setColour()` did not support `StaticColor`. ([#830](https://github.com/FLAMEGPU/FLAMEGPU2/pull/830))
+ Various CMake improvements ([#804](https://github.com/FLAMEGPU/FLAMEGPU2/pull/804), [#836](https://github.com/FLAMEGPU/FLAMEGPU2/pull/836), [#897](https://github.com/FLAMEGPU/FLAMEGPU2/pull/897), [FLAMEGPU/FLAMEGPU2-visualiser#95](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser/pull/95), [#914](https://github.com/FLAMEGPU/FLAMEGPU2/pull/914), [#921](https://github.com/FLAMEGPU/FLAMEGPU2/pull/921), [#991](https://github.com/FLAMEGPU/FLAMEGPU2/pull/991), [#1014](https://github.com/FLAMEGPU/FLAMEGPU2/pull/1014))
+ Updated CI to support new CUDA repository GPG keys. ([#841](https://github.com/FLAMEGPU/FLAMEGPU2/pull/841))
+ `DeviceMacroProperty::operator+=(double)` did not support SM < 60. ([#847](https://github.com/FLAMEGPU/FLAMEGPU2/issues/847))
+ Spatial agent sorting did not support agents outside the default state. ([#861](https://github.com/FLAMEGPU/FLAMEGPU2/issues/861))
+ `AgentStateVis::setColor()` did not validate suitability of agent variable. ([#875](https://github.com/FLAMEGPU/FLAMEGPU2/pull/875))
+ Improved how NVRTC's dll is located by pyflamegpu on Windows. ([#450](https://github.com/FLAMEGPU/FLAMEGPU2/issues/450))
+ `CUDAEnsemble` progress printing nolonger goes backwards. ([#901](https://github.com/FLAMEGPU/FLAMEGPU2/issues/901))
+ `visualiser::DiscreteColor` was not support by the Python API. ([#922](https://github.com/FLAMEGPU/FLAMEGPU2/pull/922))
+ Corrected typographic error inside `CITATION.cff`. ([#929](https://github.com/FLAMEGPU/FLAMEGPU2/pull/929))
+ Python API did not correctly support `CUDASimulation::setEnvironmentProperty()`. ([#915](https://github.com/FLAMEGPU/FLAMEGPU2/pull/916), [#912](https://github.com/FLAMEGPU/FLAMEGPU2/issues/912))
+ A warning is nolonger emit by `CUDAEnsemble` if the default config is not updated. ([#949](https://github.com/FLAMEGPU/FLAMEGPU2/pull/949))
+ Replaced occurrences of `CUDAAgentModel` with `CUDASimulation` in comments and NVTX ranges. ([#951](https://github.com/FLAMEGPU/FLAMEGPU2/pull/951))
+ Corrected issues with Python packaging. ([#962](https://github.com/FLAMEGPU/FLAMEGPU2/pull/962), [#964](https://github.com/FLAMEGPU/FLAMEGPU2/pull/964))
+ Messaging internal data structures are now correctly reset at `CUDASimulation` reset. ([#972](https://github.com/FLAMEGPU/FLAMEGPU2/pull/972))
+ Removed redundant code from `CUDAFatAgent::addSubAgent()`, which could lead to spurious device initialisation. ([#968](https://github.com/FLAMEGPU/FLAMEGPU2/pull/968))
+ Improve precision of included headers to fix GCC11 builds. ([#988](https://github.com/FLAMEGPU/FLAMEGPU2/pull/988))
+ `__disown__()` is now automatically triggered when Python Host functions/conditions are attached to a model. ([#975](https://github.com/FLAMEGPU/FLAMEGPU2/pull/975), [#997](https://github.com/FLAMEGPU/FLAMEGPU2/pull/997))


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

[Unreleased]: https://github.com/FLAMEGPU/FLAMEGPU2/compare/v2.0.0-rc.1...HEAD
[2.0.0-rc.1]: https://github.com/FLAMEGPU/FLAMEGPU2/releases/tag/v2.0.0-rc.1
[2.0.0-rc]: https://github.com/FLAMEGPU/FLAMEGPU2/releases/tag/v2.0.0-rc
[2.0.0-alpha.2]: https://github.com/FLAMEGPU/FLAMEGPU2/releases/tag/v2.0.0-alpha.2
[2.0.0-alpha.1]: https://github.com/FLAMEGPU/FLAMEGPU2/releases/tag/v2.0.0-alpha.1
[2.0.0-alpha]: https://github.com/FLAMEGPU/FLAMEGPU2/releases/tag/v2.0.0-alpha