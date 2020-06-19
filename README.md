### Introduction

The project aim is to develop an existing computer application called FLAMEGPU, a high performance Graphics Processing Unit (GPU) extension to the FLAME framework that provides a mapping between a formal agent specifications with C based scripting and optimised CUDA code to support scalable muti-agent simulation.  The plan it to expand/extend features and capabilities of the exiting <a href="https://github.com/FLAMEGPU/FLAMEGPU">FLAMEGPU software</a>, in order to position it as a middle-ware for complex systems simulation. The application is expected to be the equivalent of Thrust for simplifying complex systems on GPUs.  The FLAMEGPU library includes algorithms such as spatial partitioning, network communication.

The Code is currently under active development and should **not be used** until the first release.

### Continuous Integration

Continuous Integration (CI) is provided by Github Actions, Travis (Linux only) and AppVeyor (Windows only).
CI jobs *only* include compilation, as the CI workers do not include CUDA GPUs.

| Provider           | Status |
|--------------------|--------|
| Github Actions     | [![Ubuntu](https://github.com/FLAMEGPU/FLAMEGPU2_dev/workflows/Ubuntu/badge.svg?branch=master)](https://github.com/FLAMEGPU/FLAMEGPU2_dev/actions?query=workflow%3AUbuntu+branch%3Amaster) [![Windows](https://github.com/FLAMEGPU/FLAMEGPU2_dev/workflows/Windows/badge.svg?branch=master)](https://github.com/FLAMEGPU/FLAMEGPU2_dev/actions?query=workflow%3AWindows+branch%3Amaster) [![Lint](https://github.com/FLAMEGPU/FLAMEGPU2_dev/workflows/Lint/badge.svg?branch=master)](https://github.com/FLAMEGPU/FLAMEGPU2_dev/actions?query=workflow%3ALint+branch%3Amaster) [![Docs](https://github.com/FLAMEGPU/FLAMEGPU2_dev/workflows/Docs/badge.svg?branch=master)](https://github.com/FLAMEGPU/FLAMEGPU2_dev/actions?query=workflow%3ADocs+branch%3Amaster) |
| Travis (Ubuntu)     | [![Build Status](https://travis-ci.org/FLAMEGPU/FLAMEGPU2_dev.svg?branch=master)](https://travis-ci.org/FLAMEGPU/FLAMEGPU2_dev)|
| Appveyor (Windows) | [![Build status](https://ci.appveyor.com/api/projects/status/4p58gnu8tyj7y3a7/branch/master?svg=true)](https://ci.appveyor.com/project/mondus/flamegpu2-dev/branch/master) |


### Dependencies

The dependencies below are required for building FLAME GPU 2.

Only documentation can be built without the required dependencies (however Doxygen is still required).

#### Required

* [CMake](https://cmake.org/) >= 3.12
  * CMake 3.16 is known to have issues on certain platforms
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) >= 9.0
* *Linux:*
  * [make](https://www.gnu.org/software/make/)
  * gcc/g++ >= 6 (version requirements [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements))
      * gcc/g++ >= 7 required for the test suite 
* *Windows:*
  * Visual Studio 2015 or higher (2019 preferred)

#### Optional
* [cpplint](https://github.com/cpplint/cpplint): Required for linting code
* [Doxygen](http://www.doxygen.nl/): Required for building documentation
* [git](https://git-scm.com/): Required by CMake for preparing GoogleTest, to build the test suite

### Building FLAME GPU 2

FLAME GPU 2 uses [CMake](https://cmake.org/), as a cross-platform process, for configuring and generating build directives, e.g. `Makefile` or `.vcxproj`. This is used to build the FLAMEGPU2 library, examples, tests and documentation.

#### Linux

Under Linux, `cmake` can be used to generate makefiles specific to your system:

```
mkdir -p build && cd build
cmake .. 
make -j8
```

The option `-j8` enables parallel compilation using upto 8 threads, this is recommended to improve build times.

By default a `Makefile` for the `Release` build configuration will be generated.

Alternatively, using `-DCMAKE_BUILD_TYPE=`, `Debug` or `Profile` build configurations can be generated:
 
```
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Profile
make -j8
```

#### Windows

*Note: If installing CMake on Windows ensure CMake is added to the system path, allowing `cmake` to be used via `cmd`, this option is disabled within the installer by default.*

When generating Visual studio project files, using `cmake` (or `cmake-gui`), the platform **must** be specified as `x64`.

Using `cmake` this takes the form `-A x64`:

```
mkdir build && cd build
cmake .. -A x64
FLAMEGPU2.sln
```

This command will use the latest version of Visual studio detected, and open the created Visual studio solution.
If the `cmake` command fails, the detected version does not support CUDA. Reinstalling CUDA may correct this issue.

Alternatively using `-G` the desired version of Visual studio can be specified:

```
mkdir build && cd build
cmake .. -G "Visual Studio 14 2015" -A x64
FLAMEGPU2.sln
```

`Visual Studio 14 2015` can be replaced with any supported [Visual studio generator](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#visual-studio-generators) that is installed.

#### Configuring CMake

The following options are available when calling `cmake` on either Linux or Windows. It is also possible to manage these options via `cmake-gui` on Windows, often by setting the variable of the same name.

*The examples in the following sections use Linux syntax, it is necessary to include the `-A` (and `-G`) arguments from above when calling `cmake` on Windows. Similarly `.vcxproj` and `.sln` files will be generated rather than `Makefile`.*

##### Individual examples

Build directives for individual examples can be built from their respective directories

```
cd examples/mas/
mkdir -p build && cd build
cmake ..
```

##### Visualisation

By default FGPU2 will not build visualisations, however this can be enabled with the cmake option `-DVISUALISATION=ON`. This also introduces the `VISUALISATION` preprocessor macro to all projects, so that optional visualisation code can be enabled.

On Windows this will then automatically download the visualisation repository and all dependencies, as such the initial CMake configuration after enabling visualisation may take longer than expected. 
On Linux you may need to install additional packages, refer to the CMake configure output for clarification of which packages are missing from your system.

**Visualisation Dependencies:**
* [SDL](https://www.libsdl.org/)
* [GLM](http://glm.g-truc.net/) *(consistent C++/GLSL vector maths functionality)*
* [GLEW](http://glew.sourceforge.net/) *(GL extension loader)*
* [FreeType](http://www.freetype.org/)  *(font loading)*

The visualisation codebase can be found at [this location](https://github.com/FLAMEGPU/FLAMEGPU2_visualiser) and is a fork of [sdl_exp](https://github.com/Robadob/sdl_exp).

##### Testing

The test suite can be built from the root directory using `-DBUILD_TESTS=ON`:

```
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make -j8
```

The first time CMake is configured with `-DBUILD_TESTS=ON` an internet connection is required, as [GoogleTest](https://github.com/google/googletest) is downloaded and built. 
Subsequent reconfigures will attempt to update this copy, but will continue if updating fails.
Automatic updating of GoogleTest can be disabled by passing `-DBUILD_TESTS=OFF`.

*Known Issues:* The tests do not build under the combination of Visual Studio 2015 and CUDA 9.0 or 9.1. Use CUDA 9.2 or newer if you require the tests.

GoogleTest runtime documentation can be found [here](https://github.com/google/googletest/blob/master/googletest/docs/advanced.md).
In particular `--gtest_catch_exceptions=0` may be useful during test development, so that unhandled exceptions are passed straight to the debugger.

##### Documentation

If you wish to build the documentation, [doxygen](http://www.doxygen.nl/) must be available on your system.

A build file for documentation is generated at the same time as build files for the flamegpu2 library.

This generates the target `docs` which can be called with `make`.

```
cd src
mkdir -p build && cd build
cmake ..
make docs
```

*When using Visual Studio, the target `docs` will appear as a project within the created solution.*

**Building Documentation Without CUDA**
If you have Doxygen and don't have the CUDA compiler, CMake will generate a minimal project for only building the documentation.

On Linux the standard documentation command, shown above, can be used if `make` is available.

If `make` is unavailable, *e.g. on Windows*, `make docs` can be replaced with a direct call to `doxygen`:

```
doxygen Doxyfile.docs
```

##### Device Architectures

CUDA device architectures can be specified via `-DCUDA_ARCH` when generating make files, using a semi colon, space or comma separated list of compute capability numbers. I.e to build for just SM_61 and SM_70:

```
mkdir -p build && cd build
cmake .. -DCUDA_ARCH="61;70"
make -j8
```

Pass `-DCUDA_ARCH=` to reset to the default.

##### NVTX Markers

[NVTX markers](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx) can be enabled to improve the profiling experience, using the `USE_NVTX` Cmake option. 

I.e. `-DUSE_NVTX=ON` will enable NVTX markers and allow custom markers to be included. This is implied by the `Profile` build configuration. 
See `include/util/nvtx.h` and the associated documentation for how to apply custom markers.

### Running FLAME GPU 2
Examples for FLAME GPU 2 build to `<cmake_build_path>/bin/<operating_system>-x64/<Debug|Release|Profile>/`, and can be executed from that directory via command line.

If wishing to run examples within Visual Studio it is necessary to right click the desired example in the Solution Explorer and select `Debug > Start New Instance`. Alternatively, if `Set as StartUp Project` is selected, the main debugging menus can be used to initiate execution. To configure command line argument for execution within Visual Studio, right click the desired example in the Solution Explorer and select `Properties`, in this dialog select `Debugging` in the left hand menu to display the entry field for `command arguments`. Note, it may be necessary to change the configuration as the properties dialog may be targeting a different configuration to the current build configuration.

#### Runtime compilation of Agent Functions

FLAME GPU 2 supports runtime compilation of agent functions. E.g. specification of agent functions, or agent function conditions, as strings. This is a prerequisite for other language support via [SWIG](http://www.swig.org/). In order to test RTC features (in `tests/test_cases/runtime/test_rtc_device_api.cu`) you must set the following environment variables.

1. `FLAMEGPU2_INC_DIR` This should be set to the `\include` directory of the main FLAMEGPU2 directory.
2. `CUDA_PATH` This may be set when installing CUDA, however if it does not exist or has been changed this then it **must** point to the CUDA installation folder of the CUDA version used to compile FLAMEGPU (e.g. the one detected by CMake).
