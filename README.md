### Introduction

The project aim is to develop an existing computer application called FLAMEGPU, a high performance Graphics Processing Unit (GPU) extension to the FLAME framework that provides a mapping between a formal agent specifications with C based scripting and optimised CUDA code to support scalable muti-agent simulation.  The plan it to expand/extend features and capabilities of the exiting <a href="https://github.com/FLAMEGPU/FLAMEGPU">FLAMEGPU software</a>, in order to position it as a middle-ware for complex systems simulation. The application is expected to be the equivalent of Thrust for simplifying complex systems on GPUs.  The FLAMEGPU library includes algorithms such as spatial partitioning, network communication.

The Code is currently under active development and should **not be used** until the first release.

### Continuous Integration

Continuous integration is provided by Travis (Linux) and windows (AppVeyor). This performs only build tests and the virtual machines do not support executing the Boost unit tests. Each build has a script which is required to install the CUDA toolkit on the VM worker node. See the scripts folder for more details.

#### Current Master Branch Build Status

[![Build status](https://ci.appveyor.com/api/projects/status/4p58gnu8tyj7y3a7/branch/master?svg=true)](https://ci.appveyor.com/project/mondus/flamegpu2-dev/branch/master)

[![Build Status](https://travis-ci.org/FLAMEGPU/FLAMEGPU2_dev.svg?branch=master)](https://travis-ci.org/FLAMEGPU/FLAMEGPU2_dev)

### Dependencies

The dependencies below are required for building FLAME GPU 2.

Only documentation can be built without the required dependencies (however Doxygen is still required).

#### Required

* [CMake](https://cmake.org/) >= 3.12
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) >= 1.54

**Linux**
* [make](https://www.gnu.org/software/make/)
* gcc (version requirements [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements))

**Windows**
* Visual studio 2015 or higher


#### Optional
* [cpplint](https://github.com/cpplint/cpplint): Required for linting code
* [Doxygen](http://www.doxygen.nl/): Required for building documentation
* [Boost](https://www.boost.org/) >= 1.54 : Required for building tests


### Building FLAME GPU 2

FLAME GPU 2 uses [CMake](https://cmake.org/), as a cross-platform process, for configuring and generating build directives, e.g. `Makefile` or `.vcxproj`. This is used to build the FLAMEGPU2 library, examples, tests and documentation.

#### Linux

Under Linux, `cmake` can be used to generate makefiles specific to your system:

```
mkdir -p build && cd build
cmake .. 
make
```

By default a `Makefile` for the `Release` build configuration will be generated.

Alternatively, using `-DCMAKE_BUILD_TYPE=`, `Debug` or `Profile` build configurations can be generated:
 
```
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Profile
make
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

##### Testing

If you wish to build the unit tests, [boost](https://www.boost.org/) must be available on your system. 

The test suite can be built from the root directory using `-DBUILD_TEST=ON`:

```
mkdir -p build && cd build
cmake .. -DBUILD_TEST=ON
make
```

*On Windows it is also necessary to set the `BOOST_ROOT` environment variable to assist CMake in finding boost.*

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

Cuda device architectures can be specified via `-DSMS` when generating make files, using a semi colon, space or comma separated list of compute capability numbers. I.e to build for just SM_61 and SM_70:

```
mkdir -p build && cd build
cmake .. -DSMS="61;70"
make
```

Pass `-DSMS=` to reset to the default.