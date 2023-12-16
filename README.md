# FLAME GPU 2

[![MIT License](https://img.shields.io/github/license/FLAMEGPU/FLAMEGPU2)](https://github.com/FLAMEGPU/FLAMEGPU2/blob/master/LICENSE.md)
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/FLAMEGPU/FLAMEGPU2?include_prereleases)](https://github.com/FLAMEGPU/FLAMEGPU2/releases/latest)
[![GitHub issues](https://img.shields.io/github/issues/FLAMEGPU/FLAMEGPU2)](https://github.com/FLAMEGPU/FLAMEGPU2/issues)
[![DOI](https://zenodo.org/badge/34064755.svg)](https://zenodo.org/badge/latestdoi/34064755)
[![Website](https://img.shields.io/badge/Website-flamegpu.com-d16525)](https://flamegpu.com)
[![Userguide](https://img.shields.io/badge/Userguide-docs.flamegpu.com-fed43b)](https://docs.flamegpu.com)
[![Ubuntu](https://github.com/FLAMEGPU/FLAMEGPU2/workflows/Ubuntu/badge.svg?branch=master)](https://github.com/FLAMEGPU/FLAMEGPU2/actions?query=workflow%3AUbuntu+branch%3Amaster)
[![Windows](https://github.com/FLAMEGPU/FLAMEGPU2/workflows/Windows/badge.svg?branch=master)](https://github.com/FLAMEGPU/FLAMEGPU2/actions?query=workflow%3AWindows+branch%3Amaster)
[![Lint](https://github.com/FLAMEGPU/FLAMEGPU2/workflows/Lint/badge.svg?branch=master)](https://github.com/FLAMEGPU/FLAMEGPU2/actions?query=workflow%3ALint+branch%3Amaster)
[![Docs](https://github.com/FLAMEGPU/FLAMEGPU2/workflows/Docs/badge.svg?branch=master)](https://github.com/FLAMEGPU/FLAMEGPU2/actions?query=workflow%3ADocs+branch%3Amaster)

FLAME GPU is a GPU accelerated agent-based simulation library for domain independent complex systems simulations.
Version 2 is a complete re-write of the existing library offering greater flexibility, an improved interface for agent scripting and better research software engineering, with CUDA/C++ and Python interfaces.

FLAME GPU provides a mapping between a formal agent specifications with C++ based scripting and optimised CUDA code.
This includes a number of key Agent-based Modelling (ABM) building blocks such as multiple agent types, agent communication and birth and death allocation.

+ Agent-based (AB) modellers are able to focus on specifying agent behaviour and run simulations without explicit understanding of CUDA programming or GPU optimisation strategies.
+ Simulation performance is significantly increased in comparison with CPU alternatives. This allows simulation of far larger model sizes with high performance at a fraction of the cost of grid based alternatives.
+ Massive agent populations can be visualised in real time as agent data is already located on the GPU hardware.

## Project Status
<!-- Remove this section once it is no longer in pre-release / alpha state -->

FLAME GPU 2 is currently in an pre-release (release candidate) state, and although we hope there will not be significant changes to the API prior to a stable release there may be breaking changes as we fix issues, adjust the API and improve performance.
The use of native Python agent functions (agent functions expressed as Python syntax which are transpiled to C++) is currently supported (see examples) but classed as an *experimental* feature.

If you encounter issues while using FLAME GPU, please provide bug reports, feedback or ask questions via [GitHub Issues](https://github.com/FLAMEGPU/FLAMEGPU2/issues) and [Discussions](https://github.com/FLAMEGPU/FLAMEGPU2/discussions).

## Documentation and Support

+ [Quickstart Guide](https://docs.flamegpu.com/quickstart)
+ [Documentation and User Guide](https://docs.flamegpu.com)
+ [GitHub Discussions](https://github.com/FLAMEGPU/FLAMEGPU2/discussions)
+ [GitHub Issues](https://github.com/FLAMEGPU/FLAMEGPU2/issues)
+ [Website](https://flamegpu.com/)

## Installation

Pre-compiled python wheels are available for installation from [Releases](https://github.com/FLAMEGPU/FLAMEGPU2/releases/), and can also be installed via pip via [whl.flamegpu.com](https://whl.flamegpu.com). Wheels are not currently manylinux compliant.
Please see the [latest release](https://github.com/FLAMEGPU/FLAMEGPU2/releases/latest) for more information on the available wheels and installation instructions.

C++/CUDA installation is not currently available. Please refer to the section on [Building FLAME GPU](#Building-FLAME-GPU).

## Creating your own FLAME GPU Model

Template repositories are provided as a simple starting point for your own FLAME GPU models, with separate template repositories for the CUDA C++ and Python interfaces.
See the template repositories for further information on their use.

+ CUDA C++: [FLAME GPU 2 example template project](https://github.com/FLAMEGPU/FLAMEGPU2-example-template)
+ Python3: [FLAME GPU 2 python example template project](https://github.com/FLAMEGPU/FLAMEGPU2-python-example-template)

## Building FLAME GPU

FLAME GPU 2 uses [CMake](https://cmake.org/), as a cross-platform process, for configuring and generating build directives, e.g. `Makefile` or `.vcxproj`.
This is used to build the FLAMEGPU2 library, examples, tests and documentation.

### Requirements

Building FLAME GPU has the following requirements. There are also optional dependencies which are required for some components, such as Documentation or Python bindings.

+ [CMake](https://cmake.org/download/) `>= 3.18`
  + `>= 3.20` if building python bindings using a multi-config generator (Visual Studio, Eclipse or Ninja Multi-Config)
+ [CUDA](https://developer.nvidia.com/cuda-downloads) `>= 11.0` and a [Compute Capability](https://developer.nvidia.com/cuda-gpus) `>= 3.5` NVIDIA GPU.
+ C++17 capable C++ compiler (host), compatible with the installed CUDA version
  + [Microsoft Visual Studio 2019 or 2022](https://visualstudio.microsoft.com/) (Windows)
    + *Note:* Visual Studio must be installed before the CUDA toolkit is installed. See the [CUDA installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) for more information.
  + [make](https://www.gnu.org/software/make/) and [GCC](https://gcc.gnu.org/) `>= 8.1` (Linux)
+ [git](https://git-scm.com/)

Optionally:

+ [cpplint](https://github.com/cpplint/cpplint) for linting code
+ [Doxygen](http://www.doxygen.nl/) to build the documentation
+ [Python](https://www.python.org/) `>= 3.8` for python integration
  + With `setuptools`, `wheel`, `build` and optionally `venv` python packages installed
+ [swig](http://www.swig.org/) `>= 4.0.2` for python integration
  + Swig `4.x` will be automatically downloaded by CMake if not provided (if possible).
+ MPI (e.g. [MPICH](https://www.mpich.org/), [OpenMPI](https://www.open-mpi.org/)) for distributed ensemble support
  + MPI 3.0+ tested, older MPIs may work but not tested.
  + CMake `>= 3.20.1` may be required for some MPI libraries / platforms.
+ [FLAMEGPU2-visualiser](https://github.com/FLAMEGPU/FLAMEGPU2-visualiser) dependencies
  + [SDL](https://www.libsdl.org/)
  + [GLM](http://glm.g-truc.net/) *(consistent C++/GLSL vector maths functionality)*
  + [GLEW](http://glew.sourceforge.net/) *(GL extension loader)*
  + [FreeType](http://www.freetype.org/)  *(font loading)*
  + [DevIL](http://openil.sourceforge.net/)  *(image loading)*
  + [Fontconfig](https://www.fontconfig.org/)  *(Linux only, font detection)*

### Building with CMake

Building via CMake is a three step process, with slight differences depending on your platform.

1. Create a build directory for an out-of tree build
2. Configure CMake into the build directory
    + Using the CMake GUI or CLI tools
    + Specifying build options such as the CUDA Compute Capabilities to target, the inclusion of Visualisation or Python components, or performance impacting features such as `FLAMEGPU_SEATBELTS`. See [CMake Configuration Options](#CMake-Configuration-Options) for details of the available configuration options
    + CMake will automatically find and select compilers, libraries and python interpreters based on current environmental variables and default locations. See [Mastering CMake](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Getting%20Started.html#specifying-the-compiler-to-cmake) for more information.
        + Python dependencies must be installed in the selected python environment. If needed you can instruct CMake to use a specific python implementation using the `Python_ROOT_DIR` and `Python_Executable` CMake options at configure time.
3. Build compilation targets using the configured build system
    + See [Available Targets](#Available-targets) for a list of available targets.

#### Linux

To build under Linux using the command line, you can perform the following steps.

For example, to configure CMake for `Release` builds, for consumer Pascal GPUs (Compute Capability `61`), with python bindings enabled, producing the static library and `boids_bruteforce` example binary.

```bash
# Create the build directory and change into it
mkdir -p build && cd build

# Configure CMake from the command line passing configure-time options. 
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=61 -DFLAMEGPU_BUILD_PYTHON=ON

# Build the required targets. In this case all targets
cmake --build . --target flamegpu boids_bruteforce -j 8

# Alternatively make can be invoked directly
make flamegpu boids_bruteforce -j8

```

#### Windows

Under Windows, you must instruct CMake on which Visual Studio and architecture to build for, using the CMake `-A` and `-G` options.
This can be done through the GUI or the CLI.

I.e. to configure CMake for consumer Pascal GPUs (Compute Capability `61`), with python bindings enabled, and build the producing the static library and `boids_bruteforce` example binary in the Release configuration:

```cmd
REM Create the build directory 
mkdir build
cd build

REM Configure CMake from the command line, specifying the -A and -G options. Alternatively use the GUI
cmake .. -A x64 -G "Visual Studio 16 2019" -DCMAKE_CUDA_ARCHITECTURES=61 -DFLAMEGPU_BUILD_PYTHON=ON

REM You can then open Visual Studio manually from the .sln file, or via:
cmake --open . 
REM Alternatively, build from the command line specifying the build configuration
cmake --build . --config Release --target flamegpu boids_bruteforce --verbose
```

#### Configuring and Building a single example

It is also possible to configure and build individual examples as standalone CMake projects.

I.e. to configure and build `game_of_life` example in release mode from the command line, using linux as an example:

```bash
cd examples/game_of_life
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build . --target all
```

#### CMake Configuration Options

| Option                               | Value                       | Description                                                                                                |
| -------------------------------------| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `CMAKE_BUILD_TYPE`                   | `Release` / `Debug` / `MinSizeRel` / `RelWithDebInfo` | Select the build configuration for single-target generators such as `make`   |
| `CMAKE_CUDA_ARCHITECTURES`           | e.g `60`, `"60;70"`         | [CUDA Compute Capabilities][cuda-CC] to build/optimise for, as a `;` separated list. See [CMAKE_CUDA_ARCHITECTURES][cmake-CCA]. Defaults to `all-major` or equivalent. Alternatively use the `CUDAARCHS` environment variable. |
| `FLAMEGPU_SEATBELTS`                 | `ON`/`OFF`                  | Enable / Disable additional runtime checks which harm performance but increase usability. Default `ON`     |
| `FLAMEGPU_BUILD_PYTHON`              | `ON`/`OFF`                  | Enable Python target `pyflamegpu` via Swig. Default `OFF`. Python packages `setuptools`, `build` & `wheel` required |
| `FLAMEGPU_BUILD_PYTHON_VENV`         | `ON`/`OFF`                  | Use a python `venv` when building the python Swig target. Default `ON`. Python package `venv` required     |
| `FLAMEGPU_BUILD_TESTS`               | `ON`/`OFF`                  | Build the C++/CUDA test suite. Default `OFF`.                                                              |
| `FLAMEGPU_BUILD_TESTS_DEV`           | `ON`/`OFF`                  | Build the reduced-scope development test suite. Default `OFF`                                              |
| `FLAMEGPU_ENABLE_GTEST_DISCOVER`     | `ON`/`OFF`                  | Run individual CUDA C++ tests as independent `ctest` tests. This dramatically increases test suite runtime. Default `OFF`. |
| `FLAMEGPU_VISUALISATION`             | `ON`/`OFF`                  | Enable Visualisation. Default `OFF`.                                                                       |
| `FLAMEGPU_VISUALISATION_ROOT`        | `path/to/vis`               | Provide a path to a local copy of the visualisation repository.                                            |
| `FLAMEGPU_ENABLE_NVTX`               | `ON`/`OFF`                  | Enable NVTX markers for improved profiling. Default `OFF`                                                  |
| `FLAMEGPU_WARNINGS_AS_ERRORS`        | `ON`/`OFF`                  | Promote compiler/tool warnings to errors are build time. Default `OFF`                                     |
| `FLAMEGPU_RTC_EXPORT_SOURCES`        | `ON`/`OFF`                  | At runtime, export dynamic RTC files to disk. Useful for debugging RTC models. Default `OFF`               |
| `FLAMEGPU_RTC_DISK_CACHE`            | `ON`/`OFF`                  | Enable/Disable caching of RTC functions to disk. Default `ON`.                                             |
| `FLAMEGPU_VERBOSE_PTXAS`             | `ON`/`OFF`                  | Enable verbose PTXAS output during compilation. Default `OFF`.                                             |
| `FLAMEGPU_CURAND_ENGINE`             | `XORWOW` / `PHILOX` / `MRG` | Select the CUDA random engine. Default `XORWOW`                                                            |
| `FLAMEGPU_ENABLE_GLM`                | `ON`/`OFF`                  | Experimental feature for GLM type support within models. Default `OFF`.                                    |
| `FLAMEGPU_ENABLE_MPI`                | `ON`/`OFF`                  | Enable MPI support for distributed CUDAEnsembles, each MPI worker should have exclusive access to it's GPUs e.g. 1 MPI worker per node. Default `OFF`.                                           |
| `FLAMEGPU_ENABLE_ADVANCED_API`       | `ON`/`OFF`                  | Enable advanced API functionality (C++ only), providing access to internal sim components for high-performance extensions. No stability guarantees are provided around this interface and the returned objects.  Documentation is limited to that found in the source. Default `OFF`. |
| `FLAMEGPU_SHARE_USAGE_STATISTICS`    | `ON`/`OFF`                  | Share usage statistics ([telemetry](https://docs.flamegpu.com/guide/telemetry)) to support evidencing usage/impact of the software. Default `ON`. |
| `FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE` | `ON`/`OFF`                  | Suppress notice encouraging telemetry to be enabled, which is emitted once per binary execution if telemetry is disabled. Defaults to `OFF`, or the value of a system environment variable of the same name. |
| `FLAMEGPU_TELEMETRY_TEST_MODE`       | `ON`/`OFF`                  | Submit telemetry values to the test mode of TelemetryDeck. Intended for use during development of FLAMEGPU rather than use. Defaults to `OFF`, or the value of a system environment variable of the same name.|
| `FLAMEGPU_ENABLE_LINT_FLAMEGPU`      | `ON`/`OFF`                  | Enable/Disable creation of the `lint_flamegpu` target. Default `ON` if this repository is the root CMAKE_SOURCE_DIR, otherwise `OFF` |

<!-- Additional options which users can find if they need them.
| `FLAMEGPU_BUILD_API_DOCUMENTATION` | `ON`/`OFF` | Build the documentation target. Default `ON` |
| `FLAMEGPU_BUILD_ALL_EXAMPLES` | `ON`/`OFF` | Build the suite of example models. Default `ON` if FLAMEGPU is the top-level CMake project |
| `FLAMEGPU_BUILD_EXAMPLE_*` | `ON`/`OFF` | Build individual examples as required, if `FLAMEGPU_BUILD_ALL_EXAMPLES` is `OFF`. Default `OFF` |
 -->

[cuda-CC]: https://developer.nvidia.com/cuda-gpus
[cmake-CCA]: https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html

For a list of available CMake configuration options, run the following from the `build` directory:

```bash
cmake -LH ..
```

#### Available Targets

| Target         | Description                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------------- |
| `all`          | Linux target containing default set of targets, including everything but the documentation and lint targets   |
| `ALL_BUILD`    | The windows equivalent of `all`                                                                               |
| `all_lint`     | Run all available Linter targets                                                                              |
| `flamegpu`     | Build FLAME GPU static library                                                                                |
| `pyflamegpu`   | Build the python bindings for FLAME GPU                                                                       |
| `docs`         | The FLAME GPU API documentation (if available)                                                                |
| `tests`        | Build the CUDA C++ test suite, if enabled by `FLAMEGPU_BUILD_TESTS=ON`                                                 |
| `tests_dev`    | Build the CUDA C++ test suite, if enabled by `FLAMEGPU_BUILD_TESTS_DEV=ON`                                             |
| `<example>`    | Each individual model has it's own target. I.e. `boids_bruteforce` corresponds to `examples/boids_bruteforce` |
| `lint_<other>` | Lint the `<other>` target. I.e. `lint_flamegpu` will lint the `flamegpu` target                               |

For a full list of available targets, run the following after configuring CMake:

```bash
cmake --build . --target help
```

## Usage

Once compiled individual models can be executed from the command line, with a range of default command line arguments depending on whether the model implements a single Simulation, or an Ensemble of simulations.

To see the available command line arguments use the `-h` or `--help` options, for either C++ or python models.

I.e. for a `Release` build of the `game_of_life` model, run:

```bash
./bin/Release/game_of_life --help
```

### Visual Studio

If wishing to run examples within Visual Studio it is necessary to right click the desired example in the Solution Explorer and select `Debug > Start New Instance`.
Alternatively, if `Set as StartUp Project` is selected, the main debugging menus can be used to initiate execution.
To configure command line argument for execution within Visual Studio, right click the desired example in the Solution Explorer and select `Properties`, in this dialog select `Debugging` in the left hand menu to display the entry field for `command arguments`.
Note, it may be necessary to change the configuration as the properties dialog may be targeting a different configuration to the current build configuration.

### Environment Variables

Several environmental variables are used or required by FLAME GPU 2.

| Environment Variable                 | Description |
|--------------------------------------|-------------|
| `CUDA_PATH`                          | Required when using RunTime Compilation (RTC), pointing to the root of the CUDA Toolkit where NVRTC resides. <br /> i.e. `/usr/local/cuda-11.0/` or `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0`. <br /> Alternatively `CUDA_HOME` may be used if `CUDA_PATH` was not set. |
| `FLAMEGPU_INC_DIR`                   | When RTC compilation is required, if the location of the `include` directory cannot be found it must be specified using the `FLAMEGPU_INC_DIR` environment variable. |
| `FLAMEGPU_TMP_DIR`                   | FLAME GPU may cache some files to a temporary directory on the system, using the temporary directory returned by [`std::filesystem::temp_directory_path`](https://en.cppreference.com/w/cpp/filesystem/temp_directory_path). The location can optionally be overridden using the `FLAMEGPU_TMP_DIR` environment variable. |
| `FLAMEGPU_RTC_INCLUDE_DIRS`          | A list of include directories that should be provided to the RTC compiler, these should be separated using `;` (Windows) or `:` (Linux). If this variable is not found, the working directory will be used as a default. |
| `FLAMEGPU_SHARE_USAGE_STATISTICS`    | Enable / Disable sending of telemetry data, when set to `ON` or `OFF` respectively. |
| `FLAMEGPU_TELEMETRY_SUPPRESS_NOTICE` | Enable / Disable a once per execution notice encouraging the use of telemetry, if telemetry is disabled, when set to `ON` or `OFF` respectively. |
| `FLAMEGPU_TELEMETRY_TEST_MODE`       | Enable / Disable sending telemetry data to a test endpoint, for FLAMEGPU development to separate user statistics from developer statistics. Set to `ON` or `OFF`. |
| `FLAMEGPU_GLM_INC_DIR`                        | When RTC compilation is required and GLM support has been enabled, if the location of the GLM include directory cannot be found it must be specified using the `FLAMEGPU_GLM_INC_DIR` environment variable. |

## Running the Test Suite(s)

### CUDA C++ Test Suites

The test suite for the CUDA/C++ library can be executed using CTest, or by manually running the test executable(s).

 can be used to orchestrate running multiple test suites for different aspects of FLAME GPU 2.

The test suite can be executed using [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html) by running `ctest`, or `ctest -VV` for verbose output of sub-tests, from the the build directory.

More verbose CTest output for the GoogleTest based CUDA C++ test suite(s) can be enabled by configuring CMake with `FLAMEGPU_ENABLE_GTEST_DISCOVER` set to `ON`.
This however will dramatically increase test suite execution time.

1. Configure CMake to build the desired tests suites as desired, using `FLAMEGPU_BUILD_TESTS=ON`, `FLAMEGPU_BUILD_TESTS_DEV=ON` and optionally `FLAMEGPU_ENABLE_GTEST_DISCOVER=ON`.
2. Build the `tests`, `tests_dev` targets as required
3. Run the test suites via ctest, using `-vv` for more-verbose output. Multiple tests can be ran concurrently using `-j <jobs>`. Use `-R <regex>` to only run matching tests.

    ```bash
    ctest -vv -j 8
    ```

To run the CUDA/C++ test suite(s) manually, which allows use of `--gtest_filter`:

1. Configure CMake with `FLAMEGPU_BUILD_TESTS=ON`
2. Build the `tests` target
3. Run the test suite executable for the selected configuration i.e.

    ```bash
    ./bin/Release/tests
    ```

### Python Testing via pytest

To run the python test suite:

1. Configure CMake with `FLAMEGPU_BUILD_PYTHON=ON`
2. Build the `pyflamegpu` target
3. Activate the generated python `venv` for the selected configuration, which has `pyflamegpu` and `pytest` installed

    If using Bash (linux, bash for windows)

    ```bash
    source lib/Release/python/venv/bin/activate
    ```

    If using `cmd`:

    ```bat
    call lib\Release\python\venv\Scripts\activate.bat
    ```

    Or if using Powershell:

    ```powershell
    . lib\Release\python\venv\Scripts\activate.ps1
    ```

4. Run `pytest` on the `tests/python` directory. This may take some time.

    ```bash
    python3 -m pytest ../tests/python
    ```

## Usage Statistics (Telemetry)

Support for academic software is dependant on evidence of impact. Without evidence it is difficult/impossible to justify investment to add features and provide maintenance. We collect a minimal amount of anonymous usage data so that we can gather usage statistics that enable us to continue to develop the software under a free and permissible licence.

Information is collected when a simulation, ensemble or test suite run have completed.

The [TelemetryDeck](https://telemetrydeck.com/) service is used to store telemetry data. 
All data is sent to their API endpoint of https://nom.telemetrydeck.com/v1/ via https. For more details please review the [TelmetryDeck privacy policy](https://telemetrydeck.com/privacy/).

We do not collect any personal data such as usernames, email addresses or machine identifiers.

More information can be found in the [FLAMEGPU documentation](https://docs.flamegpu.com/guide/telemetry).

Telemetry is enabled by default, but can be opted out by:

+ Setting an environment variable `FLAMEGPU_SHARE_USAGE_STATISTICS` to `OFF`, `false` or `0` (case insensitive).
  + If this is set during the first CMake configuration it will be used for all subsequent CMake configurations until the CMake Cache is cleared, or it is manually changed.
  + If this is set during simulation, ensemble or test execution (i.e. runtime) it will also be respected
+ Setting the `FLAMEGPU_SHARE_USAGE_STATISTICS` CMake option to `OFF` or another false-like CMake value, which will default telemetry to be off for executions.
+ Programmatically overriding the default value by:
  + Calling `flamegpu::io::Telemetry::disable()` or `pyflamegpu.Telemetry.disable()` prior to the construction of any `Simulation`, `CUDASimulation` or `CUDAEnsemble` objects.
  + Setting the `telemetry` config property of a `Simulation.Config`, `CUDASimulation.SimulationConfig` or `CUDAEnsemble.EnsembleConfig` to `false`.

## Contributing

Feel free to submit [Pull Requests](https://github.com/FLAMEGPU/FLAMEGPU2/pulls), create [Issues](https://github.com/FLAMEGPU/FLAMEGPU2/issues) or open [Discussions](https://github.com/FLAMEGPU/FLAMEGPU2/discussions).

See [CONTRIBUTING.md](https://github.com/FLAMEGPU/FLAMEGPU2/blob/master/CONTRIBUTING.md) for more detailed information on how to contribute to FLAME GPU.

## Authors and Acknowledgment

See [Contributors](https://github.com/FLAMEGPU/FLAMEGPU2/graphs/contributors) for a list of contributors towards this project.

If you use this software in your work, please cite DOI [10.5281/zenodo.5428984](https://doi.org/10.5281/zenodo.5428984). Release specific DOI are also provided via Zenodo.

Alternatively, [CITATION.cff](https://github.com/FLAMEGPU/FLAMEGPU2/blob/master/CITATION.cff) provides citation metadata, which can also be accessed from [GitHub](https://github.com/FLAMEGPU/FLAMEGPU2).

## License

FLAME GPU is distributed under the [MIT Licence](https://github.com/FLAMEGPU/FLAMEGPU2/blob/master/LICENSE.md).

## Known issues

There are currently several known issues which will be fixed in future releases (where possible).
For a full list of known issues pleases see the [Issue Tracker](https://github.com/FLAMEGPU/FLAMEGPU2/issues).

+ Warnings and a loss of performance due to hash collisions in device code ([#356](https://github.com/FLAMEGPU/FLAMEGPU2/issues/356))
+ Multiple known areas where performance can be improved (e.g. [#449](https://github.com/FLAMEGPU/FLAMEGPU2/issues/449), [#402](https://github.com/FLAMEGPU/FLAMEGPU2/issues/402))
+ Windows/MSVC builds using CUDA 11.0 may encounter errors when performing incremental builds if the static library has been recompiled. If this presents itself, re-save any `.cu` file in your executable producing project and re-trigger the build.
+ Debug builds under linux with CUDA 11.0 may encounter cuda errors during `validateIDCollisions`. Consider using an alternate CUDA version if this is required ([#569](https://github.com/FLAMEGPU/FLAMEGPU2/issues/569)).
+ CUDA 11.0 with GCC 9 may encounter a segmentation fault during compilation of the test suite. Consider using GCC 8 with CUDA 11.0.