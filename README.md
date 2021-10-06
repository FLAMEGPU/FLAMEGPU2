# FLAME GPU 2

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/FLAMEGPU/FLAMEGPU2/blob/master/LICENSE.MD)
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

FLAME GPU 2 is currently in an pre-release (alpha) state, and although we hope there will not be significant changes to the API prior to a stable release there may be breaking changes as we fix issues, adjust the API and improve performance.

If you encounter issues while using FLAME GPU, please provide bug reports, feedback or ask questions via [GitHub Issues](https://github.com/FLAMEGPU/FLAMEGPU2/issues) and [Discussions](https://github.com/FLAMEGPU/FLAMEGPU2/discussions).

## Documentation and Support

+ [Quickstart Guide](https://docs.flamegpu.com/quickstart)
+ [Documentation and User Guide](https://docs.flamegpu.com)
+ [GitHub Discussions](https://github.com/FLAMEGPU/FLAMEGPU2/discussions)
+ [GitHub Issues](https://github.com/FLAMEGPU/FLAMEGPU2/issues)
+ [Website](https://flamegpu.com/)

## Installation

Pre-compiled python wheels are available for installations from [Releases](https://github.com/FLAMEGPU/FLAMEGPU2/releases/).
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
+ [CUDA](https://developer.nvidia.com/cuda-downloads) `>= 11.0` and a [Compute Capability](https://developer.nvidia.com/cuda-gpus) `>= 3.5` NVIDIA GPU.
  + CUDA `>= 10.0` currently works, but support will be dropped in a future release.
+ C++17 capable C++ compiler (host), compatible with the installed CUDA version
  + [Microsoft Visual Studio 2019](https://visualstudio.microsoft.com/) (Windows)
  + [make](https://www.gnu.org/software/make/) and [GCC](https://gcc.gnu.org/) `>= 7` (Linux)
  + Older C++ compilers which support C++14 may currently work, but support will be dropped in a future release.
+ [git](https://git-scm.com/)

Optionally:

+ [cpplint](https://github.com/cpplint/cpplint) for linting code
+ [Doxygen](http://www.doxygen.nl/) to build the documentation
+ [Python](https://www.python.org/) `>= 3.6` for python integration
+ [swig](http://www.swig.org/) `>= 4.0.2` for python integration
  + Swig `4.x` will be automatically downloaded by CMake if not provided (if possible).
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
    + Specifying build options such as the CUDA Compute Capabilities to target, the inclusion of Visualisation or Python components, or performance impacting features such as `SEATBELTS`. See [CMake Configuration Options](#CMake-Configuration-Options) for details of the available configuration options
3. Build compilation targets using the configured build system
    + See [Available Targets](#Available-targets) for a list of available targets.

#### Linux

To build under Linux using the command line, you can perform the following steps.

For example, to configure CMake for `Release` builds, for consumer Pascal GPUs (Compute Capability `61`), with python bindings enabled, producing the static library and `boids_bruteforce` example binary.

```bash
# Create the build directory and change into it
mkdir -p build && cd build

# Configure CMake from the command line passing configure-time options. 
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=61 -DBUILD_SWIG_PYTHON=ON

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
cmake .. -A x64 -G "Visual Studio 16 2019" -DCUDA_ARCH=61 -DBUILD_SWIG_PYTHON=ON

REM You can then open Visual Studio manually from the .sln file, or via:
cmake --open . 
REM Alternatively, build from the command line specifying the build configuration
cmake --build . --config Release --target flamegpu boids_bruteforce --verbose
```

*Note, CUDA must be installed after Visual Studio, otherwise CMake may fail to detect CUDA and the configuration stage will fail.*

#### Configuring and Building a single example

It is also possible to configure and build individual examples as standalone CMake projects.

I.e. to configure and build `game_of_life` example in release mode from the command line, using linux as an example:

```bash
cd examples/game_of_life
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=61
cmake --build . --target all
```

#### CMake Configuration Options

| Option                   | Value             | Description                                                                                                |
| ------------------------ | ----------------- | ---------------------------------------------------------------------------------------------------------- |
| `CMAKE_BUILD_TYPE`       | `Release`/`Debug` | Select the build configuration for single-target generators such as `make`                                 |
| `SEATBELTS`              | `ON`/`OFF`        | Enable / Disable additional runtime checks which harm performance but increase usability. Default `ON`     |
| `CUDA_ARCH`              | `"52 60 70 80"`   | Select [CUDA Compute Capabilities](https://developer.nvidia.com/cuda-gpus) to build/optimise for, as a space or `;` separated list. Defaults to `""` |
| `BUILD_SWIG_PYTHON`      | `ON`/`OFF`        | Enable Python target `pyflamegpu` via Swig. Default `OFF`                                                  |
| `BUILD_SWIG_PYTHON_VENV` | `ON`/`OFF`        | Use a python `venv` when building the python Swig target. Default `ON`.                                    |
| `BUILD_TESTS`            | `ON`/`OFF`        | Build the C++/CUDA test suite. Default `OFF`.                                                              |
| `BUILD_TESTS_DEV`        | `ON`/`OFF`        | Build the reduced-scope development test suite. Default `OFF`                                              |
| `VISUALISATION`          | `ON`/`OFF`        | Enable Visualisation. Default `OFF`.                                                                       |
| `VISUALISATION_ROOT`     | `path/to/vis`     | Provide a path to a local copy of the visualisation repository.                                            |
| `USE_NVTX`               | `ON`/`OFF`        | Enable NVTX markers for improved profiling. Default `OFF`                                                  |
| `WARNINGS_AS_ERRORS`     | `ON`/`OFF`        | Promote compiler/tool warnings to errors are build time. Default `OFF`                                     |
| `EXPORT_RTC_SOURCES`     | `ON`/`OFF`        | At runtime, export dynamic RTC files to disk. Useful for debugging RTC models. Default `OFF`               |
| `RTC_DISK_CACHE`         | `ON`/`OFF`        | Enable/Disable caching of RTC functions to disk. Default `ON`.                                             |
| `USE_GLM`                | `ON`/`OFF`        | Experimental feature for GLM type support in RTC models. Default `OFF`.                                    |

<!-- Additional options which users can find if they need them.
| `BUILD_FLAMEGPU` | `ON`/`OFF` | Build the main FLAMEGPU static library target. Default `ON` |
| `BUILD_API_DOCUMENTATION` | `ON`/`OFF` | Build the documentation target. Default `ON` |

| `BUILD_ALL_EXAMPLES` | `ON`/`OFF` | Build the suite of example models. Default `ON` |

| `BUILD_EXAMPLE_*` | `ON`/`OFF` | Build individual examples as required, if `BUILD_ALL_EXAMPLES` is `OFF`. Default `OFF` |
 -->

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
| `tests`        | Build the CUDA C++ test suite, if enabled by `BUILD_TESTS=ON`                                                 |
| `tests_dev`    | Build the CUDA C++ test suite, if enabled by `BUILD_TESTS_DEV=ON`                                             |
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

+ `CUDA_PATH` - Required when using RunTime Compilation (RTC), pointing to the root of the CUDA Toolkit where NVRTC resides.
  + i.e. `/usr/local/cuda-11.0/` or `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0`.
  + Alternatively `CUDA_HOME` may be used if `CUDA_PATH` was not set.
+ `FLAMEGPU_INC_DIR` - When RTC compilation is required, if the location of the `include` directory cannot be found it must be specified using the `FLAMEGPU_INC_DIR` environment variable.
+ `FLAMEGPU_TMP_DIR` - FLAME GPU may cache some files to a temporary directory on the system, using the temporary directory returned by [`std::filesystem::temp_directory_path`](https://en.cppreference.com/w/cpp/filesystem/temp_directory_path). The location can optionally be overridden using the `FLAMEGPU_TMP_DIR` environment variable.

## Running the Test Suite(s)

To run the CUDA/C++ test suite:

1. Configure CMake with `BUILD_TESTS=ON`
2. Build the `tests` target
3. Run the test suite executable for the selected configuration i.e.

    ```bash
    ./bin/Release/tests
    ```

To run the python test suite:

1. Configure CMake with `BUILD_SWIG_PYTHON=ON`
2. Build the `pyflamegpu` target
3. Activate the generated python `venv` for the selected configuration, which has `pyflamegpu` and `pytest` installed

    If using Bash (linux, bash for windows)

    ```bash
    source lib/Release/lib/Release/python/venv/bin/activate
    ```

    If using `cmd`:

    ```bat
    call lib\Release\python\venv\Scripts\activate.bat
    ```

    Or if using Powershell:

    ```powershell
    . lib\Release\python\venv\activate.ps1
    ```

4. Run `pytest` on the `tests/swig/python` directory. This may take some time.

    ```bash
    python3 -m pytest ../tests/swig/python
    ```

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

+ Performance regressions in CUDA 11.3+, due to changes in compiler register usage ([#560](https://github.com/FLAMEGPU/FLAMEGPU2/issues/560)).
+ Segfault when using `flamegpu::DependencyGraph` via the default constructor ([#555](https://github.com/FLAMEGPU/FLAMEGPU2/issues/555)). This will require an API break to resolve.
+ Warnings and a loss of performance due to hash collisions in device code ([#356](https://github.com/FLAMEGPU/FLAMEGPU2/issues/356))
+ Multiple known areas where performance can be improved (e.g. [#449](https://github.com/FLAMEGPU/FLAMEGPU2/issues/449), [#402](https://github.com/FLAMEGPU/FLAMEGPU2/issues/402))
+ Windows/MSVC builds using CUDA < 11.0 may encounter intermittent compiler failures. Please use CUDA 11.0+.
  + This will be resolved by dropping CUDA 10 support in a future release.
+ Windows/MSVC builds using CUDA 11.0 may encounter errors when performing incremental builds if the static library has been recompiled. If this presents itself, re-save any `.cu` file in your executable producing project and re-trigger the build.
+ Debug builds under linux with CUDA 11.0 may encounter cuda errors during `validateIDCollisions`. Consider using an alternate CUDA version if this is required ([#569](https://github.com/FLAMEGPU/FLAMEGPU2/issues/569)).
+ CUDA 11.0 with GCC 9 may encounter a segmentation fault during compilation of the test suite. Consider using GCC 8 with CUDA 11.0.
+ CMake 3.16 has known issues on some platforms. CMake versions less than 3.18 are deprecated and support will be removed in a future release.
