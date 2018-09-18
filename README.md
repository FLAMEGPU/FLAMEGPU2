### Introduction

The project aim is to develop an existing computer application called FLAMEGPU, a high performance Graphics Processing Unit (GPU) extension to the FLAME framework that provides a mapping between a formal agent specifications with C based scripting and optimised CUDA code to support scalable muti-agent simulation.  The plan it to expand/extend features and capabilities of the exiting <a href="https://github.com/FLAMEGPU/FLAMEGPU">FLAMEGPU software</a>, in order to position it as a middle-ware for complex systems simulation. The application is expected to be the equivalent of Thrust for simplifying complex systems on GPUs.  The FLAMEGPU library includes  algorithms such as spatial partitioning, network communication.

The Code is currently under active development and should **not be used** until the first release.

### Continuous Integration

Continuous integration is provided by Travis (Linux) and windows (AppVeyor). This performs only build tests and the virtual machines do not support executing the Boost unit tests. Each build has a script which is required to install the CUDA toolkit on the VM worker node. See the scripts folder for more details.

#### Current Master Branch Build Status

[![Build status](https://ci.appveyor.com/api/projects/status/4p58gnu8tyj7y3a7/branch/master?svg=true)](https://ci.appveyor.com/project/mondus/flamegpu2-dev/branch/master)

[![Build Status](https://travis-ci.org/FLAMEGPU/FLAMEGPU2_dev.svg?branch=master)](https://travis-ci.org/FLAMEGPU/FLAMEGPU2_dev)

### Building FLAME GPU 2


#### Visual Studio

Visual studio project files are provided to build the static library and example projects.

##### Testing

If you wish to build and run the unit tests, boost must be installed and the environment variable `BOOST_ROOT` must be set. 

#### CMake

Under linux, `cmake` can be used to generate makefiles specific to your system.

I.e

```
mkdir -p build && cd build
cmake .. 
make
```

Individual examples can be built from their respective directories

```
cd examples/mas/
mkdir -p build && cd build
cmake ..
make 
```

##### Testing
If you wish to build the unit tests, boost must be available on your system. The test suite can be built from the root directory via

```
mkdir -p build && cd build
cmake .. -DBUILD_TEST=ON
make
```

##### Device Architectures

Cuda device architectures can be specified when generating make files, using a semi colon, space or comma separated list of compute capability numbers. I.e to build for just SM_61 and SM_70:

```
mkdir -p build && cd build
cmake .. -DSMS="60;70"
make
```

Pass `-DSMS=` to reset to the default.


### Doxygen Readme 

> @todo - this may not be required.

To create a new Doxyfile, run:
```
doxygen -g
```
Then configure the Doxyfile and set below:

```
INPUT                = api/ gpu/ model/ pop/ sim/ main.cpp
PROJECT_NAME         = "FLAMEGPU 2.0"
GENERATE_LATEX       = NO
EXTRACT_ALL          = YES
CLASS _DIAGRAMS      = YES
HIDE_UNDOC_RELATIONS = NO
HAVE_DOT             = YES
CLASS_GRAPH          = YES
COLLABORATION_GRAPH  = YES
UML_LOOK             = YES
UML_LIMIT_NUM_FIELDS = 50
TEMPLATE_RELATIONS   = YES
DOT_GRAPH_MAX_NODES  = 100
MAX_DOT_GRAPH_DEPTH  = 0
DOT_TRANSPARENT      = NO
CALL_GRAPH           = YES
CALLER_GRAPH         = YES
GENERATE_TREEVIEW    = YES
HTML_OUTPUT          = docs
DOT_IMAGE_FORMAT     = png # can be  svg, but the add --> INTERACTIVE_SVG      = YES

# -- added later

EXTRACT_PRIVATE        = YES
EXTRACT_STATIC         = YES
EXTRACT_LOCAL_METHODS  = NO
```
now, run:
```
doxygen Doxyfile
```

To view, open:
docs/index.html

#https://www.daniweb.com/programming/software-development/threads/398953/doxygen-multiple-files
