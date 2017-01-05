### Introduction

The project aim is to develop an existing computer application called FLAMEGPU, a high performance Graphics Processing Unit (GPU) extension tothe FLAME framework that provides a mapping between a formal agentspecifications with C based scripting and optimised CUDA code to supportscalable muti-agent simulation.  The plan it to expand/extend features andcapabilities of the exiting <a href="https://github.com/FLAMEGPU/FLAMEGPU">FLAMEGPU software</a>, in order to position it as a middleware for complex systems simulation. The application is expected to be the equivalent of Thrust for simplifying complex systems on GPUs.  The FLAMEGPU library includes  algorithms such as spatial partitioning, network communication.  

The Code is currently under active development and should **not be used** until the first release (after Sprint 1 in completed, est Easter 2017).

### Continuous Intrgation

Continuous integration is provided by Travis (linux) and windows (AppVeyor). This performs only build tests and the virtual machines do not support executing the Boost unit tests. Each build has a script which is required to install the CUDA tookit on the VM worker node. See the scripts folder for more details.

#### Current Master Branch Build Status

[![Build status](https://ci.appveyor.com/api/projects/status/4p58gnu8tyj7y3a7/branch/master?svg=true)](https://ci.appveyor.com/project/mondus/flamegpu2-dev/branch/master)

[![Build Status](https://travis-ci.org/FLAMEGPU/FLAMEGPU2_dev.svg?branch=master)](https://travis-ci.org/FLAMEGPU/FLAMEGPU2_dev)
