### Introduction

The project aim is to develop an existing computer application called FLAMEGPU, a high performance Graphics Processing Unit (GPU) extension tothe FLAME framework that provides a mapping between a formal agentspecifications with C based scripting and optimised CUDA code to supportscalable muti-agent simulation.  The plan it to expand/extend features andcapabilities of the exiting <a href="https://github.com/FLAMEGPU/FLAMEGPU">FLAMEGPU software</a>, in order to position it as a middleware for complex systems simulation. The application is expected to be the equivalent of Thrust for simplifying complex systems on GPUs.  The FLAMEGPU library includes  algorithms such as spatial partitioning, network communication.  

The Code is currently under active development and should **not be used** until the first release (after Sprint 1 in completed, est Easter 2017).

### Continuous Intrgation

Continuous integration is provided by Travis. This performs only build tests and the virtual machines do not support executing the Boost unit tests.

[![Build Status](https://travis-ci.org/FLAMEGPU/FLAMEGPU2_dev.svg?branch=feature%2Fcontinuous-integration)](https://travis-ci.org/FLAMEGPU/FLAMEGPU2_dev)
