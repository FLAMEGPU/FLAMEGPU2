# FLAME GPU Examples

This document briefly summarises the purpose of each example model provided, and links to any additional resources.

The examples directory has four subdirectories, corresponding to different model language options:

* `cpp`: Pure c++ models.
* `cpp_rtc`: C++ models with agent functions implemented as [runtime compiled CUDA](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html#c-and-python-runtime-compiled-agent-functions).
* `python_native`: C++ models with agent functions implemented as runtime transpiled and compiled [agent python](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html#flame-gpu-python-agent-functions).
* `python_rtc`: Python models with agent functions implemented as [runtime compiled CUDA](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html#c-and-python-runtime-compiled-agent-functions).


## Boids

The Boids example is a classic agent based model, first [published by Craig Reynolds in 1986](https://en.wikipedia.org/wiki/Boids). Agent's represent birds flying in a flock according to three simple rules:

* Separation: Avoid crowding local neighbours
* Alignment: Follow the average direction of local neighbours
* Cohesion: Move towards the centre of mass of local neighbours

As a classic model, this has been implemented in many frameworks and provides a means of comparison.

This example is available using both brute force and spatial messaging, and is only available in 3 dimensions.


[![Boids Youtube thumbnail](https://img.youtube.com/vi/eE646_6NfqQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=eE646_6NfqQ)

### Versions

* `cpp/boids_bruteforce`: 3D, BruteForce messaging, Bounded environment
* `cpp/boids_spatial3D`: 3D, Spatial messaging, Bounded environment
* `cpp_rtc/boids_bruteforce`: RTC, 3D, BruteForce messaging, Bounded environment
* `cpp_rtc/boids_spatial3D`: RTC, 3D, Spatial messaging, Bounded environment
* `python_native/boids_spatial3D_wrapped`: RTC, 3D, Spatial messaging, Wrapped environment
* `python_rtc/boids_spatial3D_bounded`: RTC, 3D, Spatial messaging, Bounded environment

## Circles

The Circles example was developed as a simple agent model for benchmarking communication strategies. It is an improvement over the [version published in 2017](https://link.springer.com/chapter/10.1007/978-3-319-58943-5_25), which smooths agent movement to reduce jitter.

[![Circles Youtube thumbnail](https://img.youtube.com/vi/ZedroqmOaHU/maxresdefault.jpg)](https://www.youtube.com/watch?v=ZedroqmOaHU)

### Versions

* `cpp/circles_bruteforce`: 3D, BruteForce messaging, Bounded environment.
* `cpp/circles_spatial3D`: 3D, Spatial messaging, Bounded environment
*  [FLAME GPU Tutorial](https://docs.flamegpu.com/tutorial/index.html#complete-tutorial-code): 2D, Spatial messaging, C++, Python with RTC, Python with Agent Python

## Game of Life

The Game of Life example implements [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life), a classic and relatively simple cellular automata.

It allows the demonstration of how cellular automata and other discrete models can be implemented within FLAME GPU. Unlike the original FLAME GPU 1, FLAME GPU 2 does not have distinct discrete agents.

As FLAME GPU is GPU accelerated it is possible to push the scale of the Game of Life to millions of agents whilst maintaining responsive performance.

### Versions

* `cpp/game_of_life`

## Sugarscape

The Sugarscape example implements a version of [Sugarscape](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life), which is a cellular automata that models social behaviour of agents navigating to find cells that contain sugar. As they consume sugar it degrades within the cell and must grow back, causing agents to then move on. Agents able to consume high amounts of sugar replicate, whereas agents that starve are removed. There are many different rule sets for Sugarscape, FLAME GPU does not implement any particular one, the model is merely representative of the Sugarscape class of models.

This example within FLAME GPU allows submodels to be demonstrated. The submodel is used for the iterative parallel conflict resolution algorithm that decides which agent can move into a cell where two or more agents all wish to occupy the same cell. The four agent function's within the submodel iterate until the exit condition detects that all agents have moved.

Besides the submodel, the model only has 1 other agent function `metabolise_and_growback`, which controls sugar growth.

*Note: The agents within the FLAME GPU model represent the cells. The agents which replicate and consume sugar are implicit, instead existing as two variables `agent_id` and `status` within the cell agents.*

[![Sugarscape Youtube thumbnail](https://img.youtube.com/vi/tSLV19AWfwg/maxresdefault.jpg)](https://www.youtube.com/watch?v=tSLV19AWfwg)

### Versions

* `cpp/sugarscape`

## Pedestrian Navigation
The pedestrian navigation example implements a copy of the original FLAME GPU 1 pedestrian navigation example.

It was created to demonstrate keyframe animation and user interface support within FLAME GPU. Previously in FLAME GPU 1 much of this visualisation functionality had to be hand coded.

Pedestrian agents walk between pairs of entrances whilst avoiding collisions with one another (via the social forces model of pedestrian collision avoidance).

[![Pedestrian Navigation Youtube thumbnail](https://img.youtube.com/vi/9jPfBs81XmE/maxresdefault.jpg)](https://www.youtube.com/watch?v=9jPfBs81XmE)

### Versions

The pedestrian navigation exists as a C++ model in a stand-alone [GitHub repository](https://github.com/FLAMEGPU/FLAMEGPU2-pedestrian_navigation-example).

## Diffusion

The diffusion example implements a heat equation, to show how heat diffuses from a material, based on [this example](https://enccs.github.io/OpenACC-CUDA-beginners/2.02_cuda-heat-equation/).

It provides a demonstration of both Array2D message communication and exit conditions.

### Versions

* `cpp/diffusion`

## Ensemble

The ensemble example exists to demonstrate how a FLAME GPU ensemble of simulations can be executed, the model is more of a basic test case and does not have a visualisation.

### Versions

* `cpp/ensemble`

## Host Functions

The host functions example exists to demonstrate the range of supported host function behaviours within FLAME GPU, the model is more of a basic test case and does not have a visualisation.

### Versions

* `cpp/host_functions`
