# Python Boids example

This FLAME GPU 2 example uses the `pyflamegpu` library. It uses Spatial 3D messaging to demonstrate communication over a fixed radius. The agent function behaviour is specified using ["Agent Python"](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html) a subset of Python which is translated to C++ and compiled just in time.

# Running the example

Running the model requires the pyflamegpu module to be installed. Ensure that you have installed the prerequisites listed in the main README.md and that you have build FLAME GPU using the `BUILD_SWIG_PYTHON` CMake option. This will build a virtual environment which you can activate before executing this script. E.g.

`../../build/lib/Release/python/venv/Scripts/activate`

Where `build` is the CMake build directory you specified and `Release` is the build configuration (this may also be `Debug`). *Note: activation for Powershell has a separate script file in the `Scripts` directory*.

After activating the environment the example can be launched.

`python boids_spatial3D.py`

or just

`./boids_spatial3D.py`

in Linux.

# Visualisation

If running with visualisation enabled then passing the argument `--steps 0` will ensure the simulation runs indefinitely. E.g. 

`python boids_spatial3D.py --steps 0`

The visualisation in this example allows you to control some fo the environment variables effecting behaviour. E.g. 

* **TIME_SCALE**: Controls the time steps each simulation step represents. Increasing this value will increase simulation speed but reduce the numerical accuracy of the velocity calculations. 
* **GLOBAL_SCALE**: Controls the maximum force that can be applied to an agent per time step.
* **SEPARATION_RADIUS**: Controls the distance of the separation force. This is the distance over which agents will avoid each other due to collision.
* **STEER_SCALE**: Controls the weight of the steering force (cohesion) applied to the agents per time step. The steering force encourages agents to form cohesive groups.
* **COLLISION_SCALE**: Controls the weight of the collision (separation) force applied to the agents per time step. The collision force is applied over a small distance (the SEPARATION_RADIUS).
* **MATCH_SCALE**:  Controls the weight of the match speed force  (alignment) applied to the agents per time step. The match speed force encourages agents to steer towards the average heading.
