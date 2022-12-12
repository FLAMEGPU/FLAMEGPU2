# Python Boids example

This FLAME GPU 2 example uses the `pyflamegpu` library. It uses Spatial 3D messaging to demonstrate communication over a fixed radius. The agent function behaviour is specified using C++ style agent function strings which are compiled just in time. The environment in this example is bounded and agents will bounce off the sides.

# Running the example

Running the model requires the pyflamegpu module to be installed. Ensure that you have installed the prerequisites listed in the main README.md and that you have build FLAME GPU using the `FLAMEGPU_BUILD_PYTHON` CMake option. This will build a virtual environment which you can activate before executing this script. E.g.

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

Alternatively, specifying 100 steps would allow you to explore the final simulation state after 100 steps have completed.