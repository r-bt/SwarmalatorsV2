# Swarmalators

The swarmalator model consists of agents who couple synchronize (couple their internal states) and swarm (couple their positions). Using this model, agents give rise to many emergent behaviours. This repo was developed for our paper [**Realizing Emergent Collective Behaviors Through Robotic Swarmalators**](doi.org/10.1109/ICRA55743.2025.11128695) where we investigated realizing these behaviours on Sphero BOLT robots.

## Overview

This repo consists of multiple parts. Some of the important files and folders are descripted below

```
- swarmalators
    - main.py - Sets up the swarmalator parameters for the experiments and IDs of the Spheros being used and camera address
    - swarmalator.py - Implements the swarmalator model using numpy for faster computation. Includes chiral behaviours
    - experiments.py - Manages the experiment by connecting to nRF5380s, tracking Spheros with overhead camera, updating the swarmalator model, etc
    - tracker/ - Includes camera settings and classes for tracking Spheros and determining their initial direction
    - nRFSwarmalator/ - Handles communicating with nRF5380 boards to send commands to the Sphero Bolts
- tests
    - camera_test.py - Applys camera settings in direction_camera.py and shows live-view
    - swarmalator_test.py - Simulates the swarmalator model and shows results in GUI
```

## Installation

First, clone the repository

Then:
1. Install [**uv**](https://docs.astral.sh/uv/)
2. Run `uv run tests/swarmalator.py` – You should see the swarmalator model running
3. Connect a webcam
4. Run `uv run tests/camera_test.py` – You should see a livestream from your camera with the settings in `direction_camera.json` applied. You should change these settings to get optimal performance for your setup.

You will also need to connect nRF5380 boards to your laptop and flash them with the firmware from [nrf-Spheros](https://github.com/r-bt/nRF-Spheros/tree/master)

Finally, you'll need a black mat for the experiments to take place on. The tracker currently looks for a black mat, crops out everything else, and then looks for Spheros on that map.

## Setup

In `main.py` you should
- Replace `PORT1`, `PORT2`, etc with the ports for your nRF5380s. Each nRF5380 can handle 15 Spheros
- Replace the ids in `SPHERO_OLD` and `SPHERO_NEW` with the IDs of your Sphero BOLTs
- Replace `CAMERA_ID` with the id of your camera from pyuvc

Once done, you can change the swarmalator parameters by changing inside the `main()` function

Then you can run `uv run swarmalators/main.py`. If all goes well, a GUI (or multiple) should appear showing the livestream.

The Spheros will first all set their positive x-direction to the camera's global positive x direction

Then the experiment will begin.


