import sys
import random

# Get current directory
cur_dir = sys.path[0]

# Add the parent directory to the path
sys.path.append(cur_dir + "/../")

# from swarmalators import swarmalator as sw
# from swarmalators import swarmalator as sw
from swarmalators import swarmalator as sw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import colorsys

# from swarmalators import LaserTracker

np.random.seed(0)  # Debug have the same random numbers

agent_count = 15

positions = np.random.uniform(low=-1, high=1, size=(agent_count, 2))

natural_frequencies = np.zeros(agent_count)

phase = np.linspace(0, 2 * np.pi, agent_count, endpoint=False)
# phase = np.ones(agent_count) * np.pi / 2
random.shuffle(phase)

# phase = np.ones(agent_count) * np.pi / 2

swarm = sw.Swarmalator(agent_count, -1, 0, phase, natural_frequencies, chiral=False)


def angles_to_rgb(angles_rad):
    # Convert the angles to hue values (ranges from 0.0 to 1.0 in the HSV color space)
    hues = angles_rad / (2 * np.pi)

    # Set fixed values for saturation and value (you can adjust these as desired)
    saturation = np.ones_like(hues)
    value = np.ones_like(hues)

    hsv_colors = np.stack((hues, saturation, value), axis=-1)
    rgb_colors = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), -1, hsv_colors)

    # Scale RGB values to 0-255 range
    rgb_colors *= 255
    rgb_colors = rgb_colors.astype(np.uint8)

    return rgb_colors


time_multipler = 5


def plot_swarm():
    global now
    global start
    fig, ax = plt.subplots()

    centers = []

    def update(frame):
        global positions
        global now
        global start
        global count
        global index

        # # Update the model
        swarm.update(positions)
        dt = (time.time() - now) * time_multipler
        swarm.update_phase(dt)
        # # Update position based on previous velocity

        positions += swarm.get_velocity() * dt

        phase_state = swarm.get_phase_state()
        colors = angles_to_rgb(phase_state) / 255.0

        # # # Plot the positions
        ax.clear()
        ax.scatter(positions[:, 0], positions[:, 1], c=colors)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

        now = time.time()

        # Get the center of the swarm
        center = np.mean(positions, axis=0)

        # If the center is close to the target, change the target

        if swarm._target is not None:
            if np.linalg.norm(center - swarm._target) < 0.1:
                index += 1
                index = index % len(target_positions)
                swarm.set_target(target_positions[index])
                # Add to centers
                centers.append(center)

        # Draw a line between center history
        for i in range(1, len(centers)):
            ax.plot(
                [centers[i - 1][0], centers[i][0]],
                [centers[i - 1][1], centers[i][1]],
                color="black",
            )

        count += 1

    # Set up the animation
    now = time.time()
    start = time.time()
    animation = FuncAnimation(fig, update, frames=200, interval=1, blit=False)
    plt.show()


count = 0
now = time.time()
start = time.time()


# laser_tracker = LaserTracker()

# laser_tracker.track()

# path = laser_tracker.get_path()

# target_positions = path

# print(target_positions)

target_positions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]

# # Scale the path to be within the -2 to 2 range

# path = np.array(path)

# path = path - np.min(path, axis=0)
# path = path / np.max(path, axis=0) * 4 - 2

# target_positions = path

# print(target_positions)

index = 0

# swarm.set_target(target_positions[index])

plot_swarm()
