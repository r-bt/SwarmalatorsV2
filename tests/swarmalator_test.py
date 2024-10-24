import sys
import random

# Get current directory
cur_dir = sys.path[0]

# Add the parent directory to the path
sys.path.append(cur_dir + "/../")

from swarmalators import swarmalator as sw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import colorsys

np.random.seed(0)  # Debug have the same random numbers

agent_count = 100

positions = np.random.uniform(low=-1, high=1, size=(agent_count, 2))

natural_frequencies = np.ones(agent_count)
half = agent_count // 2
natural_frequencies[:half] = 1
natural_frequencies[half:] = -1
phase = np.linspace(0, 2 * np.pi, agent_count, endpoint=False)

random.shuffle(phase)

swarm = sw.Swarmalator(agent_count, 1, -0.5, phase, natural_frequencies, chiral=True)


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

    def update(frame):
        global positions
        global now
        global start
        global count

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

        # Draw square from (-2,-2) to (2,2)
        ax.plot([-2, 2], [2, 2], color="black")
        ax.plot([2, 2], [2, -2], color="black")
        ax.plot([2, -2], [-2, -2], color="black")
        ax.plot([-2, -2], [-2, 2], color="black")

        now = time.time()

        count += 1

    # Set up the animation
    now = time.time()
    start = time.time()
    animation = FuncAnimation(fig, update, frames=200, interval=1, blit=False)
    plt.show()


count = 0
now = time.time()
start = time.time()
plot_swarm()
