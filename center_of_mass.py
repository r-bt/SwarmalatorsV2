import matplotlib.pyplot as plt

import numpy as np
import csv
from swarmalators import waypoints

smooth_waypoints = waypoints.SAM_waypoints_smooth

with open("outputs/state_20241207183524.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    positions = []
    for i, row in enumerate(reader):
        if i == 0:
            continue
        # Add last five columns to positions
        positions.append([])
        for pos in row[-10:]:
            # Extract ints from '[-1.53333333 -0.98333333]'
            pos = pos.replace("[", "").replace("]", "")
            pos = pos.split()
            pos = [float(x) for x in pos]
            positions[-1].append(pos)

    # Center of mass
    data = np.array(positions[1:])

    data = np.mean(data, axis=1)

    # Plot as x, y
    plt.plot(data[:, 0], data[:, 1])
    plt.plot([x[0] for x in smooth_waypoints], [x[1] for x in smooth_waypoints])

    plt.show()

# print(data)
