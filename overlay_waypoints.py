import cv2
import csv
import numpy as np

video_path = "outputs/output_20241207182642.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get some properties of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate video duration
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_duration = frame_count / fps  # Video duration in seconds

scale_factor = 8.0 / max(frame_width, frame_height)


def unnormalize_coordinates(scale_factor, width, height, x, y):
    unnormalized_x = (x / scale_factor) + width / 2
    unnormalized_y = (-y / scale_factor) + height / 2
    return unnormalized_x, unnormalized_y


with open("outputs/state_20241207182642.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    positions = []
    times = []
    for i, row in enumerate(reader):
        if i == 0:
            continue
        # Add last five columns to positions
        positions.append([])
        for pos in row[-10:]:
            # Extract floats from '[-1.53333333 -0.98333333]'
            pos = pos.replace("[", "").replace("]", "")
            pos = pos.split()
            pos = [float(x) for x in pos]
            positions[-1].append(pos)
        times.append(float(row[0]))

times = np.array(times) - times[0]

# Scale times to fit video duration
max_time = times[
    -1
]  # Assume the last time in the CSV is the total duration of the data
scaled_times = [t * (video_duration / max_time) for t in times]

c_o_m = np.mean(np.array(positions[1:]), axis=1)

points = [
    unnormalize_coordinates(scale_factor, frame_width, frame_height, x, y)
    for x, y in c_o_m
]

# Create new video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "DRL_waypoints_20241207182642.mp4", fourcc, 30, (frame_width, frame_height)
)

plotted_points = []
index = 0  # Track the current point index

while True:
    ret, frame = cap.read()

    if not ret:  # Break if no frames left
        break

    # Get the current video time in seconds
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Plot the next point only if the video time has reached or surpassed the point's scaled time
    if index < len(scaled_times) and current_time >= scaled_times[index]:
        plotted_points.append(points[index])
        index += 1

    # Draw lines connecting the plotted points
    for i in range(1, len(plotted_points)):
        cv2.line(
            frame,
            (int(plotted_points[i - 1][0]), int(plotted_points[i - 1][1])),
            (int(plotted_points[i][0]), int(plotted_points[i][1])),
            # Red
            (0, 0, 255),
            2,
        )

    # Write the frame
    out.write(frame)

# Close the video writer
out.release()

# Release resources
cap.release()
cv2.destroyAllWindows()
