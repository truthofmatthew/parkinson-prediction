import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Path to CSV file
csv_path = './dataset/turning_parkingson/Pt204_C_n_350.csv'

# Load keypoints from CSV
data = pd.read_csv(csv_path, header=None)
num_joints = 17
dimension = 2
keypoints = data.values.reshape(data.shape[0], num_joints, dimension)

# Connections between joints
connections = [
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
    [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
    [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]
]
LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

# Colors
lcolor = (255, 0, 0)  # Left connections
rcolor = (0, 0, 255)  # Right connections

# Create figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 640)
ax.set_ylim(0, 480)
ax.invert_yaxis()
ax.set_aspect('equal')

# Function to plot skeleton on a frame
def plot_skeleton(frame_idx, ax):
    ax.clear()
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(f"Frame {frame_idx}")

    kps = keypoints[frame_idx]

    for j, (start_idx, end_idx) in enumerate(connections):
        start = kps[start_idx].astype(int)
        end = kps[end_idx].astype(int)
        color = lcolor if LR[j] else rcolor

        ax.plot([start[0], end[0]], [start[1], end[1]], 'o-', color=np.array(color) / 255, markersize=3, linewidth=1)
        ax.plot(start[0], start[1], 'o', color="green", markersize=4)
        ax.plot(end[0], end[1], 'o', color="green", markersize=4)

# Update function for animation
def update(frame_idx):
    plot_skeleton(frame_idx, ax)

# Create animation
ani = FuncAnimation(fig, update, frames=len(keypoints), interval=100, repeat=False)

# Save the animation as a GIF
ani.save('./output/turning_gif.gif', writer='pillow', fps=5)