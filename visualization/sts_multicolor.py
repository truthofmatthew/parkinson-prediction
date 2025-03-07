import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
data_path = '../dataset/sts_parkinson/Pt744_PD_n_545_fix.csv'  # Replace with your CSV file path
data = pd.read_csv(data_path)



# Extract time and joint coordinates
time = data['time (s)']
num_joints = 25

# Function to plot skeleton
def plot_skeleton(frame, ax, color):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13),
        (13, 14), (0, 15), (15, 17), (0, 16), (16, 18),
        (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)
    ]

    for connection in connections:
        joint1, joint2 = connection
        x1, y1 = frame[joint1 * 2], frame[joint1 * 2 + 1]
        x2, y2 = frame[joint2 * 2], frame[joint2 * 2 + 1]

        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:  # Valid coordinates
            ax.plot([x1, x2], [y1, y2], 'o-', color=color, markersize=3, linewidth=1)

# Select the first frame for each second
frames_per_second = 5  # Assuming 29 frames per second
first_frames = data.iloc[::frames_per_second, :]

# Create a single plot
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 640)  # Adjust based on your data resolution
ax.set_ylim(0, 480)  # Adjust based on your data resolution
ax.invert_yaxis()
ax.set_aspect('equal')
ax.set_title("Skeleton Movement - One Frame Per Second")

# Define a colormap for different frames
colors = plt.cm.viridis(np.linspace(0, 1, len(first_frames)))

for i, (row, color) in enumerate(zip(first_frames.iterrows(), colors)):
    frame = row[1][2:].values  # Skip frame number and time columns
    plot_skeleton(frame, ax, color)

# Save the plot as an image
plt.savefig('../output/sts_stillimage_plot.png', dpi=300, bbox_inches='tight')

# Optionally, show the plot
# plt.show()
