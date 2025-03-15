import matplotlib.pyplot as plt
import numpy as np



import matplotlib.pyplot as plt
import numpy as np

# Define the coordinates of two triangles with float values
# Define the coordinates of two triangles
triangle1 = np.array([[28.74525862068965, 31.575431034482758], [19.31135057471264, 7.990660919540231] ,
                      [47.61307471264368, 7.990660919540231], [28.74525862068965, 31.575431034482758],
])
triangle2 = np.array([[28.695195841948085, 31.57055478980014], [19.267789455547895, 8.002038823799671], 
                      [47.550008614748464, 8.002038823799671], [28.695195841948085, 31.57055478980014],
])
# Combine all points to determine axis limits
all_points = np.vstack([triangle1, triangle2])
x_min, x_max = int(np.floor(min(all_points[:, 0]))), int(np.ceil(max(all_points[:, 0])))
y_min, y_max = int(np.floor(min(all_points[:, 1]))), int(np.ceil(max(all_points[:, 1])))

# Grid properties
cell_size_cm = 5  # Each grid cell is 5cm x 5cm
num_x_cells = x_max - x_min + 1
num_y_cells = y_max - y_min + 1
fig_size_cm = (num_x_cells * cell_size_cm) / 2.54, (num_y_cells * cell_size_cm) / 2.54  # Convert cm to inches

# Create a new figure with the correct size
plt.figure(figsize=fig_size_cm)

# Draw the first triangle (blue lines)
plt.plot(triangle1[:, 0], triangle1[:, 1], color='blue', linewidth=2, label="Triangle 1")

# Draw the second triangle (red lines)
plt.plot(triangle2[:, 0], triangle2[:, 1], color='red', linewidth=2, label="Triangle 2")

# Set axis limits based on grid cells
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Set major ticks at every 1 unit to create a 1x1 grid
plt.xticks(range(x_min, x_max + 1, 1))
plt.yticks(range(y_min, y_max + 1, 1))

# Add a grid with 1x1 spacing
plt.grid(True, linestyle="--", alpha=0.6)

# Add labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Two Triangles in 2D Plane with 5cm x 5cm Grid Cells")

# Show legend
plt.legend()

# Show the plot
plt.show()