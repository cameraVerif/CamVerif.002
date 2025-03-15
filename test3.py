import matplotlib.pyplot as plt

# Sample points
points =  [[28.74525862, 31.57543103],
 [28.69519584, 31.57055479],
 [19.31135057,  7.99066092],
 [19.26778946,  8.00203882],
 [47.61307471,  7.99066092],
 [47.55000861,  8.00203882]]

##  Extract x and y coordinates
x_coords, y_coords = zip(*points)

# Grid properties
cell_size_cm = 2  # Each cell is 2 cm x 2 cm
num_cells = 49  # Grid size is 49 x 49
fig_size_cm = (num_cells * cell_size_cm) / 2.54  # Convert cm to inches

# Create the plot with the correct figure size
plt.figure(figsize=(fig_size_cm, fig_size_cm))
plt.scatter(x_coords, y_coords, color='red', label='Points')

# Set axis limits
plt.xlim(0, num_cells)
plt.ylim(0, num_cells)

# Set major ticks at every 1 unit (each cell corresponds to a 2cm area)
plt.xticks(range(0, num_cells + 1, 1))
plt.yticks(range(0, num_cells + 1, 1))

# Add grid with 1x1 spacing
plt.grid(True, linestyle='--', alpha=0.6)

# Labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("2D Points Plot with 2cm x 2cm Grid Cells")
plt.legend()

# Show the plot
plt.show()