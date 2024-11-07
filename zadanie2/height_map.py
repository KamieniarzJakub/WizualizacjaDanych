import matplotlib.pyplot as plt
import numpy as np
import colorsys

def hsv2rgb(h, s, v):
    # Convert HSV to RGB for matplotlib compatibility
    return colorsys.hsv_to_rgb(h, s, v)

def height_gradient(h, angle_factor):
    # Calculate color based on height and lighting angle (brightness change based on angle)
    return hsv2rgb((1 - h) / 3, 1, max(0, min(1, 1 - angle_factor)))  # Adjust brightness based on angle_factor

def load_data(file_name):
    # Load the data from file
    with open(file_name) as f:
        return f.readlines()

file = load_data("big.dem")
MAP_WIDTH, MAP_HEIGHT, DISTANCE = [int(x) for x in file[0].split()]
heights = [float(x) for line in file[1:] for x in line.split()]

# Prepare grid coordinates and normalize height data
x_axis = np.tile([x for x in range(0, MAP_WIDTH)], MAP_HEIGHT)
y_axis = np.repeat([x for x in range(0, MAP_HEIGHT)], MAP_WIDTH)
normalize_heights = (np.array(heights) - min(heights)) / (max(heights) - min(heights))

# Define light vector in 3D (e.g., direction of sunlight) and normalize it
light_vector = np.array([1, 1, 10^8])  # Example sunlight direction
light_vector = light_vector / np.linalg.norm(light_vector)  # Normalize the light vector

# Initialize arrays for gradient, colors, and lighting adjustments
colors = np.zeros((MAP_WIDTH * MAP_HEIGHT, 3))
angles = []

# Loop through each point to calculate slopes and apply lighting effect
for x in range(1, MAP_WIDTH - 1):
    for y in range(1, MAP_HEIGHT - 1):
        # Get height values at center, left, right, top, and bottom for gradient calculation
        h_center = normalize_heights[y * MAP_WIDTH + x]
        h_left = normalize_heights[y * MAP_WIDTH + x - 1]
        h_right = normalize_heights[y * MAP_WIDTH + x + 1]
        h_top = normalize_heights[(y - 1) * MAP_WIDTH - x]
        h_bottom = normalize_heights[(y + 1) * MAP_WIDTH + x]  # Corrected bottom coordinate

        # Create two vectors in 3D space to form the surface triangle
        vector_a = np.array([DISTANCE, 0, h_right - h_center])   # Vector from center to right
        vector_b = np.array([0, DISTANCE, h_bottom - h_center])  # Vector from center to bottom

        # Calculate the normal vector (perpendicular to the surface)
        slope_vector = np.cross(vector_a, vector_b)
        if np.linalg.norm(slope_vector) != 0:
            slope_vector /= np.linalg.norm(slope_vector)  # Normalize the normal vector

        # Calculate the angle between the light vector and the slope vector
        angle_factor = np.dot(light_vector, slope_vector)  # Cosine of angle between light and slope
        angles.append(angle_factor)

        # Adjust color based on height and lighting angle
        colors[y * MAP_WIDTH + x] = height_gradient(h_center, angle_factor)

# Plot the results
plt.scatter(x_axis, y_axis, c=colors)
plt.xlim(0, MAP_WIDTH)
plt.ylim(MAP_HEIGHT, 0)

print(max(angles))
print(min(angles))

# Configure axis appearance for better visualization
ax = plt.gca()
ax.set_xticks(range(0, MAP_WIDTH, MAP_WIDTH // 5))
ax.set_yticks(range(0, MAP_HEIGHT, MAP_WIDTH // 5))
ax.tick_params(left=True, right=True, top=True, bottom=True, direction='in')

# Save the generated map image
plt.savefig("map.png")
