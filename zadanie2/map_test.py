import matplotlib.pyplot as plt
import numpy as np
import colorsys
import math

def hsv2rgb(h, s, v):
    return colorsys.hsv_to_rgb(h, s, v)

def height_gradient(h, dx, dy):
    v = dx*2+dy*3
    if v>0:
        return hsv2rgb((1-h)/3, 1-min(1, v*4), 1)
    else:
        return hsv2rgb((1-h)/3, 1, 1+max(-1, v*1.5))
    
def load_data(file_name):
    with open(file_name) as f:
        return f.readlines()

file = load_data("big.dem")
MAP_WIDTH, MAP_HEIGHT, DISTANCE = [int(x) for x in file[0].split()]
heights = [float(x) for line in file[1:] for x in line.split()]
light_vector = np.array([1, 1])
light_vector = light_vector / np.linalg.norm(light_vector)  # Normalize the light vector

x_axis = np.tile([x for x in range(0, MAP_WIDTH)], MAP_HEIGHT)
y_axis = np.repeat([x for x in range(0, MAP_HEIGHT)], MAP_WIDTH)
normalize_heights = (np.array(heights) - min(heights)) / (max(heights) - min(heights))
#heights_differences = np.zeros(MAP_WIDTH*MAP_HEIGHT)
dx_diffrences = []
dy_diffrences = []
colors = np.zeros((MAP_WIDTH*MAP_HEIGHT, 3))
angles = []

for x in range(1, MAP_WIDTH-1):
    for y in range(1, MAP_HEIGHT-1):
        h_center = normalize_heights[y * MAP_WIDTH + x]
        h_left = normalize_heights[y * MAP_WIDTH + x - 1]
        h_right = normalize_heights[y * MAP_WIDTH + x + 1]
        h_top = normalize_heights[(y - 1) * MAP_WIDTH + x]
        h_bottom = normalize_heights[(y - 1) * MAP_WIDTH - x]
        dx = h_center - h_left
        dy = h_center - h_top

        slope_vector = np.array([dx, dy])
        if np.linalg.norm(slope_vector) != 0:
            slope_vector /= np.linalg.norm(slope_vector)  # Normalize the normal vector

        # Calculate the angle between the light vector and the slope vector
        angle_factor = np.dot(light_vector, slope_vector)  # Cosine of angle between light and slope
        angles.append(angle_factor)

        dx_diffrences.append(dx)
        dy_diffrences.append(dy)

        #print(dx, dy, angle_factor)
        #print()

        colors[y*MAP_WIDTH + x] = height_gradient(normalize_heights[y*MAP_WIDTH + x], dx, dy)
#colors = np.array([height_gradient(height) for height in normalize_heights])
#print(max(heights_differences))
#print(min(heights_differences))

#print(max(angles))
#print(min(angles))

# print(max(dx_diffrences))
# print(min(dx_diffrences))
# print(max(dy_diffrences))
# print(min(dy_diffrences))
plt.figure(figsize=(10,10))
plt.scatter(x_axis, y_axis, c=colors)
plt.xlim(0, MAP_WIDTH)
plt.ylim(MAP_HEIGHT, 0)
ax = plt.gca()
ax.set_xticks(range(0, MAP_WIDTH, MAP_WIDTH//5))
ax.set_yticks(range(0, MAP_HEIGHT, MAP_WIDTH//5))  
ax.tick_params(left=True, right=True, top=True, bottom=True, direction='in')
plt.savefig("height_map.png")