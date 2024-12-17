import matplotlib.pyplot as plt
import numpy as np
import colorsys

def hsv2rgb(h, s, v):
    return colorsys.hsv_to_rgb(h, s, v)

def height_gradient(h, angle_factor):
    angle_factor **= 3
    if angle_factor > 0.95:
        return hsv2rgb((1-h)/3, 1-min(0.8, angle_factor*0.7), 1)
    if angle_factor>0:
        return hsv2rgb((1-h)/3, 1-min(0.3, angle_factor), 1)
    else:
        return hsv2rgb((1-h)/3, 1, 1+max(-0.1, angle_factor))

def load_data(file_name):
    with open(file_name) as f:
        return f.readlines()

file = load_data("big.dem")
MAP_WIDTH, MAP_HEIGHT, DISTANCE = [int(x) for x in file[0].split()]
heights = [float(x) for line in file[1:] for x in line.split()]
normalize_heights = (np.array(heights) - min(heights)) / (max(heights) - min(heights))

light_vector = np.array([1, 1])
light_vector = light_vector / np.linalg.norm(light_vector)

colors = np.zeros((MAP_WIDTH*MAP_HEIGHT, 3))
for x in range(1, MAP_WIDTH-1):
    for y in range(1, MAP_HEIGHT-1):
        h_center = normalize_heights[y * MAP_WIDTH + x]
        h_left = normalize_heights[y * MAP_WIDTH + x - 1]
        #h_right = normalize_heights[y * MAP_WIDTH + x + 1]
        h_top = normalize_heights[(y - 1) * MAP_WIDTH + x]
        #h_bottom = normalize_heights[(y + 1) * MAP_WIDTH + x]
        dx = h_center - h_left
        dy = h_center - h_top

        slope_vector = np.array([dx, dy])
        if np.linalg.norm(slope_vector) != 0:
            slope_vector /= np.linalg.norm(slope_vector)

        angle_factor = np.dot(light_vector, slope_vector)
        colors[y*MAP_WIDTH + x] = height_gradient(normalize_heights[y*MAP_WIDTH + x], angle_factor)

x_axis = np.tile([x for x in range(0, MAP_WIDTH)], MAP_HEIGHT)
y_axis = np.repeat([x for x in range(0, MAP_HEIGHT)], MAP_WIDTH)

plt.figure(figsize=(10, 10))
plt.scatter(x_axis, y_axis, c=colors, s=1)
plt.xlim(0, MAP_WIDTH)
plt.ylim(MAP_HEIGHT, 0)
ax = plt.gca()
ax.set_xticks(range(0, MAP_WIDTH, MAP_WIDTH//5))
ax.set_yticks(range(0, MAP_HEIGHT, MAP_WIDTH//5))  
ax.tick_params(left=True, right=True, top=True, bottom=True, direction='in')
plt.savefig("JakubKamieniarz155845_map2.png")