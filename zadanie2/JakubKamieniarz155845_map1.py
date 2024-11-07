import matplotlib.pyplot as plt
import numpy as np
import colorsys

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
normalize_heights = (np.array(heights) - min(heights)) / (max(heights) - min(heights))

colors = np.zeros((MAP_WIDTH*MAP_HEIGHT, 3))
for x in range(1, MAP_WIDTH):
    for y in range(1, MAP_HEIGHT):
        h_center = normalize_heights[y * MAP_WIDTH + x]
        h_left = normalize_heights[y * MAP_WIDTH + x - 1]
        h_top = normalize_heights[(y - 1) * MAP_WIDTH + x]

        dx = h_center - h_left
        dy = h_center - h_top
        colors[y*MAP_WIDTH + x] = height_gradient(h_center, dx, dy)

x_axis = np.tile([x for x in range(0, MAP_WIDTH)], MAP_HEIGHT)
y_axis = np.repeat([x for x in range(0, MAP_HEIGHT)], MAP_WIDTH)

plt.figure(figsize=(10,10))
plt.scatter(x_axis, y_axis, c=colors)
plt.xlim(0, MAP_WIDTH)
plt.ylim(MAP_HEIGHT, 0)
ax = plt.gca()
ax.set_xticks(range(0, MAP_WIDTH, MAP_WIDTH//5))
ax.set_yticks(range(0, MAP_HEIGHT, MAP_HEIGHT//5))  
ax.tick_params(left=True, right=True, top=True, bottom=True, direction='in')
plt.savefig("JakubKamieniarz155845_map1.png")