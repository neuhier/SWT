#-------------------------------------------------------+
# Trying to implement the stroke width transformation
# Took some code from: https://github.com/mypetyak/StrokeWidthTransform/blob/master/swt.py
#-------------------------------------------------------+

# Libs
import os
import numpy as np
import math
import cv2 # OpenCV
import matplotlib.pyplot as plt # matplotlib for img display


#-------------------------------------------------------+
# Load Edge Image
#-------------------------------------------------------+
cwd = os.getcwd() # Current working directory
edg_mtg       = cv2.imread(os.path.join(cwd, "Edge Imgs/mtg_cropped.jpg"), 0) # Can I reproduce the edges shown in multiple tutorials?

#-------------------------------------------------------+
# Find stroke derivates
#-------------------------------------------------------+
dx = cv2.Sobel(edg_mtg, cv2.CV_64F, 1, 0, ksize=5)
dy = cv2.Sobel(edg_mtg, cv2.CV_64F, 0, 1, ksize=5)

# cv2.imwrite(os.path.join(cwd, "Edge Imgs/sobelx.jpg"), dx)
# cv2.imwrite(os.path.join(cwd, "Edge Imgs/sobely.jpg"), dy)

theta = np.arctan2(dy, dx) # Derivates for each pixel

# cv2.imwrite(os.path.join(cwd, "Edge Imgs/theta.jpg"), (theta + np.pi)*255/(2*np.pi))

#-------------------------------------------------------+
# Perform Stroke Width Transform
#-------------------------------------------------------+
swt = np.empty(theta.shape)
swt[:] = np.Infinity
rays = []

step_x_g = -dx
step_y_g = -dy

magnitudes = np.sqrt(np.square(dx) + np.square(dy))

grad_x_g = step_x_g/magnitudes
grad_y_g = step_y_g/magnitudes

edges_locations = np.argwhere(edg_mtg != 0) # indices of all pixels that are edges based on canny

for edg in edges_locations:
    step_x = step_x_g[edg[0], edg[1]]
    step_y = step_y_g[edg[0], edg[1]]
    mag = magnitudes[edg[0], edg[1]]
    grad_x = grad_x_g[edg[0], edg[1]]
    grad_y = grad_y_g[edg[0], edg[1]]

    ray = []
    ray.append((edg[0], edg[1]))
    prev_x, prev_y, i = edg[0], edg[1], 0

    while True:
        i += 1
        cur_x = np.floor(edg[0] + grad_x * i)
        cur_y = np.floor(edg[1] + grad_y * i)

        if cur_x != prev_x or cur_y != prev_y:
            # we have moved to the next pixel!
            try:
                if edg_mtg[cur_y, cur_x] > 0:
                    # found edge,
                    ray.append((cur_x, cur_y))
                    theta_point = theta[edg[1], edg[0]]
                    alpha = theta[cur_y, cur_x]
                    if np.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) < np.pi/2.0:
                        thickness = np.sqrt( (cur_x - edg[0]) * (cur_x - edg[0]) + (cur_y - edg[1]) * (cur_y - edg[1]) )
                        for (rp_x, rp_y) in ray:
                            swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                        rays.append(ray)
                    break
                # this is positioned at end to ensure we don't add a point beyond image boundary
                ray.append((cur_x, cur_y))
            except IndexError:
                # reached image boundary
                break
            prev_x = cur_x
            prev_y = cur_y

# Compute median SWT
for ray in rays:
    median = np.median([swt[y, x] for (x, y) in ray])
    for (x, y) in ray:
        swt[y, x] = min(median, swt[y, x])

# cv2.imwrite(os.path.join(cwd, "Edge Imgs/swt.jpg"), swt)
