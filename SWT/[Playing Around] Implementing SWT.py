#-------------------------------------------------------+
# Trying to implement the stroke width transformation
# Took some code from: https://github.com/mypetyak/StrokeWidthTransform/blob/master/swt.py
#-------------------------------------------------------+

# Libs
import os
import numpy as np
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

edges_locations = np.argwhere(edg_mtg != 0) # indices of all pixels that are edges based on canny

