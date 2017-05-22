#
# Implemntation of the "Stroke Width Transformation"-algorithm
# for detecting text in natural images.
#

# Libs
import cv2 # OpenCV
from skimage import feature # Alternative lib to OpenCV
import matplotlib.pyplot as plt # matplotlib for img display

# 1. Step: Canny Edge Detection




# Playing around

#-------------------------+
# Image Preparation
#-------------------------+

# Load test image
img = cv2.imread("/Users/basti/Documents/Projekte/PySWT/test imgs/IMG_2834_2.jpg",0)

# Resize to 15%
img = cv2.resize(img, dsize=(0,0), fx=0.15, fy=0.15)

# Display image
plt.imshow(img, cmap="gray")

#-------------------------+
# Edge Detection
#-------------------------+

# Canny edge detection
edges_cv = cv2.Canny(img, 2, 5) # cv2
plt.imshow(edges_cv, cmap="gray")

edges_ski = feature.canny(img,0.1) # skimage
plt.imshow(edges_ski, cmap="gray")

# Laplacian edge detection !WINNER for resized image!
edges_laplace = cv2.Laplacian(img,cv2.CV_64F)
plt.imshow(edges_laplace, cmap="gray")

# Scharr edge detection
edges_scharrx = cv2.Scharr(img, ddepth=2, dx=1, dy=0)
edges_scharry = cv2.Scharr(img, ddepth=2, dx=0, dy=1)
plt.imshow(edges_scharrx, cmap="gray")
plt.imshow(edges_scharry, cmap="gray")
plt.imshow(edges_scharry+edges_scharrx, cmap="gray")

# Sobel edge detection
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

plt.imshow(sobelx, cmap="gray")
plt.imshow(sobely, cmap="gray")
plt.imshow(sobelx+sobely, cmap="gray")

#-------------------------+
# Make edge image binary
#-------------------------+
plt.imshow(edges_laplace, cmap="gray")

# Summary of edges
edg.max()
edg.min()
plt.hist(edg.flatten())

# Make binary
edg[edg<150] = 0


