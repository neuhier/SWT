#
# Implemntation of the "Stroke Width Transformation"-algorithm
# for detecting text in natural images.
#

# Libs
import os
import numpy as np
import cv2 # OpenCV
import matplotlib.pyplot as plt # matplotlib for img display

# Helper functions

# Plot multiple images in one window
def show_img_list(lst, ncol=2, titles=[]):
    n = len(lst)
    nrow = np.ceil(n/ncol)
    counter = 1
    fig = plt.figure()
    plt.set_cmap('gray')
    for i in lst:
        a = fig.add_subplot(nrow, ncol, counter)
        a.axis('off')
        a.set_xticklabels([])
        a.set_yticklabels([])
        plt.imshow(i)
        if len(titles)>0:
            a.set_title(titles[counter-1])
        counter+=1
    fig.subplots_adjust(wspace=.1, hspace=.1)

#----------------------------------------------------------+
# 1. Step: Canny Edge Detection
#----------------------------------------------------------+

# Load different test images
cwd = os.getcwd() # Current working directory

messi       = cv2.imread(os.path.join(cwd, "Sample Imgs/messi.jpg"), 0) # Can I reproduce the edges shown in multiple tutorials?
mtg_blurry  = cv2.imread(os.path.join(cwd, "Sample Imgs/IMG_2834.jpg"),0) # A bit of a blurry pic of mtg cards
mtg_cropped = cv2.imread(os.path.join(cwd, "Sample Imgs/IMG_2855_2.jpg"),0) # Better shot of mtg cards, cropped to card titels


#----------------------------------------------------------+
# Trying different edge detection algorithms on the pics
# First: Canny
#----------------------------------------------------------+

edg_messi = cv2.Canny(messi, 100, 200)
edg_mtg_blurr = cv2.Canny(mtg_blurry, 25, 200)
edg_mtg_cropped = cv2.Canny(mtg_cropped, 25, 200)

canny_edg = [messi, edg_messi, mtg_blurry, edg_mtg_blurr, mtg_cropped, edg_mtg_cropped]

show_img_list(canny_edg)


#----------------------------------------------------------+
# Trying different edge detection algorithms on the pics
# First: Canny + some img preprocessing
#----------------------------------------------------------+

# Contrast Limited Adaptive Histogram Equalization
clahe = cv2.createCLAHE(clipLimit=2.0)
mtg_blurry_contr = clahe.apply(mtg_blurry)
mtg_cropped_contr = clahe.apply(mtg_cropped)

edg_mtg_blurr_contr = cv2.Canny(mtg_blurry_contr, 90, 225)
edg_mtg_cropped_contr = cv2.Canny(mtg_cropped_contr,90, 225)

canny_contr_edg = [mtg_blurry_contr, edg_mtg_blurr_contr, mtg_cropped_contr, edg_mtg_cropped_contr]
show_img_list(canny_contr_edg)


#----------------------------------------------------------+
# Systematic approach to figure out right params for
# clahe and canny for mtg_cropped
#----------------------------------------------------------+

cliplimits = [1.5, 2]
canny1 = [75, 100, 150]
canny2 = [250]
titles = []

param_optim = []

for i,j,k in [(i,j,k) for i in  cliplimits for j in canny1 for k in canny2]:
    titles.append("ClipsLimits: " + str(i) + " c1: " + str(j) + " c2: " + str(k))
    clahe = cv2.createCLAHE(clipLimit=i)
    thisimg = clahe.apply(mtg_cropped)
    thisedg = cv2.Canny(thisimg, j,k)
    param_optim.append(thisedg)


show_img_list(param_optim, 3, titles) # cliplimit: 1.5 c1: 100 c2: 250 winner


#----------------------------------------------------------+
# Create winner edge img and save it for the next steps
#----------------------------------------------------------+

clahe = cv2.createCLAHE(clipLimit=1.5) # Contrast adjustment
mtg_final = clahe.apply(mtg_cropped) # Generate contrast adjusted img
edg_mtg_final = cv2.Canny(mtg_final, 100, 250) # Run canny on it
cv2.imwrite(os.path.join(cwd, "Edge Imgs/mtg_cropped.jpg"), edg_mtg_final)

