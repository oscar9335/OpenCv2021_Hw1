import numpy as np
import cv2
from scipy import signal

path = "hw\Q3_Image\House.jpg"
img = cv2.imread(path)

h, w, c = img.shape

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', img_gray)


#3*3 Gassian filter
x, y = np.mgrid[-1:2, -1:2]
#[[-1 -1 -1]
# [ 0  0  0]
# [ 1  1  1]]
#[[-1  0  1]
# [-1  0  1]
# [-1  0  1]]
gaussian_kernel = np.exp(-(x**2+y**2))
#Normalization
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

#[[0.04491922 0.12210311 0.04491922] 
# [0.12210311 0.33191066 0.12210311] 
# [0.04491922 0.12210311 0.04491922]]

img_gray = signal.convolve(img_gray,gaussian_kernel)
img_gray = img_gray.astype(np.uint8)
cv2.imshow('hw3_1', img_gray)

#print(img_gray)


cv2.waitKey(0)
cv2.destroyWindow()