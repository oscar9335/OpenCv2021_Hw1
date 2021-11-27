import numpy as np
import cv2
from scipy import signal

path = "hw\Q3_Image\House.jpg"
img = cv2.imread(path)

h, w, c = img.shape

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x, y = np.mgrid[-1:2, -1:2]
gaussian_kernel = np.exp(-(x**2+y**2))
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
img_gray = signal.convolve(img_gray,gaussian_kernel)
img_gray = img_gray.astype(np.uint8)

sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_y = np.zeros(img_gray.shape[:2])   # create a array

for i in range(h):
    for j in range(w):
        img_y[i,j] = abs(np.sum(sobel_y * img_gray[i:i + 3, j:j + 3]))   #3x3 matrix cross and abs 
                
img_y = img_y.astype(np.uint8)


cv2.imshow('hw3_3', img_y)


cv2.waitKey(0)
cv2.destroyWindow()