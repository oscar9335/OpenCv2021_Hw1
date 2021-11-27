import numpy as np
import cv2

path = "hw\Q2_Image\Lenna_whiteNoise.jpg"
img = cv2.imread(path)


blur = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow('Origin', img)
cv2.imshow('Gaussian Blur', blur)

cv2.waitKey(0)
cv2.destroyWindow()