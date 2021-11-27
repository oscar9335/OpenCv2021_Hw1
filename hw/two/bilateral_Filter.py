import numpy as np
import cv2

path = "hw\Q2_Image\Lenna_whiteNoise.jpg"
img = cv2.imread(path)

bilateral = cv2.bilateralFilter(img, 9, 90, 90)

cv2.imshow('Origin', img)
cv2.imshow('Bilateral Blur15', bilateral)

cv2.waitKey(0)
cv2.destroyWindow()