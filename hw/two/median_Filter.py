import numpy as np
import cv2

path = "hw\Q2_Image\Lenna_pepperSalt.jpg"
img = cv2.imread(path)


median3 = cv2.medianBlur(img, 3)
median5 = cv2.medianBlur(img, 5)
cv2.imshow('Origin', img)
cv2.imshow('Median Blur 3x3', median3)
cv2.imshow('Median Blur 5x5', median5)


cv2.waitKey(0)
cv2.destroyWindow()