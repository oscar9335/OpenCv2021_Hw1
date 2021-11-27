import numpy as np
import cv2

path = "hw\Q1_Image\Sun.jpg"

img = cv2.imread(path)

b,g,r = cv2.split(img)
merged = cv2.merge([b,g,r])



zeros = np.zeros(img.shape[:2], dtype = "uint8")

cv2.imshow('Hw1-2 blue', cv2.merge([b,zeros,zeros]))
cv2.imshow('Hw1-2 green', cv2.merge([zeros,g,zeros]))
cv2.imshow('Hw1-2 red', cv2.merge([zeros,zeros,r]))

cv2.waitKey(0)
cv2.destroyAllWindows()