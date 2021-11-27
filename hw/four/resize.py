import numpy as np
import cv2

path = "hw\Q4_Image\SQUARE-01.png"
img = cv2.imread(path)
cv2.imshow('Origin', img)

imgre = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
cv2.imshow('Resize',imgre)


cv2.waitKey(0)
cv2.destroyWindow()