import numpy as np
import cv2

path = "hw\Q1_Image\Sun.jpg"
img = cv2.imread(path)
#print(type(img))


#cv2.namedWindow('Hw1-1', cv2.WINDOW_NORMAL) # 讓視窗可以自由縮放大小
cv2.imshow('Hw1-1', img)
h, w, c = img.shape
print('height: ', h)
print('width:  ', w)
#print('channel:', c)


cv2.waitKey(0)
cv2.destroyWindow()

