import numpy as np
import cv2

#load img
path1 = "hw\Q1_Image\Dog_Strong.jpg"
path2 = "hw\Q1_Image\Dog_Weak.jpg"
img1 = cv2.imread(path2)
img2 = cv2.imread(path1)



title_window = 'Blending'

def on_trackbar(val):    #0 <= value <=255    change with the bar drag
    alpha = val / 255
    beta = ( 1.0 - alpha )      #alpha is img1 transparency, beta is img2 transparency
    dst = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
    cv2.imshow(title_window, dst)


cv2.namedWindow(title_window)
cv2.createTrackbar("Blend", title_window , 0, 255, on_trackbar)
cv2.setTrackbarPos("Blend",title_window,0)
# Show some stuff
on_trackbar(0)

cv2.waitKey(0)
cv2.destroyWindow()