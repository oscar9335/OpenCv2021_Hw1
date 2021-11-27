import numpy as np
import cv2

path = "hw\Q4_Image\SQUARE-01.png"
img = cv2.imread(path)
cv2.imshow('Origin', img)

imgre = cv2.resize(img,(256,256))

height, width = imgre.shape[:2]

retval=cv2.getRotationMatrix2D(((width-1)/2.0,(height-1)/2.0), 10, 0.5)
rotate = cv2.warpAffine(src=imgre, M=retval, dsize=(400, 300))

#use the rotate result image to do the following things
pts1 = np.float32([[50,50],[200,50],[50,200],[200,200]])
pts2 = np.float32([[10,100],[200,50],[100,250],[290,200]])
M = cv2.getPerspectiveTransform(pts1,pts2)
result = cv2.warpPerspective(rotate,M,(400,300))


name = "Resize"
#cv2.namedWindow(name,0)
#cv2.resizeWindow(name,400,300)
cv2.imshow(name,result)



cv2.waitKey(0)
cv2.destroyWindow()