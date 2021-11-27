import numpy as np
import cv2

path = "hw\Q1_Image\Sun.jpg"

#
img = cv2.imread(path)
b,g,r = cv2.split(img)

result = b/3 + g/3 + r/3

myArray = np.asarray(result, dtype=np.uint8)
print(myArray)
cv2.imshow('(r+g+b)/3', myArray)



#Transform “Sun.jpg” into grayscale image I1 by calling OpenCV function directly
img_gray = cv2.imread(path)
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
cv2.imshow('Call function directly', img_gray)
#print(img_gray)


cv2.waitKey(0)
cv2.destroyAllWindows()

