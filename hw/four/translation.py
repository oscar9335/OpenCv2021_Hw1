import numpy as np
import cv2

path = "hw\Q4_Image\SQUARE-01.png"
img = cv2.imread(path)
cv2.imshow('Origin', img)

imgre = cv2.resize(img,(256,256))
#cv2.imshow('resize', imgre)

height, width = imgre.shape[:2]


# get tx and ty values for translation
# you can specify any value of your choice
tx, ty = 0 , 60

# create the translation matrix using tx and ty, it is a NumPy array 
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty]
], dtype=np.float32)


ranslated_image = cv2.warpAffine(src=imgre, M=translation_matrix, dsize=(400, 300))

name = "Resize"
#cv2.namedWindow(name,0)
#cv2.resizeWindow(name,400,300)
cv2.imshow(name,ranslated_image)



cv2.waitKey(0)
cv2.destroyWindow()