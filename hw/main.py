from PyQt5 import QtWidgets, QtGui, QtCore
import cv2
import numpy as np
from scipy import signal

import sys
import Ui_hw1

class MainUi(QtWidgets.QWidget, Ui_hw1.Ui_Dialog):
    def __init__(self, parent = None):
        super(MainUi, self).__init__(parent)
        self.setupUi(self)


        self.button1_1.clicked.connect(self.button1_1Clicked)
        self.button1_2.clicked.connect(self.button1_2Clicked)
        self.button1_3.clicked.connect(self.button1_3Clicked)
        self.button1_4.clicked.connect(self.button1_4Clicked)
        self.button2_1.clicked.connect(self.button2_1Clicked)
        self.button2_2.clicked.connect(self.button2_2Clicked)
        self.button2_3.clicked.connect(self.button2_3Clicked)
        self.button3_1.clicked.connect(self.button3_1Clicked)
        self.button3_2.clicked.connect(self.button3_2Clicked)
        self.button3_3.clicked.connect(self.button3_3Clicked)
        self.button3_4.clicked.connect(self.button3_4Clicked)
        self.button4_1.clicked.connect(self.button4_1Clicked)
        self.button4_2.clicked.connect(self.button4_2Clicked)
        self.button4_3.clicked.connect(self.button4_3Clicked)
        self.button4_4.clicked.connect(self.button4_4Clicked)
       


    def button1_1Clicked(self):
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
    def button1_2Clicked(self):
        path = "hw\Q1_Image\Sun.jpg"
        img = cv2.imread(path)
        b,g,r = cv2.split(img)
        merged = cv2.merge([b,g,r])
        zeros = np.zeros(img.shape[:2], dtype = "uint8")
        cv2.imshow('Hw1-2 blue', cv2.merge([b,zeros,zeros]))
        cv2.imshow('Hw1-2 green', cv2.merge([zeros,g,zeros]))
        cv2.imshow('Hw1-2 red', cv2.merge([zeros,zeros,r]))
        cv2.waitKey(0)
    def button1_3Clicked(self):
        path = "hw\Q1_Image\Sun.jpg"
        img = cv2.imread(path)
        b,g,r = cv2.split(img)
        result = b/3 + g/3 + r/3
        myArray = np.asarray(result, dtype=np.uint8)
        #print(myArray)
        cv2.imshow('(r+g+b)/3', myArray)
        #Transform “Sun.jpg” into grayscale image I1 by calling OpenCV function directly
        img_gray = cv2.imread(path)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Call function directly', img_gray)
        cv2.waitKey(0)
    def button1_4Clicked(self):   
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
    def button2_1Clicked(self):
        path = "hw\Q2_Image\Lenna_whiteNoise.jpg"
        img = cv2.imread(path)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imshow('Origin', img)
        cv2.imshow('Gaussian Blur', blur)
        cv2.waitKey(0)
    def button2_2Clicked(self):
        path = "hw\Q2_Image\Lenna_whiteNoise.jpg"
        img = cv2.imread(path)
        bilateral = cv2.bilateralFilter(img, 9, 90, 90)
        cv2.imshow('Origin', img)
        cv2.imshow('Bilateral Blur15', bilateral)
        cv2.waitKey(0)
    def button2_3Clicked(self):
        path = "hw\Q2_Image\Lenna_pepperSalt.jpg"
        img = cv2.imread(path)
        median3 = cv2.medianBlur(img, 3)
        median5 = cv2.medianBlur(img, 5)
        cv2.imshow('Origin', img)
        cv2.imshow('3x3', median3)
        cv2.imshow('5x5', median5)
        cv2.waitKey(0)
    def button3_1Clicked(self):
        path = "hw\Q3_Image\House.jpg"
        img = cv2.imread(path)
        h, w, c = img.shape
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale', img_gray)
        #3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        #[[-1 -1 -1]
        # [ 0  0  0]
        # [ 1  1  1]]
        #[[-1  0  1]
        # [-1  0  1]
        # [-1  0  1]]
        gaussian_kernel = np.exp(-(x**2+y**2))
        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        #[[0.04491922 0.12210311 0.04491922] 
        # [0.12210311 0.33191066 0.12210311] 
        # [0.04491922 0.12210311 0.04491922]]
        img_gray = signal.convolve(img_gray,gaussian_kernel)
        img_gray = img_gray.astype(np.uint8)
        cv2.imshow('Gaussian Blur', img_gray)
        #print(img_gray)
        cv2.waitKey(0)
    def button3_2Clicked(self):
        path = "hw\Q3_Image\House.jpg"
        img = cv2.imread(path)
        h, w, c = img.shape
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        img_gray = signal.convolve(img_gray,gaussian_kernel)
        img_gray = img_gray.astype(np.uint8)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        img_x = np.zeros(img_gray.shape[:2])   # create a array
        for i in range(h):
            for j in range(w):
                img_x[i,j] = abs(np.sum(sobel_x * img_gray[i:i + 3, j:j + 3]))   #3x3 matrix cross and abs             
        img_x = img_x.astype(np.uint8)
        cv2.imshow('Sobel X', img_x)
        cv2.waitKey(0)
    def button3_3Clicked(self):
        path = "hw\Q3_Image\House.jpg"
        img = cv2.imread(path)
        h, w, c = img.shape
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x  , y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        img_gray = signal.convolve(img_gray,gaussian_kernel)
        img_gray = img_gray.astype(np.uint8)
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        img_y = np.zeros(img_gray.shape[:2])   # create a array
        for i in range(h):
            for j in range(w):
                img_y[i,j] = abs(np.sum(sobel_y * img_gray[i:i + 3, j:j + 3]))   #3x3 matrix cross and abs           
        img_y = img_y.astype(np.uint8)
        cv2.imshow('Sobel Y', img_y)
        cv2.waitKey(0)
    def button3_4Clicked(self):
        path = "hw\Q3_Image\House.jpg"
        img = cv2.imread(path)
        h, w, c = img.shape
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        img_gray = signal.convolve(img_gray,gaussian_kernel)
        img_gray = img_gray.astype(np.uint8)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        img_x = np.zeros(img_gray.shape[:2])   # create a array
        img_y = np.zeros(img_gray.shape[:2])
        magnitude = np.zeros(img_gray.shape[:2])
        for i in range(h):
            for j in range(w):
                img_x[i,j] = abs(np.sum(sobel_x * img_gray[i:i + 3, j:j + 3]))   #3x3 matrix cross and abs 
                img_y[i,j] = abs(np.sum(sobel_y * img_gray[i:i + 3, j:j + 3]))               
        img_x = img_x.astype(np.int32)
        img_y = img_y.astype(np.int32)
        for i in range(h) :
            for j in range(w):
                magnitude[i, j] = abs((img_x[i, j])*(img_x[i, j]) + (img_y[i, j])*(img_y[i, j]))**0.5    
        magnitude = magnitude.astype(np.uint8)
        cv2.imshow('Magnitude', magnitude)       
        cv2.waitKey(0)
    def button4_1Clicked(self):
        path = "hw\Q4_Image\SQUARE-01.png"
        img = cv2.imread(path)
        cv2.imshow('Origin', img)
        imgre = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)
        cv2.imshow('Resize',imgre)
        cv2.waitKey(0)
    def button4_2Clicked(self):
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
        name = "Translation"
        #cv2.namedWindow(name,0)
        #cv2.resizeWindow(name,400,300)
        cv2.imshow(name,ranslated_image)
        cv2.waitKey(0)
    def button4_3Clicked(self):
        path = "hw\Q4_Image\SQUARE-01.png"
        img = cv2.imread(path)
        cv2.imshow('Origin', img)
        imgre = cv2.resize(img,(256,256))
        height, width = imgre.shape[:2]
        retval=cv2.getRotationMatrix2D(((width-1)/2.0,(height-1)/2.0), 10, 0.5)
        rotate = cv2.warpAffine(src=imgre, M=retval, dsize=(400, 300))
        name = "Angle"
        #cv2.namedWindow(name,0)
        #cv2.resizeWindow(name,400,300)
        cv2.imshow(name,rotate)
        cv2.waitKey(0)
    def button4_4Clicked(self):
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
        name = "Shearing"
        #cv2.namedWindow(name,0)
        #cv2.resizeWindow(name,400,300)
        cv2.imshow(name,result)
        cv2.waitKey(0)
    
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = MainUi()
    ui.show()
    sys.exit(app.exec_())