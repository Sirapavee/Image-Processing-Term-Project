import cv2 as cv
import imutils
import numpy as np

#read image
img = cv.imread("advertising7.jpg")
#copy image for use as result
img2 = img.copy()
#resize image
img = imutils.resize(img, 640, 640)

#convert BGR image to HSV
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#convert BGR image to grayscale
imgGray1 = cv.cvtColor(imgHSV, cv.COLOR_BGR2GRAY)
imgGray2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#create lower and upper bound for filter light blue color out
lo=np.array([30,0,0])
hi=np.array([100,255,255])
#create mask
mask=cv.inRange(imgHSV,lo,hi)
img[mask>0]=(0,0,0)

#Image Thresholding
_, thresh1 = cv.threshold(imgGray1, 150, 255, cv.THRESH_BINARY_INV)
_, thresh2 = cv.threshold(imgGray2, 150, 255, cv.THRESH_BINARY_INV)

#Morphological Transformations
closing1 = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, (3,3), iterations=1)
closing2 = cv.morphologyEx(thresh2, cv.MORPH_CLOSE, (3,3), iterations=1)

#Find contours of image
contours1, heirachy1 = cv.findContours(closing1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contours2, heirachy2 = cv.findContours(closing2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

for contour in contours1:
    #find and draw rectangle on contour area in range (600, 6000)
    if cv.contourArea(contour)>200 and cv.contourArea(contour)<6000:

        #ignore contour area in range (220, 230) and (1050, 1100)
        if  cv.contourArea(contour)>1050 and cv.contourArea(contour)<1100 or cv.contourArea(contour)>220 and cv.contourArea(contour)<230:
            continue
            
        #compute bounding rectangle
        x, y, w, h = cv.boundingRect(contour)

        #draw bounding rectangle on contour
        cv.rectangle(img2, (x,y), (x+w,y+h), (0,255,0), 2)

for contour in contours2:
    #find and draw rectangle on contour area in range (90, 2500)
    if cv.contourArea(contour)>90 and cv.contourArea(contour)<2500:
        
        #ignore contour area in range (205, 210), (238, 239) and (445, 460)
        if cv.contourArea(contour)>238 and cv.contourArea(contour)<239 or cv.contourArea(contour)>205 and cv.contourArea(contour)<210 or cv.contourArea(contour)>445 and cv.contourArea(contour)<460:
            continue

        #compute bounding rectangle
        x, y, w, h = cv.boundingRect(contour)

        #draw bounding rectangle on contour
        cv.rectangle(img2, (x,y), (x+w,y+h), (0,255,0), 2)

#showing image
cv.imshow('Result', img2)
#press x on keyboard to close
if cv.waitKey(0) & 0xFF == ord('x'):
    cv.destroyAllWindows()

#save result image
cv.imwrite('result7.jpg', img2)