import cv2 as cv
import imutils

#read image
img = cv.imread("advertising3.jpg")

#convert image to grayscale
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Image Thresholding
_, thresh = cv.threshold(imgGray, 180, 255, cv.THRESH_BINARY_INV)

#Morphological Transformations
closing = cv.morphologyEx(thresh, cv.MORPH_OPEN, (3,3), iterations=1)

#Find contours of image
contours, heirachy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

for contour in contours:
    #find and draw rectangle on contour area in range (300, 4200) but not in range that listed in forbiddenAreaRange
    #forbiddenAreaRange = [(430,435), (960,1000), (1100,1150), (3900,4000)]
    if cv.contourArea(contour)>300 and cv.contourArea(contour)<4200:
        if cv.contourArea(contour)>430 and cv.contourArea(contour)<435 or cv.contourArea(contour)>960 and cv.contourArea(contour)<1000 or cv.contourArea(contour)>1100 and cv.contourArea(contour)<1150 or cv.contourArea(contour)>3900 and cv.contourArea(contour)<4000:
            continue
            
        #compute bounding rectangle
        x, y, w, h = cv.boundingRect(contour)

        #draw bounding rectangle on contour
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

#showing image
cv.imshow('Result', img)
#press x on keyboard to close
if cv.waitKey(0) & 0xFF == ord('x'):
    cv.destroyAllWindows()

#save result image
cv.imwrite('result3.jpg', img)