import cv2 as cv
import imutils

#read image
img = cv.imread("advertising2.jpg")

#resize image
img = imutils.resize(img, 480, 600)

#convert BGR image to grayscale
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Image Thresholding
_, thresh = cv.threshold(imgGray, 150, 255, cv.THRESH_BINARY_INV)

#Morphological Transformations
closing = cv.morphologyEx(thresh, cv.MORPH_OPEN, (3,3), iterations=1)

#Find contours of image
contours, heirachy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

for contour in contours:
    #find and draw rectangle on contour area in range (107, 2000)
    if cv.contourArea(contour)>107 and cv.contourArea(contour)<2000:

        #ignore contour area in range (20, 100), (190, 200), (210, 220), (221, 225), (230, 235,) (760, 800), (1200, 1250), (3000,4200)
        if cv.contourArea(contour)>20 and cv.contourArea(contour)<100 or cv.contourArea(contour)>110 and cv.contourArea(contour)<120 or cv.contourArea(contour)>190 and cv.contourArea(contour)<200 or cv.contourArea(contour)>210 and cv.contourArea(contour)<220 or cv.contourArea(contour)>221 and cv.contourArea(contour)<225 or cv.contourArea(contour)>230 and cv.contourArea(contour)<235 or cv.contourArea(contour)>760 and cv.contourArea(contour)<800 or cv.contourArea(contour)>1200 and cv.contourArea(contour)<1250 or cv.contourArea(contour)>3000 and cv.contourArea(contour)<4200:
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
cv.imwrite('result_ad2.jpg', img)