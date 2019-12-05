import cv2 as cv

#read image
img = cv.imread("advertising1.jpg")

#convert BGR image to grayscale
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Image Thresholding
_, thresh = cv.threshold(imgGray, 100, 255, cv.THRESH_BINARY_INV)

#Morphological Transformations
closing = cv.morphologyEx(thresh, cv.MORPH_OPEN, (11,11), iterations=1)

#Find contours of image
contours, heirachy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

for contour in contours:
    #find and draw rectangle on contour area in range (2300, 3000) but not in range (20, 400)
    #contour area in range (2300, 3000) is alphabet contour area for letter 'M, D'
    if cv.contourArea(contour)>2300 and cv.contourArea(contour)<3000:
        #contour area in range (20, 400) is hole in letter 'A, P, R'
        if cv.contourArea(contour)>20 and cv.contourArea(contour)<400:
            continue

        #compute bounding rectangle
        x, y, w, h = cv.boundingRect(contour)

        #draw bounding rectangle on contour
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
    #find and draw rectangle on contour area in range (300, 2000)
    #contour area in range (2300, 3000) is alphabet contour area for all letter in advertising1 picture
    if cv.contourArea(contour)>300 and cv.contourArea(contour)<2000:

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
cv.imwrite('result.jpg', img)
