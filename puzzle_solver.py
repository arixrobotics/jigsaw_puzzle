# import the necessary packages
import imutils
import cv2
import numpy as np
from math import atan2

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("pics/IMG-3865.JPG")
image = imutils.resize(image, width=500)
# ratio = image.shape[0] / float(resized.shape[0])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 9), 0)
# cv2.imshow("gray", gray)
# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.Canny(gray, 15, 200) 
# thresh = imutils.auto_canny(gray)
# thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
# thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
# cv2.imshow("processed", thresh)
# find contours in thresholded image, then grab the largest
# one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# c = max(cnts, key=cv2.contourArea)

# loop over the contours
for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]))
    cY = int((M["m01"] / M["m00"]))
    print cX, cY

    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    # cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
    #   0.5, (255, 255, 255), 2)
    # # determine the most extreme points along the contour
    # extLeft = tuple(c[c[:, :, 0].argmin()][0])
    # extRight = tuple(c[c[:, :, 0].argmax()][0])
    # extTop = tuple(c[c[:, :, 1].argmin()][0])
    # extBot = tuple(c[c[:, :, 1].argmax()][0])

    # # draw the outline of the object, then draw each of the
    # # extreme points, where the left-most is red, right-most
    # # is green, top-most is blue, and bottom-most is teal
    # cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
    # cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
    # cv2.circle(image, extRight, 8, (0, 255, 0), -1)
    # cv2.circle(image, extTop, 8, (255, 0, 0), -1)
    # cv2.circle(image, extBot, 8, (255, 255, 0), -1) 

    # # Bounding Rectangle
    # x,y,w,h = cv2.boundingRect(c)
    # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    # # Rotated rect
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(image,[box],0,(0,0,255),2)

    # # Corner detector
    # dst = cv2.cornerHarris(thresh,2,3,0.04)
    # #result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst,None)
    # # Threshold for an optimal value, it may vary depending on the image.
    # image[dst>0.01*dst.max()]=[0,0,255]

    # Approc PoliDP
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True) #0.005
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
    print "Approx:"
    print approx
    for pt in approx:
        for p in pt:
            # print p
            cv2.circle(image, (p[0],p[1]), 8, (255, 255, 255), 1) 

    print len(approx)
    angle = [0.0]*len(approx)
    for i in range(len(approx)):
        # print i
        
        if i == len(approx)-1:
            pt1 = (approx[i][0])
            pt2 = (approx[0][0])
            # print pt1, pt2
        else:
            pt1 = (approx[i][0])
            pt2 = (approx[i+1][0])
            # print pt1, pt2
        # angle = atan2(approx[i+1][1] - approx[i][1], approx[i+1][0] - approx[i][0]) * 180.0 / np.pi;
        angle[i] = atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180.0 / np.pi 
        print i, pt1, pt2, angle[i]
        # cv2.putText(image, str(i), (pt1[0],pt1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    print "len angle:"
    print len(angle)
    for i in range(len(angle)):
        if i == 0:#len(angle)-1:
            angle_diff = abs(angle[i] - angle[-1])
            pt1 = (approx[i][0])
            
        else:
            angle_diff = abs(angle[i] - angle[i-1])
            pt1 = (approx[i][0])
            # pt2 = (approx[i+1][0])
        print i, angle_diff
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        if angle_diff > 75.0 :
            cv2.circle(image, (pt1[0],pt1[1]), 8, (255, 255, 255), -1) 
        # cv2.putText(image, str(angle_diff), (pt1[0],pt1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # show the output image
    cv2.imshow("Image", image)
    cv2.imshow("Images", np.hstack([gray, thresh]))
    cv2.waitKey(0)
