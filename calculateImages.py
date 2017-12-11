#!/usr/bin/python

# Import the modules
import cv2
import sys
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse as ap

#convert number array into a single variable
def toNumber(list):
    count = len(list)
    i = 0
    buff = 0

    while(count > 0):
        temp = list[i]
        temp = temp * 10**i

        buff = buff + temp
        i = i + 1
        count = count - 1

    return buff

#basic four function operation
def operation(num1, num2, op):
    if op == "+":
        return num1 + num2
    if op == "-":
        return num1 - num2
    if op == "*":
        return num1 * num2
    if op == "/":
        return num1 / num2

#File arguments
classiferPath = sys.argv[1]
image1 = sys.argv[2]
image2 = sys.argv[4]
op = sys.argv[3]

# Load the classifier
clf, pp = joblib.load(classiferPath)

# Read the input image 
im = cv2.imread(image1)
im2 = cv2.imread(image2)

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.GaussianBlur(im2_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
ret2, im2_th = cv2.threshold(im2_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
beep, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
beep2, ctrs2, hier2 = cv2.findContours(im2_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour on image1 and create array to hold image1 values.
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
mathNumIM1 = []

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
    nbr = clf.predict(roi_hog_fd)
    mathNumIM1.append(nbr)
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

# Get rectangles contains each contour on image2 and create array to hold image2 values.
rects2 = [cv2.boundingRect(ctr2) for ctr2 in ctrs2]
mathNumIM2 = []

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect2 in rects2:
    # Draw the rectangles
    cv2.rectangle(im2, (rect2[0], rect2[1]), (rect2[0] + rect2[2], rect2[1] + rect2[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng2 = int(rect2[3] * 1.6)
    pt12 = int(rect2[1] + rect2[3] // 2 - leng2 // 2)
    pt22 = int(rect2[0] + rect2[2] // 2 - leng2 // 2)
    roi2 = im2_th[pt12:pt12+leng2, pt22:pt22+leng2]
    # Resize the image
    roi2 = cv2.resize(roi2, (40, 40), interpolation=cv2.INTER_AREA)
    roi2 = cv2.dilate(roi2, (3, 3))
    # Calculate the HOG features
    roi_hog_fd2 = hog(roi2, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    roi_hog_fd2 = pp.transform(np.array([roi_hog_fd2], 'float64'))
    nbr2 = clf.predict(roi_hog_fd2)
    mathNumIM2.append(nbr2)
    cv2.putText(im2, str(int(nbr2[0])), (rect2[0], rect2[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

num1 = toNumber(mathNumIM1)
num2 = toNumber(mathNumIM2)
total = operation(num1, num2, op)
print("You entered:", num1, op, num2)
print("Total =", total)



#Resulting Image with Rectangular ROIs
cv2.namedWindow("Image 1", cv2.WINDOW_NORMAL)
cv2.imshow("Image 1", im)
cv2.namedWindow("Image 2", cv2.WINDOW_NORMAL)
cv2.imshow("Image 2", im2)
cv2.waitKey()







