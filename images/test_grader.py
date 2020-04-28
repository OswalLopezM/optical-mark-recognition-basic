# USAGE
# python test_grader.py --image images/test_01.png

# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import base64
import math


ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
with open("test_01.png", "rb") as img_file:
    my_string = base64.b64encode(img_file.read())
image= base64.b64decode(my_string)
filename = 'example.jpg'  # I assume you have a way of picking unique filenames
with open(filename, 'wb') as f:
    f.write(image)
args = vars(ap.parse_args())
image = cv2.imread(filename)
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
answersArray = []

# load the image, convert it to grayscale, blur it
# slightly, then find edges
print (args)
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)


# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

# ensure that at least one contour was found
if len(cnts) > 0:
	# sort the contours according to their size in
	# descending order
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	paperimage = []
	# loop over the sorted contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points,
		# then we can assume we have found the paper
		if len(approx) == 4:
			docCnt = approx
			paperimage = c
			break

# cv2.drawContours(image, [docCnt], -1, 255, -1)
# cv2.imshow("paper", image)

# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# apply Otsu's thresholding method to binarize the warped
# piece of paper
thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

# cv2.drawContours(image, [cnts[1]], -1, 255, -1)
# cv2.imshow("paper", image)

# cv2.waitKey(0)
# loop over the contours
for c in cnts:
	# compute the bounding box of the contour, then use the
	# bounding box to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	# print("for  w: " + str(w) + " y: " + str(y) + " w: " + str(w) + " h: " + str(h) + " ar: " + str(ar ))
	# in order to label the contour as a question, region
	# should be sufficiently wide, sufficiently tall, and
	# have an aspect ratio approximately equal to 1
	if w >= 15 and h >= 15 and ar >= 0.8 and ar <= 1.5:
		# print("entro")
		questionCnts.append(c)


# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionCnts = contours.sort_contours(questionCnts,
	method="top-to-bottom")[0]
correct = 0


ARRAYS_LENGHT = [4 , 5, 6 , 2 , 5, 6 , 6, 6, 4]
stop = 0
i = 0
start = 0
# each question has 5 possible answers, to loop over the
# question in batches of 5
while i < len(ARRAYS_LENGHT):
	stop = stop + ARRAYS_LENGHT[i] 
	actualcnts = questionCnts[start : stop]
	cnts = contours.sort_contours(questionCnts[start : stop])[0]
	bubbled = None
	start = stop 
	# loop over the sorted contours
	for (j, c) in enumerate(cnts):
		# construct a mask that reveals only the current
		# "bubble" for the question
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		
		# apply the mask to the thresholded image, then
		# count the number of non-zero pixels in the
		# bubble area
		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
		total = cv2.countNonZero(mask)

		# cv2.imshow("buble " + str(i) + str(j) , mask)

		# if the current total has a larger number of total
		# non-zero pixels, then we are examining the currently
		# bubbled-in answer
		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)

	# initialize the contour color and the index of the
	# *correct* answer
	answersArray.append(bubbled[1])
	
	i = i + 1

# grab the test taker
print(answersArray)
# score = (correct / 5.0) * 100
# print("[INFO] score: {:.2f}%".format(score))
# cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
# 	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
# cv2.imshow("Original", image)
# cv2.imshow("Exam", paper)
cv2.waitKey(0)