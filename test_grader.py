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
import random

#Import for neural network

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array



ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image",
	help="path to the input image")
with open("images/jeni/hojajeni112bn.jpg", "rb") as img_file:
    my_string = base64.b64encode(img_file.read())
image= base64.b64decode(my_string)
filename = 'output.jpg'  # I assume you have a way of picking unique filenames
with open(filename, 'wb') as f:
    f.write(image)
args = vars(ap.parse_args())
image = cv2.imread(filename)



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
	help="path to the input image")
args = vars(ap.parse_args())

# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
answersArray = []


# # #DENSE Model
# new_model = tf.keras.models.load_model('emnist_trained_dense.h5')

#CNN Model
new_model = tf.keras.models.load_model('emnist_trained.h5')


letters ={0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,
10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',
20:'K',21:'l',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',
30:'u',31:'V',32:'W',33:'X',34:'Y',35:'Z',36:'a',37:'b',38:'d',39:'e',
40:'f',41:'g',42:'h',43:'n',44:'q',45:'r',46:'t',47:'அ',48:'ஆ'}

# load the image, convert it to grayscale, blur it
# slightly, then find edges
print (args)
# image = cv2.imread(args["image"])


# cv2.imshow("image", image)
# cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)


# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

questions = image.copy()
for ct in cnts:
	pass
	cv2.drawContours(questions, [ct], -1, (random.randint(1,254),random.randint(1,254),random.randint(1,254)), -1)

# cv2.drawContours(questions, [cnts[1]], -1, 255, -1)
imS = cv2.resize(questions, (540,960))   
cv2.imshow("questions", questions)

cv2.waitKey(0)


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

		
		print((peri))
		print(len(approx))

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

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour, then use the
	# bounding box to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	print("for w: " + str(w) + " h: " + str(h) + " ar: " + str(ar ))
	# in order to label the contour as a question, region
	# should be sufficiently wide, sufficiently tall, and
	# have an aspect ratio approximately equal to 1
	# imS = cv2.resize(questions, (540,960))   
	# cv2.imshow("questions", paper[ y-5 : y+h+5 , x : x-5 + w+5])

	# cv2.waitKey(0)

	# box
	if w >= 100 and h >= 50 and ar > 4:
		print(h,w,ar)
		questionCnts.append(c)
		
	# bubble
	if w >= 15 and h >= 15 and ar >= 0.8 and ar <= 1.5:
		questionCnts.append(c)


# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionCnts = contours.sort_contours(questionCnts,
	method="top-to-bottom")[0]
correct = 0

questions = paper.copy()
for ct in questionCnts:
	pass
	cv2.drawContours(questions, [ct], -1, (random.randint(1,254),random.randint(1,254),random.randint(1,254)), -1)

# cv2.drawContours(questions, [cnts[1]], -1, 255, -1)
imS = cv2.resize(questions, (540,960))   
cv2.imshow("questions", imS)

cv2.waitKey(0)

ARRAYS_LENGHT = [3 , 3, 1 , 5 , 3 , 3, 1 ,  5, 3 , 3, 1 ]
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

	if(ARRAYS_LENGHT[i] != 1):

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
		# correct answer
		answersArray.append(bubbled[1])

	else:
		#development question
		# answersArray.append(False)

		x,y,w,h = cv2.boundingRect(cnts[0])

		# box = thresh[y+4 : y+27 , x + 3: x + 27] #tamano de la caja
		box = thresh[y : y+h , x : x + w] #tamano de la caja
		

		print("x: " + str(x))
		print("x: " + str(10 * (x + 27)))

		firstY = y
		firstX = x
		stopX = (x + w) - 27
		stopY = y + h - 27

		ib = 0
		jb = 0


		# print(cnts[0])
		print(x,y,stopX,stopY)
		preprocessed_digits = []
		textAnswer = ""
		while y <  stopY:

			x = firstX
			while x < stopX:

				littleBox = thresh[y+4 : y+27 , x + 3: x + 26]

				

				# print(jb, x, firstX, stopX)
				# cv2.imshow("littleBox", littleBox)
				# cv2.waitKey(0)


				# box = cv2.rectangle(box, (y+4 , x + 3 ), ( y + 27 , x + 27), ( 255, 100, 255)) 

				
				# Resizing that digit to (18, 18)
				resized_digit = cv2.resize(littleBox, (18,18))
				
				# Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
				padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
				
				# Adding the preprocessed digit to the list of preprocessed digits
				preprocessed_digits.append(padded_digit)

				# # En caso de usar el model de DENSE
				# prediction = new_model.predict(digit.flatten().reshape(-1, 28*28))  

				# En caso de usar el model de CONVOLUTIONAL
				prediction = new_model.predict(padded_digit.reshape(1, 28, 28, 1))

				textAnswer = textAnswer + str(letters[int(np.argmax(prediction))])



				jb = jb + 1
				x = x + 28

			print(ib, y, firstY, stopY)
			jb = 0
			ib = ib + 1
			y = y + 28
		
		print(textAnswer)
		answersArray.append(textAnswer)
		

		cv2.imshow("rectangle", box)

		cv2.waitKey(0)

	
	i = i + 1

# grab the test taker
print(answersArray)

cv2.waitKey(0)