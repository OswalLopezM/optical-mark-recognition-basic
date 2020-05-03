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
with open("images/bienimpresas/marcadors.png", "rb") as img_file:
	my_string = base64.b64encode(img_file.read())
image= base64.b64decode(my_string)
filename = 'output.jpg'  # I assume you have a way of picking unique filenames
with open(filename, 'wb') as f:
	f.write(image)
args = vars(ap.parse_args())
image = cv2.imread(filename)

height, width, channels = image.shape
print("shape: ", image.shape)
percent = (1200 * 100) /height 
print(percent)

width = int(width * percent / 100)
height = int(height * percent / 100)

originalImage = image.copy()
# image = cv2.resize(image, (width,height))

print("shape: ", image.shape)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
	help="path to the input image")
args = vars(ap.parse_args())


# load the image, convert it to grayscale, blur it
# slightly, then find edges
print (args)

# #DENSE Model
# new_model = tf.keras.models.load_model('emnist_trained_dense.h5')

# #CNN Model
new_model = tf.keras.models.load_model('emnist_trained.h5')


letters ={0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,
10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',
20:'K',21:'l',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',
30:'u',31:'V',32:'W',33:'X',34:'Y',35:'Z',36:'a',37:'b',38:'d',39:'e',
40:'f',41:'g',42:'h',43:'n',44:'q',45:'r',46:'t',47:'அ',48:'ஆ'}



# cv2.imshow("image", image)
# cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (21,21), 0)
edged = cv2.Canny(blurred, threshold1=5, threshold2=20)


# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
	
# ret, edged = cv2.threshold(blurred.copy(), 75, 255, cv2.THRESH_BINARY_INV)
# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
docCnt = None


imS = cv2.resize(image, (750,1000))   
imS2 = cv2.resize(edged, (750,1000))  
# cv2.imshow("image", imS) 
# cv2.imshow("edged", imS2)

print(len(cnts))


questions = blurred.copy()
idx = 0
for ct in cnts:
	pass
	# cv2.drawContours(questions, [ct], -1, (random.randint(1,254),random.randint(1,254),random.randint(1,254)), -1)


# imS = cv2.resize(questions, (750,1000))   
# cv2.imshow("all countours", imS)

# cv2.waitKey(0)


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


paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))



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
	if w >= 1000 and h >= 200 and ar > 4:
		print(h,w,ar)
		questionCnts.append(c)
		
	# bubble
	# if w >= 40 and h >= 40 and ar >= 0.8 and ar <= 1.5:
	# 	questionCnts.append(c)


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
imS = cv2.resize(questions, (750,1000))   
cv2.imshow("questions", imS)

cv2.waitKey(0)



ARRAYS_LENGHT = [1,1,1]
stop = 0
i = 0
start = 0
answersArray= []
id = 0

while i < len(ARRAYS_LENGHT):
	idx = idx +1
	stop = stop + ARRAYS_LENGHT[i] 
	actualcnts = questionCnts[start : stop]
	cnts = contours.sort_contours(questionCnts[start : stop])[0]
	bubbled = None
	start = stop 
	if(idx > 0):

		if(ARRAYS_LENGHT[i] != 1):

			# loop over the sorted contours
			for (j, c) in enumerate(cnts):
				# construct a mask that reveals only the current
				# "bubble" for the question
				mask = np.zeros(thresh.shape, dtype="uint8")
			# 	cv2.drawContours(mask, [c], -1, 255, -1)
				
			# 	# apply the mask to the thresholded image, then
			# 	# count the number of non-zero pixels in the
			# 	# bubble area
			# 	mask = cv2.bitwise_and(thresh, thresh, mask=mask)
			# 	total = cv2.countNonZero(mask)

			# 	# cv2.imshow("buble " + str(i) + str(j) , mask)

			# 	# if the current total has a larger number of total
			# 	# non-zero pixels, then we are examining the currently
			# 	# bubbled-in answer
			# 	if bubbled is None or total > bubbled[0]:
			# 		bubbled = (total, j)

			# # initialize the contour color and the index of the
			# # correct answer
			# answersArray.append(bubbled[1])

		else:
			#development question
			# answersArray.append(False)

			
			preprocessed_digits = []
			textAnswer = ""
			textLine = ""

			x,y,w,h = cv2.boundingRect(cnts[0])

			# cuadrito = thresh[y+4 : y+27 , x + 4: x + 27] #tamano del cuadrito

			box = thresh[y+3 : y+h -3 , x+3 : x + w - 3] #tamano de la caja de la respuesta de desarrollo
			boxPaper = paper[y+3 : y+h -3 , x+3 : x + w - 3] #tamano de la caja de la respuesta de desarrollo

			

			cv2.imshow("rectangle", box)
			cv2.imshow("rectangle paper", boxPaper)

			cv2.waitKey(0)

			wBox = math.ceil(w/19) - 1
			
			hBox = math.ceil(h/4) - 1

			cuadritoNuevo = thresh[y+4 : y+hBox , x + 4: x + wBox]

			#####################################################CODIGO MEDIUM PARA DETECTAR CUADRADOS#########################################################

			# Defining a kernel length
			kernel_length = np.array(box).shape[1]//80
			
			# A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
			verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
			# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
			hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
			# A kernel of (3 X 3) ones.
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


			# Morphological operation to detect vertical lines from an image
			img_temp1 = cv2.erode(box, verticle_kernel, iterations=3)
			verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
			cv2.imwrite("verticle_lines.jpg",verticle_lines_img)
			# Morphological operation to detect horizontal lines from an image
			img_temp2 = cv2.erode(box, hori_kernel, iterations=3)
			horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
			cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)

			
			# cv2.imshow("horizontal_lines_img", horizontal_lines_img)
			# cv2.imshow("verticle_lines_img", verticle_lines_img)
			# cv2.waitKey(0)

			# Weighting parameters, this will decide the quantity of an image to be added to make a new image.
			alpha = 0.5
			beta = 1.0 - alpha
			# This function helps to add two image with specific weight parameter to get a third image as summation of two image.
			img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
			img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
			(threshFinal, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			cv2.imwrite("img_final_bin.jpg",img_final_bin)

			
			# cv2.imshow("img_final_bin", img_final_bin)
			# cv2.waitKey(0)

			# # Find contours for image, which will detect all the boxes
			ctnsBox, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			# Sort all the contours by top to bottom.
			(ctnsBox, boundingBoxes) = contours.sort_contours(ctnsBox, method="top-to-bottom")
			



			# for ct in ctnsBox:
			# 	pass
			# 	cv2.drawContours(box, [ct], -1, (random.randint(1,254),random.randint(1,254),random.randint(1,254)), -1)

				
			# cv2.imshow("final boxs", box)
			# cv2.waitKey(0)

			idx = 0

			print(len(ctnsBox))
			startBox = 0
			stopBox = startBox + 19

			textAnswer1 = ""
			textAnswer2 = ""




			while idx < 4:
			
				print("idx, startBox , stopBox" )
				print(idx, startBox , stopBox )
				partCtnsBox = contours.sort_contours(ctnsBox[startBox : stopBox ])[0]

				startBox = stopBox
				stopBox = startBox+19
				print(idx)
				print(len(partCtnsBox))
				for c in partCtnsBox:

					# xBoxito, yBoxito, wBoxito, hBoxito = cv2.boundingRect(c)
					# new_img = boxPaper[yBoxito:yBoxito+wBoxito, xBoxito:xBoxito+wBoxito]
					
					# cv2.imshow("final boxs", new_img)
					# cv2.waitKey(0)


					xBoxito, yBoxito, wBoxito, hBoxito = cv2.boundingRect(c)
					# print(xBoxito, yBoxito, wBoxito, hBoxito)
					if (wBoxito > 17 and hBoxito > 17):
						
						# idx += 1
						new_img = boxPaper[yBoxito:yBoxito+wBoxito, xBoxito:xBoxito+wBoxito]
						new_img2 = box[yBoxito:yBoxito+wBoxito, xBoxito:xBoxito+wBoxito]
						

						####################INVENTO DE CORTAR LA LETRA######################
												
						grey = cv2.cvtColor(new_img.copy(), cv2.COLOR_BGR2GRAY)	
						ret, threshLetter = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
						ctnsLetter, _ = cv2.findContours(threshLetter.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
						print("STR(LEN(ctnsLetter))" , len(ctnsLetter))

						if(len(ctnsLetter) == 0):
							# no letter / space
							print("space")
							
							textAnswer1 = textAnswer1 + " "


						elif(len(ctnsLetter) > 1):
							# letter i or j
							print("i or j")

							rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
							dilation = cv2.dilate(threshLetter, rect_kernel, iterations = 1)
							
							im2 = new_img.copy()
							contoursSpecial, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
							xi, yi, wi, hi = cv2.boundingRect(contoursSpecial[0])
							digit = threshLetter[yi:yi+hi, xi:xi+wi]

							
							# plt.imshow(im2, cmap="gray")
							# plt.show()
								
							# # Resizing that digit to (18, 18)
							resized_digit = cv2.resize(digit, (18,18))
							
							# # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
							padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)

							
							prediction = new_model.predict(padded_digit.reshape(1, 28, 28, 1))

							
							textAnswer1 = textAnswer1 + str(letters[int(np.argmax(prediction))])

							textLine = textLine + str(letters[int(np.argmax(prediction))])
							#PRUEBA HACIENDO LA PREDICITION CON ESTA IMAGEN
							print(str(letters[int(np.argmax(prediction))]), np.argmax(prediction))

						else:	
							#Normal letter
							print("normal")

							# (ctnsLetter, boundingBoxes) = contours.sort_contours(ctnsLetter, method="top-to-bottom")
							print("STR(LEN(ctnsLetter))" , len(ctnsLetter))
							cLetter = ctnsLetter[0]
							xL,yL,wL,hL = cv2.boundingRect(cLetter)
								
							# Cropping out the digit from the image corresponding to the current contours in the for loop
							digit = threshLetter[yL:yL+hL, xL:xL+wL]
							
							# # Resizing that digit to (18, 18)
							resized_digit = cv2.resize(digit, (18,18))
							
							# # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
							padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)

							
							prediction = new_model.predict(padded_digit.reshape(1, 28, 28, 1))


							textAnswer1 = textAnswer1 + str(letters[int(np.argmax(prediction))])


							print(str(letters[int(np.argmax(prediction))]), np.argmax(prediction))
								
								# Adding the preproces



						
						####################INVENTO DE CORTAR LA LETRA######################
						
						if(len(ctnsLetter) > 0):
							# Resizing that digit to (18, 18)
							resized_digit2 = cv2.resize(new_img2, (28,28))
							
							
							
							# Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
							padded_digit = np.pad(resized_digit2, ((5,5),(5,5)), "constant", constant_values=0)
							
							# Adding the preprocessed digit to the list of preprocessed digits
							preprocessed_digits.append(resized_digit2)

							# # En caso de usar el model de DENSE
							# prediction = new_model.predict(padded_digit.flatten().reshape(-1, 28*28))  

							# # En caso de usar el model de CONVOLUTIONAL
							prediction2 = new_model.predict(resized_digit2.reshape(1, 28, 28, 1))

							textLine = textLine + str(letters[int(np.argmax(prediction))])

							
							textAnswer2 = textAnswer2 + str(letters[int(np.argmax(prediction2))])
							print(str(letters[int(np.argmax(prediction2))]), np.argmax(prediction2))

							
							# plt.imshow(padded_digit, cmap="gray")
							# plt.show()
							
							# plt.imshow(resized_digit.reshape(28, 28), cmap="gray")
							# plt.show()
							
							
								
							
							print("\n\n\n----------------Contoured Image--------------------")

							# if( idx == 19):
							# 	idx = 0
							# 	# print(textLine[::-1])
							# 	textAnswer = textAnswer + textLine[::-1]
							# 	textLine = ""
							
							# cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)

						else: 
							textAnswer2 = textAnswer2 + " "
				idx = idx + 1





			# for c in ctnsBox:
				# Returns the location and width,height for every contour
				

			###############################CODIGO MEDIUM PARA DETECTAR CUADRADOS#########################################################
					

			print("textAnswer: " + textAnswer)
			print("textAnswer1: " + textAnswer1)
			print("textAnswer2: " + textAnswer2)
			answersArray.append(textAnswer)
		

	
	i = i + 1