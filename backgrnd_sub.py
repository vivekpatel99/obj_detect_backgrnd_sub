"""
https://medium.com/@adamaulia/object-tracking-using-opencv-python-windows-616fb23da720

https://docs.opencv.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html
"""

import numpy as np
import cv2
import sys
import imutils

def main():
	#read video file
	cap = cv2.VideoCapture(0)

	# BackgroundSubtractorMOG2
	fgbg= cv2.createBackgroundSubtractorMOG2()
		
	while (cap.isOpened):
		#if ret is true than no error with cap.isOpened
		ret, frame = cap.read()
		if frame  is None:
			print("ERROR: No frame available !!")
		
		# resize the image inorder to have less processing time
		frame = imutils.resize(frame, width= 500)
		# color has no bearing on algorithm
		gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# to smooth the image and remove noise(if not then could throw algorithm off)
		# smothing avarage pixel intensities across an 21x21 region
		gray= cv2.GaussianBlur(gray, (21,21), 0)

		if ret:
			# apply background substraction
			fgmask= fgbg.apply(gray)
			(im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

			#looping for contours
			for c in contours:
				if cv2.contourArea(c) < 500:
					continue

				#get bounding box from countour
				(x, y, w, h) = cv2.boundingRect(c)

				#draw bounding box
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			cv2.imshow('BackgroundSubtractorGMG', fgmask)
			cv2.imshow('Original',frame)

			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

		else:
			print("ret is {}" .format(ret))

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
