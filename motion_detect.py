"""
https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

"""

import numpy as np
import cv2
import sys
from imutils.video import VideoStream
import imutils
import time
# minimum size (in pixel) for a region of image to be considered actual "motion" 
MIN_AREA = 500
CAM_NUM = 0

def main():
	"""
	"""
	cap = VideoStream(src=CAM_NUM).start()
	time.sleep(2.0)
	
	# initialize the firstFrame in video stream
	firstFrame = None
	
	while True:	
		
		#if ret is true than no error with cap.isOpened
		frame = cap.read()
		
		if frame  is None:
			print("ERROR: No frame available !!")
			break
			
		# resize the image inorder to have less processing time
		frame = imutils.resize(frame, width=500)
		# color has no bearing on motion detection algorithm
		gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# to smooth the image and remove noise(if not then could throw algorithm off)
		# smothing avarage pixel intensities across an 21x21 region
		gray= cv2.GaussianBlur(gray, (21, 21), 0)

		# if the first stream is not initialized, store it for reference	
		if firstFrame is None:
			firstFrame = gray
			continue
		
		# compute the absolute difference between the current frame and firstFrame
		frameDelta = cv2.absdiff(firstFrame, gray)
		thresh  = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

		# dilate the threshold image to fill in holes, then find contours on the thresholded image
		# apply background substraction
		thresh = cv2.dilate(thresh, None,  iterations=2)

		# im2, contours, hierarchy  = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = contours[0] if imutils.is_cv2() else contours[1]

		#looping for contours
		for c in contours:
			if cv2.contourArea(c) < MIN_AREA:
				continue

			#get bounding box from countour
			(x, y, w, h) = cv2.boundingRect(c)

			#draw bounding box
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		cv2.imshow('Original',frame)
		cv2.imshow('threshold', thresh)
		cv2.imshow('FrameDelta', frameDelta)

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
			
	cap.stop()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
