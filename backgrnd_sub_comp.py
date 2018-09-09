import numpy as np
import cv2
import sys


#read video file
cap = cv2.VideoCapture(0)

# BackgroundSubtractorMOG2
fgbg = cv2.createBackgroundSubtractorMOG2()

# BackgroundSubtractorGMG
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

fgbg_ = cv2.bgsegm.createBackgroundSubtractorGMG()

while (cap.isOpened):

	#if ret is true than no error with cap.isOpened
	ret, frame = cap.read()

	if ret:
		#apply background substraction
		fgmask = fgbg.apply(frame)
		(im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		# BackgroundSubtractorGMG
		fgmask_ = fgbg_.apply(frame)
		fgmask_ = cv2.morphologyEx(fgmask_, cv2.MORPH_OPEN, kernel)

		#looping for contours
		for c in contours:
			if cv2.contourArea(c) < 500:
				continue

			#get bounding box from countour
			(x, y, w, h) = cv2.boundingRect(c)

			#draw bounding box
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		cv2.imshow('BackgroundSubtractorGMG', fgmask_)
		cv2.imshow('BackgroundSubtractorMOG2',fgmask)
		cv2.imshow('rgb',frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break


cap.release()
cv2.destroyAllWindows()
