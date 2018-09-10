"""
https://medium.com/@adamaulia/object-tracking-using-opencv2-python-windows-616fb23da720

https://docs.opencv2.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html
"""

import numpy as np
import cv2
import sys

CAM_NUM = 0

class Video:
	def __init__(self):
		print("setting up the video capture")
		self.cap = cv2.VideoCapture(CAM_NUM)
		self.frame = None
		self.previousFrame = [] # Usefull for project recognize perpose
		self.windowFrame = {} #the frame for everywindows
		self.paused = False
		self.frameCount = 0 # to check if there is new processed frame exist by countin up 100
		
		
	def createNewWindow(self, name, **kwargs):
		"""
		name   : name of the window
		kwargs : "frames" - decides which frame to put on the window
		         "xPos"   - The x positon the screen to create the window
				 "yPos"   - The y positon the screen to create the window
		"""  
		# set up the variables
		frameForWindow = kwargs.get("frame", self.frame)
		
		if frameForWindow is None:
			self.getVideo()
			frameForWindow = self.frame # in case at the beginning  no frame has read
			
		xPos = kwargs.get("xPos", 20)
		yPos = kwargs.get("yPos", 20)
		
		# create new windows
		cv2.namedWindow(name)
		cv2.moveWindow(name, xPos, yPos)
		self.windowFrame[name] = frameForWindow[1] # add frame for the window to show
		
	def isCameraConnected(self):
		return self.cap.isOpened()
	
	def getVideo(self):
		if not self.paused:
			ret, nFrame = self.cap.read()
			
			try: # check if it is really a frame 
				self.frame = nFrame.copy()
			except:
				print("ERROR: frame  is not captured")
				
		if not ret: # check if there was no frame captured
			print ("ERROR: while capturing frame")
			
		return ret, nFrame	

	def display(self, window, **kwargs):
		cv2.imshow(window, self.windowFrame[window])
		
def test():
	vid = Video()
	vid.createNewWindow("original", xPos = 100, yPos = 100)
	
	while True:
		ret, nFrame = vid.getVideo()
		vid.display("original")
		
		if cv2.waitKey(30) & 0xFF == ord("q"):
			break
	# vid.release()
	cv2.destroyAllWindows()

	
def main():
	#read video file
	cap = cv2.VideoCapture(0)

	while (cap.isOpened):
		#if ret is true than no error with cap.isOpened
		ret, frame = cap.read()
		mask = np.zeros(frame.shape[:2],np.uint8)

		# ask[newmask == 0] = 0
		# mask[newmask == 255] = 1
		rect = (50,50,450,290)
		bgdModel = np.zeros((1,65),np.float64)
		fgdModel = np.zeros((1,65),np.float64)

		if ret:
			# apply background substraction
			cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
			mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
			img = frame*mask2[:,:,np.newaxis]
			# plt.imshow(img),plt.colorbar(),plt.show()

			cv2.imshow('forground', img)
			cv2.imshow('Original',frame)

			if cv2.waitKey(30) & 0xFF == ord("q"):
				break

		else:
			print("ret is {}" .format(ret))

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
	# test()
