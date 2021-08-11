import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

class SemanticGraph:
	def __init__(self):
		self.verbose = False
	
	def compute_heatmap(self, semantic_graph):
		print(semantic_graph)

	def compute_semantic_graph(self, img):
		print(img)
		colors = np.array([[255, 69, 0], [119, 119, 119], [255, 190, 190], [244, 243, 131]])
		#low_blue = np.array([255, 69, 0]) #table
		#high_blue = np.array([255, 69, 0])
		#low_blue = np.array([119, 119, 119]) #wall
		#high_blue = np.array([119, 119, 119])
		#low_blue = np.array([255, 190, 190]) #ceiling
		#high_blue = np.array([255, 190, 190])
		low_blue = np.array([170, 0, 59]) #floor
		high_blue = np.array([170, 0, 59])
		mask = cv2.inRange(img, low_blue, high_blue)
		mask = mask.reshape((480, 640, 1))
		human_img = np.where(mask != 0, 255, mask)
		#human_img = np.where(mask != 255, 0, mask)
		print(human_img.shape)

		low_blue = np.array([54, 114, 113]) #floor
		high_blue = np.array([54, 114, 113])
		mask2 = cv2.inRange(img, low_blue, high_blue)
		mask2 = mask2.reshape((480, 640, 1))
		human_img2 = np.where(mask2 != 0, 255, mask2)
		human_img = human_img + human_img2
		'''
		contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
		print(len(contours))
		centres = []
		for i in range(len(contours)):
			moments = cv2.moments(contours[i])
			centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
			cv2.circle(img, centres[-1], 5, (255, 255, 255), -1)
			

# convert image to grayscale image
		for color in colors:
			mask = cv2.inRange(img, color, color)
			mask = mask.reshape((480, 640, 1))

			# convert the grayscale image to binary image
			ret,thresh = cv2.threshold(mask,127,255,0)

			# calculate moments of binary image
			M = cv2.moments(thresh)

			# calculate x,y coordinate of center
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			centres.append((cX, cY))

			# put text and highlight the center
			cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
		
		print(centres)
		matrix_dict = {}
		
		center_idx = 0
		for center1 in centres:
			p1 = center1
			lst = []
			center_idx = center_idx + 1
			for center2 in centres:
				p2 = center2
				cv2.line(img, p1, p2, (255, 255, 255), thickness=2)
				mdpt = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
				d = math.sqrt((p1[1] - p2[1])**2 + (p1[0] - p2[0])**2)
				lst.append(str(round(d, 1)))
				if d != 0:
					print('d')				
					print(d)
					print('m')
					print(mdpt)
					cv2.putText(img, str(round(d, 1)), mdpt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)
			matrix_dict[str(center_idx)] = lst
		print(matrix_dict)
		df = pd.DataFrame(matrix_dict, index =['1',
                                '2',
                                '3',
                                '4',
				'5',
				'6'])
		with pd.option_context('display.max_rows', None, 'display.max_columns', None):
			print(df)
		'''
		plt.imshow(human_img)
		plt.show()
		return human_img
