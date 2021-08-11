import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.segmentation.segmentation import Segmentation
from src.GAN.GAN import GenerativeAdverserialNetwork

class DynamicObjectRemoval:

	def __init__(self):
		self.verbose = True
		self.segmenter = Segmentation()
		self.GAN = GenerativeAdverserialNetwork()

	def remove_object(self, image, depth, imagepath):

		segmentation_model = self.segmenter.get_model()
		print(image.shape)
		print('-'*20)
		print(depth.shape)
	
		img, mask = self.segmenter.predict(image, depth)
		img = img.reshape((1, 3, 480, 640))
		img = np.transpose(img.detach().cpu().numpy(), (0, 2,3,1))
		img = img.reshape((480, 640, 3))
		#img = img[0].reshape((480, 640, 1))
		print(mask.shape)
		mask =mask.reshape((480, 640, 1))
		'''for i in range(480):
			for j in range(640):
				if mask[i][j] != 0:
					img[i][j] = 0
		'''
		plt.imshow(img)
		plt.show()
		#print(mask)
		output = self.GAN.impaint(img, mask, imagepath)
		plt.imshow(output)
		plt.show()
##get mask
##inpaint
