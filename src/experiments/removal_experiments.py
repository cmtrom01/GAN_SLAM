import os
import numpy as np
import cv2

from src.dynamic_object_removal.dynamic_object_removal import DynamicObjectRemoval

class RemovalExperiments:

	def __init__(self):
		self.verbose = True
		self.dynamic_obj_removal = DynamicObjectRemoval()
		self.PATH_RGB = '/home/chris/GAN_SLAM/data/rgbd_dataset_freiburg3_sitting_xyz/rgb/'
		self.PATH_DEPTH = '/home/chris/GAN_SLAM/data/rgbd_dataset_freiburg3_sitting_xyz/depth/'

	def get_image_paths(self, sort = True):

		file_names_RGB = np.array(os.listdir(self.PATH_RGB))
		file_names_DEPTH = np.array(os.listdir(self.PATH_DEPTH))
		print(file_names_RGB)
	
		if sort:
			return	np.sort(file_names_RGB), np.sort(file_names_DEPTH)
		else:
			return	file_names_RGB, file_names_DEPTH

	
	def load_img(self, fp):
		img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
		if img.ndim == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img


	def run_removal_experiments(self):
		rgb_paths, depth_paths = self.get_image_paths()
		print(rgb_paths[0])
		image, depth = self.load_img(self.PATH_RGB+rgb_paths[10]), self.load_img(self.PATH_DEPTH+depth_paths[10])
		self.dynamic_obj_removal.remove_object(image, depth, self.PATH_RGB+rgb_paths[10])

