import cv2
import os
import importlib
import numpy as np
from glob import glob 

import torch
from torchvision.transforms import ToTensor

from src.GAN.model.InpaintingGAN import InpaintGenerator

class GenerativeAdverserialNetwork:

	def postprocess(self, image):
		image = torch.clamp(image, -1., 1.)
		image = (image + 1) / 2.0 * 255.0
		image = image.permute(1, 2, 0)
		image = image.cpu().numpy().astype(np.uint8)
		return image

	def impaint(self, image, mask, fn):

	    # Model and version
		model = InpaintGenerator()
		model.load_state_dict(torch.load('/home/chris/GAN_SLAM/src/GAN/trained_models/places2/G0000000.pt'))
		model.eval()

		orig_img = cv2.resize(cv2.imread(fn, cv2.IMREAD_COLOR), (512, 512))
		mask = cv2.resize(mask.reshape((480, 640)), (512, 512))
		mask = mask.reshape((512, 512, 1))
		img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0)
		h, w, c = orig_img.shape
		#mask = np.zeros([h, w, 1], np.uint8)
		image_copy = orig_img.copy()
		mask = mask.astype(np.uint8)

		print('[**] inpainting ... ')
		with torch.no_grad():
			mask_tensor = (ToTensor()(mask)).unsqueeze(0)
			masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
			pred_tensor = model(masked_tensor, mask_tensor)
			comp_tensor = (pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor))

			pred_np = self.postprocess(pred_tensor[0])
			masked_np = self.postprocess(masked_tensor[0])
			comp_np = self.postprocess(comp_tensor[0])

			return comp_np



