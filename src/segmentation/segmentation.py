import os
import argparse
from glob import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.colors

import torchvision
import torchvision.transforms as transforms

from src.segmentation.build_model import build_model
from src.semantic_graph.semantic_graph import SemanticGraph

class Segmentation:

	def __init__(self):
		self.verbose = False
		N_CLASSES = 37
		self.semantic_graph = SemanticGraph()

		self.CLASS_NAMES_ENGLISH = ['void', 'wall', 'floor', 'cabinet', 'bed', 'chair',
				           'sofa', 'table', 'door', 'window', 'bookshelf',
				           'picture', 'counter', 'blinds', 'desk', 'shelves',
				           'curtain', 'dresser', 'pillow', 'mirror',
				           'floor mat', 'clothes', 'ceiling', 'books',
				           'fridge', 'tv', 'paper', 'towel', 'shower curtain',
				           'box', 'whiteboard', 'person', 'night stand',
				           'toilet', 'sink', 'lamp', 'bathtub', 'bag']

		self.CLASS_COLORS = [(0, 0, 0), (119, 119, 119), (244, 243, 131),
				    (137, 28, 157), (150, 255, 255), (54, 114, 113),
				    (0, 0, 176), (255, 69, 0), (87, 112, 255), (0, 163, 33),
				    (255, 150, 255), (255, 180, 10), (101, 70, 86),
				    (38, 230, 0), (255, 120, 70), (117, 41, 121),
				    (150, 255, 0), (132, 0, 255), (24, 209, 255),
				    (191, 130, 35), (219, 200, 109), (154, 62, 86),
				    (255, 190, 190), (255, 0, 255), (152, 163, 55),
				    (192, 79, 212), (230, 230, 230), (53, 130, 64),
				    (155, 249, 152), (87, 64, 34), (214, 209, 175),
				    (170, 0, 59), (255, 0, 0), (193, 195, 234), (70, 72, 115),
				    (255, 255, 0), (52, 57, 131), (12, 83, 45)]
		self.class_colors = np.array(self.CLASS_COLORS, dtype='uint8')
		self.class_colors_without_void = self.class_colors[1:]
	
	def color_label(self, label, with_void=True):
		if with_void:
			colors = self.class_colors
		else:
			colors = self.class_colors_without_void
		cmap = np.asarray(colors, dtype='uint8')

		return cmap[label]


	def get_preprocessor(self, depth_mean,depth_std,depth_mode='refined',height=None,width=None,phase='train',train_random_rescale=(1.0, 1.4)):
		assert phase in ['train', 'test']

		if phase == 'train':
			transform_list = [
				RandomRescale(train_random_rescale),
				RandomCrop(crop_height=height, crop_width=width),
				RandomHSV((0.9, 1.1),
					(0.9, 1.1),
					(25, 25)),
				RandomFlip(),
				ToTensor(),
				Normalize(depth_mean=depth_mean,
					depth_std=depth_std,
					depth_mode=depth_mode),
				MultiScaleLabel(downsampling_rates=[8, 16, 32])
				]

		else:
			if height is None and width is None:
				transform_list = []
			else:
				transform_list = [Rescale(height=height, width=width)]
			transform_list.extend([
				ToTensor(),
				Normalize(depth_mean=depth_mean,
				depth_std=depth_std,
				depth_mode=depth_mode)
				])
		transform = transforms.Compose(transform_list)
		return transform



	def prepare_data(self):
		dataset, preprocessor = prepare_data(args, with_input_orig=True)
		return dataset

	def get_model(self):

		model, device = build_model(n_classes=37)
		checkpoint = torch.load('/home/chris/GAN_SLAM/src/segmentation/trained_models/sunrgbd/r34_NBt1D.pth', map_location=lambda storage, loc: storage)
		model.load_state_dict(checkpoint['state_dict'])
		
		if self.verbose == True:
			print('Loaded checkpoint from {}'.format(args.ckpt_path))

		model.eval()
		model.to(device)
		
		return model
	
	def _load_img(fp):
		img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
		if img.ndim == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img

	def predict(self, image, depth):
		model = self.get_model()
		preprocessor = self.get_preprocessor(
			height=480,
			width=640,
			depth_mean=19025.14930492213,
			depth_std=9880.916071806689,
			depth_mode='raw',
			phase='test'
		    )
		
		sample = preprocessor({'image': image, 'depth': depth})
		image = sample['image'][None].cuda()
		depth = sample['depth'][None].cuda()
		# apply network
		pred = model(image, depth)
		pred = F.interpolate(pred, (480, 640),
		                     mode='bilinear', align_corners=False)
		pred = torch.argmax(pred, dim=1)
		pred = pred.cpu().numpy().squeeze().astype(np.uint8)

		# show result
		pred_colored = self.color_label(pred, with_void=False)

		fig, axs = plt.subplots(1, 3, figsize=(16, 3))
		[ax.set_axis_off() for ax in axs.ravel()]
		axs[0].imshow(image.cpu().numpy().squeeze()[0].reshape((480, 640)))
		axs[1].imshow(depth.cpu().numpy().reshape((480, 640)), cmap='gray')
		plt.imshow(pred_colored)
		plt.show()
		# plt.savefig('./result.jpg', dpi=150)
		human_img = self.semantic_graph.compute_semantic_graph(pred_colored)
		return image, human_img

		

class Rescale:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = cv2.resize(image, (self.width, self.height),
                           interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.width, self.height),
                           interpolation=cv2.INTER_NEAREST)

        sample['image'] = image
        sample['depth'] = depth

        if 'label' in sample:
            label = sample['label']
            label = cv2.resize(label, (self.width, self.height),
                               interpolation=cv2.INTER_NEAREST)
            sample['label'] = label

        return sample


class RandomRescale:
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        target_scale = np.random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))

        image = cv2.resize(image, (target_width, target_height),
                           interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (target_width, target_height),
                           interpolation=cv2.INTER_NEAREST)

        sample['image'] = image
        sample['depth'] = depth

        return sample


class RandomCrop:
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.rescale = Rescale(self.crop_height, self.crop_width)

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        h = image.shape[0]
        w = image.shape[1]
        if h <= self.crop_height or w <= self.crop_width:
            # simply rescale instead of random crop as image is not large enough
            sample = self.rescale(sample)
        else:
            i = np.random.randint(0, h - self.crop_height)
            j = np.random.randint(0, w - self.crop_width)
            image = image[i:i + self.crop_height, j:j + self.crop_width, :]
            depth = depth[i:i + self.crop_height, j:j + self.crop_width]

            sample['image'] = image
            sample['depth'] = depth

        return sample


class RandomHSV:
    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h = img_hsv[:, :, 0]
        img_s = img_hsv[:, :, 1]
        img_v = img_hsv[:, :, 2]

        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        sample['image'] = img_new

        return sample


class RandomFlip:
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        sample['image'] = image
        sample['depth'] = depth
        sample['label'] = label

        return sample


class Normalize:
    def __init__(self, depth_mean, depth_std, depth_mode='refined'):
        assert depth_mode in ['refined', 'raw']
        self._depth_mode = depth_mode
        self._depth_mean = [depth_mean]
        self._depth_std = [depth_std]

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        image = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        if self._depth_mode == 'raw':
            depth_0 = depth == 0

            depth = torchvision.transforms.Normalize(
                mean=self._depth_mean, std=self._depth_std)(depth)

            # set invalid values back to zero again
            depth[depth_0] = 0

        else:
            depth = torchvision.transforms.Normalize(
                mean=self._depth_mean, std=self._depth_std)(depth)

        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor:
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype('float32')

        sample['image'] = torch.from_numpy(image).float()
        sample['depth'] = torch.from_numpy(depth).float()

        if 'label' in sample:
            label = sample['label']
            sample['label'] = torch.from_numpy(label).float()

        return sample


class MultiScaleLabel:
    def __init__(self, downsampling_rates=None):
        if downsampling_rates is None:
            self.downsampling_rates = [8, 16, 32]
        else:
            self.downsampling_rates = downsampling_rates

    def __call__(self, sample):
        label = sample['label']

        h, w = label.shape

        sample['label_down'] = dict()

        # Nearest neighbor interpolation
        for rate in self.downsampling_rates:
            label_down = cv2.resize(label.numpy(), (w // rate, h // rate),
                                    interpolation=cv2.INTER_NEAREST)
            sample['label_down'][rate] = torch.from_numpy(label_down)

        return sample


