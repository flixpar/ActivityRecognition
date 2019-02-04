import math
import numbers
import random

import numpy as np
import cv2

import torch
import torch.nn.functional as F

class Compose:

	def __init__(self, augmentations):
		self.augmentations = augmentations

	def __call__(self, frames):

		for transform in self.augmentations:
			frames = transform(frames)

		return frames

class TestTimeCompose:

	def __init__(self, augmentations):
		self.augmentations = augmentations
		self.n_copies = len(self.augmentations)

	def __call__(self, frames):
		c, f, h, w = frames.shape
		output = torch.empty(self.n_copies + 1, c, f, h, w)
		output[0] = frames.clone().detach()
		for i, transform in enumerate(self.augmentations):
			output[i+1] = transform(frames.clone().detach())
		return output

class NoOp:
	def __call__(self, frames):
		return frames

class RandomCrop:

	def __init__(self, size, padding=0):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.padding = padding

	def __call__(self, frames):
		if self.padding > 0:
			frames = F.pad(frames, self.padding, mode="constant", value=0)

		c, f, h, w = frames.shape
		dest_h, dest_w = self.size

		if w == dest_w and h == dest_h:
			return frames
		if w < dest_w or h < dest_h:
			frames = F.interpolate(frames, self.size)
			return frames

		x1 = random.randint(0, w - dest_w)
		y1 = random.randint(0, h - dest_h)

		x2 = x1 + dest_w
		y2 = y1 + dest_h

		frames = frames[:, :, y1:y2, x1:x2]
		return frames

class RandomFlip:

	def __init__(self, flip_horizontal=True, flip_vertical=False, p=0.5):
		self.flip_horizontal = flip_horizontal
		self.flip_vertical   = flip_vertical
		self.p = p

	def __call__(self, frames):

		if self.flip_horizontal:
			if random.random() < self.p:
				frames = frames.flip(3)

		if self.flip_vertical:
			if random.random() < self.p:
				frames = frames.flip(2)

		return frames

class ColorJitter:

	def __init__(self, amount=0.2):
		self.amount = amount

	def __call__(self, frames):

		brightness_amount = 1.0 + random.uniform(-self.amount, self.amount)
		saturation_amount = 1.0 + random.uniform(-self.amount, self.amount)

		# brightness
		frames.mul_(brightness_amount)

		# saturation
		m = frames.mean(dim=(0,2,3))
		frames.sub_(m)
		frames.mul_(saturation_amount)
		frames.add_(m)

		# noise
		noise = torch.randn_like(frames) * self.amount
		frames.add_(noise)

		return frames
