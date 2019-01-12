import math
import numbers
import random

import numpy as np
import cv2

import torch
import torch.nn.functional as F

class Compose:

	def __init__(self, *augmentations):
		self.augmentations = augmentations

	def __call__(self, frames):

		for transform in self.augmentations:
			frames = transform(frames)

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

		n, c, h, w = frames.shape
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

