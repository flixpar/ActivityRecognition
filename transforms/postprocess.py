import torch
import torch.nn.functional as F

import numpy as np
import cv2

class Compose:

	def __init__(self, transformations):
		self.transformations = transformations

	def __call__(self, frames):

		for transform in self.transformations:
			frames = transform(frames)

		return frames

class Normalize:

	def __init__(self, mean, std):
		self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None, None]
		self.std  = torch.tensor(std,  dtype=torch.float32)[:, None, None, None]

	def __call__(self, frames):
		frames.sub_(self.mean).div_(self.std)
		return frames

class Resize:

	def __init__(self, size):
		if isinstance(size, int):
			self.size = (size, size)
		elif isinstance(size, (tuple, list)):
			self.size = size
		else:
			raise ValueError("Invalid size given for resize.")

	def __call__(self, frames):
		frames = F.interpolate(frames, self.size)
		return frames

class OpticalFlow:

	def __init__(self):
		raise NotImplementedError()

	def __call__(self, frames):
		raise NotImplementedError()
