import torch
import torch.nn.functional as F

import numpy as np
import cv2

class Compose:

	def __init__(self, *transformations):
		self.transformations = transformations

	def __call__(self, frames):

		for transformations in self.transformations:
			frames = transformations(frames)

		return frames

class ToTensor:

	def __init__(self, flip_color=False):
		self.flip_color = flip_color

	def __call__(self, frames):
		frames = torch.from_numpy(frames).float()
		frames = frames.permute(0, 3, 1, 2)
		if self.flip_color: frames = frames[None, [2, 1, 0], None, None]
		frames.div_(255)
		return frames

class Resize:

	def __init__(self, size):
		self.size = size

	def __call__(self, frames):
		frames = F.interpolate(frames, self.size)
		return frames

class CropStatic:

	def __init__(self):
		raise NotImplementedError()

	def __call__(self, frames):
		raise NotImplementedError()

class CropMoving:

	def __init__(self):
		raise NotImplementedError()

	def __call__(self, frames):
		raise NotImplementedError()

class ContrastNormalization:

	def __init__(self):
		raise NotImplementedError()

	def __call__(self, frames):
		raise NotImplementedError()
