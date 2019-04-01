import os
import glob
import math
import pickle

import torch
import numpy as np

import cv2
from PIL import Image
import lintel
import av

class LintelLoader:

	def __call__(self, clip_info, frames):

		unique_frames = sorted(list(set(frames)))
		frame_inds = [unique_frames.index(f) for f in frames]

		with open(clip_info["path"], "rb") as f: vid = f.read()
		decoded_frames, width, height = lintel.loadvid_frame_nums(vid, frame_nums=unique_frames, should_seek=True)
		decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
		decoded_frames = np.reshape(decoded_frames, newshape=(-1, height, width, 3))

		decoded_frames = decoded_frames[frame_inds]
		return decoded_frames

class PyAvLoader:

	def __call__(self, clip_info, frames):
		with av.logging.Capture() as logs:
			container = av.open(clip_info["path"])
			container.seek(frames[0], whence='frame', backward=True, any_frame=True)
			init_frame = next(container.decode(video=0))
			decoded_frames = np.empty((len(frames), init_frame.height, init_frame.width, 3), dtype=np.uint8)
			decoded_frames[0] = init_frame.to_ndarray(format='rgb24')
			j = 1
			for i, frame in enumerate(container.decode(video=0)):
				if i + frames[0] not in frames: continue
				decoded_frames[j] = frame.to_ndarray(format='rgb24')
				j += 1
				if j == len(frames): break
			return decoded_frames

class OpenCVLoader:

	def __call__(self, clip_info, frames):
		path_template = clip_info["path_template"]
		clip = None
		for i, frame_num in enumerate(frames):
			if i == 0:
				frame = cv2.imread(path_template.format(frame_num))
				h, w, c = frame.shape
				clip = np.empty((len(frames), h, w, c), dtype=np.uint8)
				clip[0] = frame
			else:
				clip[i] = cv2.imread(path_template.format(frame_num))
		return clip

class PILLoader:

	def __call__(self, clip_info, frames):
		path_template = clip_info["path_template"]
		clip = None
		for i, frame_num in enumerate(frames):
			if i == 0:
				frame = Image.open(path_template.format(frame_num)).to_ndarray()
				h, w, c = frame.shape
				clip = np.empty((len(frames), h, w, c), dtype=np.uint8)
				clip[0] = frame
			else:
				clip[i] = Image.open(path_template.format(frame_num)).to_ndarray()
		return clip
