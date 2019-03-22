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
		with open(clip_info["path"], "rb") as f: vid = f.read()
		decoded_frames, width, height = lintel.loadvid_frame_nums(vid, frame_nums=frames, should_seek=True)
		decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
		decoded_frames = np.reshape(decoded_frames, newshape=(len(frames), height, width, 3))
		return decoded_frames

class PyAvLoader:

	def __call__(self, clip_info, frames):
		container = av.open(clip_info["path"])
		container.streams.video[0].thread_type = 'AUTO'
		container.seek(frames[0], whence='frame', backward=True, any_frame=True)
		decoded_frames = None
		j = 0
		for i, frame in enumerate(container.decode(video=0)):
			if i + frames[0] not in frames: continue
			if decoded_frames is None:
				decoded_frames = np.empty((len(frames), frame.height, frame.width, 3), dtype=np.uint8)
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
