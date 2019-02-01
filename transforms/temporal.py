import random

class RandomFrameSamplerWithReplacement:

	def __init__(self, k):
		self.k = k

	def __call__(self, frame_range):
		frames = random.choices(range(frame_range[0], frame_range[1]), k=self.k)
		frames = sorted(frames)
		return frames

class RandomFrameSamplerWithoutReplacement:

	def __init__(self, k):
		self.k = k

	def __call__(self, frame_range):
		clip_len = frame_range[1] - frame_range[0]
		if clip_len >= self.k:
			frames = random.sample(range(frame_range[0], frame_range[1]), k=self.k)
		else:
			frames = random.choices(range(frame_range[0], frame_range[1]), k=self.k)
		frames = sorted(frames)
		return frames

class FirstKFrameSampler:

	def __init__(self, k):
		self.k = k

	def __call__(self, frame_range):
		clip_len = frame_range[1] - frame_range[0]
		if clip_len >= self.k:
			frames = list(range(frame_range[0], frame_range[0]+self.k))
		else:
			repeats = self.k // clip_len
			extra = self.k % clip_len
			frames = list(range(frame_range[0], frame_range[1])) * repeats
			frames += list(range(frame_range[0], frame_range[0] + extra))
		return frames

class MiddleKFrameSampler:

	def __init__(self, k):
		self.k = k

	def __call__(self, frame_range):
		clip_len = frame_range[1] - frame_range[0]
		if clip_len >= self.k:
			start_frame = frame_range[0] + (clip_len // 2) - (self.k // 2)
			frames = list(range(start_frame, start_frame+self.k))
		else:
			repeats = self.k // clip_len
			extra = self.k % clip_len
			start_frame = frame_range[0] + (clip_len // 2) - (extra // 2)
			frames = list(range(frame_range[0], frame_range[1])) * repeats
			frames += list(range(start_frame, start_frame + extra))
		return frames

class UniformFrameSampler:

	def __init__(self, k):
		self.k = k

	def __call__(self, frame_range):
		clip_len = frame_range[1] - frame_range[0]
		interval = clip_len / self.k
		frames = [frame_range[0] + round(interval * i) for i in range(self.k)]
		return frames
