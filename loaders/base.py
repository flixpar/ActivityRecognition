import torch
import torch.utils.data

import random

import transforms.preprocess
import transforms.spatial
import transforms.temporal
import transforms.postprocess

import loaders.frame_loaders

class BaseDataset(torch.utils.data.Dataset):

	##############################
	######## Core Methods ########
	##############################

	def __init__(self, split, config, transform_mode="default", n_samples=None, debug=False):

		self.split = split
		self.config = config
		self.transform_mode = transform_mode
		self.n_samples = n_samples
		self.debug = debug

		self.clips = self.get_clips()
		if debug: self.clips = random.sample(self.clips, 100)
		elif n_samples is not None: self.clips = random.sample(self.clips, n_samples)

		self.frame_sampler = self.get_frame_sampler()
		self.frame_loader  = self.get_loader()
		self.preprocessor  = self.get_preprocess()
		self.transforms    = self.get_transforms()
		self.postprocessor = self.get_postprocess()

		self.class_weights = self.compute_class_weights()
		self.example_weights = self.compute_example_weights()

	def __getitem__(self, index):
		clip = self.clips[index]
		framelist = self.frame_sampler(clip["framerange"])
		frames = self.frame_loader(clip, framelist)
		frames = self.preprocessor(frames)
		frames = self.transforms(frames)
		frames = self.postprocessor(frames)
		return frames, clip["label"]

	def __len__(self):
		return len(self.clips)

	##############################
	####### Config Parsing #######
	##############################

	def get_frame_sampler(self):
		if self.config.frame_selection == "random":
			return transforms.temporal.RandomFrameSamplerWithoutReplacement(self.config.clip_length)
		elif self.config.frame_selection == "random_replace":
			return transforms.temporal.RandomFrameSamplerWithReplacement(self.config.clip_length)
		elif self.config.frame_selection == "uniform":
			return transforms.temporal.UniformFrameSampler(self.config.clip_length)
		elif self.config.frame_selection == "first":
			return transforms.temporal.FirstKFrameSampler(self.config.clip_length)
		elif self.config.frame_selection == "center":
			return transforms.temporal.MiddleKFrameSampler(self.config.clip_length)
		else:
			raise ValueError("Invalid frame selection method.")

	def get_preprocess(self):
		return transforms.preprocess.Compose([
			transforms.preprocess.ToTensor()
		])

	def get_postprocess(self):
		mean, std = self.stats["mean"], self.stats["std"]
		return transforms.postprocess.Compose([
			transforms.postprocess.Normalize(mean=mean, std=std),
			transforms.postprocess.Resize(self.config.frame_size)
		])

	def get_loader(self):
		if self.config.loader == "default": method = self.loader_method
		else: method = self.config.loader
		if method == "lintel":
			return loaders.frame_loaders.LintelLoader()
		elif method == "pyav":
			return loaders.frame_loaders.PyAvLoader()
		elif method == "opencv":
			return loaders.frame_loaders.OpenCVLoader()
		elif method == "pil":
			return loaders.frame_loaders.PILLoader()
		else:
			raise NotImplementedError("Invalid loader method selection.")

	def get_transforms(self):
		if self.transform_mode == "default": self.transform_mode = self.split

		if self.transform_mode == "train":
			return self.config.train_augmentation
		elif self.transform_mode == "val":
			return self.config.train_augmentation
		elif self.transform_mode == "test":
			return transforms.spatial.TestTimeCompose(self.config.test_augmentation)
		else:
			raise ValueError("Invalid transform mode.")

	##############################
	####### Helper Methods #######
	##############################

	def compute_class_weights(self):
		labels = [clip["label"] for clip in self.clips]
		counts = torch.tensor([labels.count(k) for k in range(self.n_classes)], dtype=torch.float32)
		weights = counts.max() / (self.n_classes * counts)
		return weights

	def compute_example_weights(self):
		labels = [clip["label"] for clip in self.clips]
		weights = torch.tensor([self.class_weights[l] for l in labels], dtype=torch.float32)
		weights = weights / (weights.max() * self.n_classes)
		return weights

	##############################
	###### Required Methods ######
	##############################

	def get_clips(self):
		raise NotImplementedError("Cannot use base dataset class.")
