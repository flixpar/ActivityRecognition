import os
import glob
import pickle

import torch
from loaders.base import BaseDataset

class KineticsDataset(BaseDataset):

	data_path = "kinetics"
	loader_method = "lintel"
	stats = {"mean": 0, "std": 0}
	n_classes = 600

	def get_clips(self):

		videos = glob.glob(os.path.join(self.config.data_base_path, self.data_path, "*/*.mp4"))
		vid_ids = [fn.split('/')[-1][:-4] for fn in videos]

		text_labels = [vid_path.split('/')[-2] for vid_path in videos]
		unique_labels = sorted(list(set(text_labels)))
		label_ids = dict(zip(unique_labels, list(range(len(unique_labels)))))
		labels = [label_ids[lbl] for lbl in text_labels]

		with open(os.path.join(self.config.data_base_path, self.data_path, "meta", "vid_lengths.pkl")) as f:
			lengths_data = pickle.load(f)
		vid_lengths = [lengths_data[i] for i in vid_ids]

		clips = list(zip(videos, labels, vid_lengths, vid_ids))
		clips = [{"path": c[0], "label": c[1], "length": c[2], "id": c[3]} for c in clips]
		return clips
