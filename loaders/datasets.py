import os
import glob
import pickle
import csv

import torch
from loaders.base import BaseDataset

class KineticsDataset(BaseDataset):

	data_path = "kinetics"
	loader_method = "lintel"
	stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
	n_classes = 600

	def get_clips(self):

		videos = glob.glob(os.path.join(self.config.data_base_path, self.data_path, "vid/*/*.mp4"))
		vid_ids = [fn.split('/')[-1][:-4] for fn in videos]

		text_labels = [vid_path.split('/')[-2] for vid_path in videos]
		unique_labels = sorted(list(set(text_labels)))
		label_ids = dict(zip(unique_labels, list(range(len(unique_labels)))))
		labels = [label_ids[lbl] for lbl in text_labels]

		with open(os.path.join(self.config.data_base_path, self.data_path, "meta", "vid_lengths.pkl")) as f:
			lengths_data = pickle.load(f)
		vid_lengths = [lengths_data[i] for i in vid_ids]

		clips = list(zip(videos, labels, vid_lengths, vid_ids))
		clips = [{"path": c[0], "label": c[1], "length": c[2], "id": c[3], "framerange": (0, c[2]-1)} for c in clips]
		return clips

class AVADataset(BaseDataset):

	data_path = "ava"
	loader_method = "lintel"
	stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
	n_classes = 80

	def get_clips(self):

		if self.split == "train": ann_files = ["ava_train_v2.1.csv",]
		elif self.split == "val": ann_files = ["ava_val_v2.1.csv",]
		elif self.split == "trainval": ann_files = ["ava_train_v2.1.csv", "ava_val_v2.1.csv"]
		else: raise ValueError("Invalid dataset split.")

		fieldnames = ["video_id", "middle_frame_timestamp", "person_box_x1", "person_box_y1", "person_box_x2", "person_box_y2", "action_id", "person_id"]
		annotations = []
		for fn in ann_files:
			fn = os.path.join(self.config.data_base_path, self.data_path, "meta", fn)
			with open(fn) as f:
				reader = csv.DictReader(f, fieldnames=fieldnames)
				annotations.extend(list(reader))

		video_extensions = [".mp4", ".mkv", ".webm"]
		for clip in annotations:
			path = os.path.join(self.config.data_base_path, self.data_path, "vid", clip["video_id"])
			found_file = False
			for ext in video_extensions:
				if os.path.isfile(path + ext):
					clip["path"] = path + ext
					found_file = True
			if not found_file:
				raise Exception("Could not find video from annotation list.")

		for clip in annotations:
			clip["label"] = clip["action_id"]

		vid_fps = {}
		vid_fns = list(set([clip["path"] for clip in annotations]))
		for vid_fn in vid_fns:
			vid = cv2.VideoCapture(vid_fn)
			if not vid.isOpened(): vid_fps[vid_fn] = 0
			vid_fps[vid_fn] = int(vid.get(cv2.CAP_PROP_FPS))

		for clip in annotations:
			fps = vid_fps[clip[path]]
			mid_frame = clip["middle_frame_timestamp"] * fps
			interval = (mid_frame - fps, mid_frame + fps)
			clip["framerange"] = interval

		return annotations
