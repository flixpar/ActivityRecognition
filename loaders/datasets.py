import os
import glob
import pickle
import csv

import torch
from loaders.base import BaseDataset

class KineticsDataset(BaseDataset):

	loader_method = "lintel"
	stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
	n_classes = 600

	def get_clips(self):

		videos = glob.glob(os.path.join(self.config.data_base_path, "kinetics/vid/*/*.mp4"))
		vid_ids = [fn.split('/')[-1][:-4] for fn in videos]

		with open(os.path.join(self.config.data_base_path, "kinetics/meta/vid_info.pkl"), "rb") as f:
			vid_info = pickle.load(f)

		indices_filter = [i for i, vid_id in enumerate(vid_ids) if vid_id in vid_info and not vid_info[vid_id]["error"]]
		videos  = [videos[i]  for i in indices_filter]
		vid_ids = [vid_ids[i] for i in indices_filter]

		vid_lengths = [vid_info[i]["n_frames"] for i in vid_ids]
		vid_lengths = [v-5 for v in vid_lengths]

		text_labels = [vid_path.split('/')[-2] for vid_path in videos]
		unique_labels = sorted(list(set(text_labels)))
		label_ids = dict(zip(unique_labels, list(range(len(unique_labels)))))
		labels = [label_ids[lbl] for lbl in text_labels]

		clips = list(zip(videos, labels, vid_lengths, vid_ids))
		clips = [{"path": c[0], "label": c[1], "length": c[2], "id": c[3], "framerange": (0, c[2]-1)} for c in clips]
		return clips

class AVADataset(BaseDataset):

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
			fn = os.path.join(self.config.data_base_path, "ava/meta/", fn)
			with open(fn, "r") as f:
				reader = csv.DictReader(f, fieldnames=fieldnames)
				annotations.extend(list(reader))

		video_extensions = [".mp4", ".mkv", ".webm"]
		for clip in annotations:
			path = os.path.join(self.config.data_base_path, "ava/vid/", clip["video_id"])
			found_file = False
			for ext in video_extensions:
				if os.path.isfile(path + ext):
					clip["path"] = path + ext
					found_file = True
			if not found_file:
				raise Exception("Could not find video from annotation list.")

		for clip in annotations:
			clip["label"] = clip["action_id"]
			clip["id"] = clip["video_id"]

		with open(os.path.join(self.config.data_base_path, "ava/meta/vid_info.pkl"), "rb") as f:
			vid_info = pickle.load(f)

		for clip in annotations:
			fps = vid_info[clip["id"]]["fps"]
			mid_frame = clip["middle_frame_timestamp"] * fps
			interval = (mid_frame - fps, mid_frame + fps)
			clip["framerange"] = interval

		return annotations
