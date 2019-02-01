import os
import glob
import math
import random
import tqdm
import pickle
import multiprocessing as mp

import cv2
import numpy as np

vid_dir = "/mnt/data2/kinetics/vid/"
out_fn = "/mnt/data2/kinetics/meta/vid_lengths.pkl"
workers = 16

def main():

	videos = glob.glob(os.path.join(vid_dir, "*/*.mp4"))
	videos = sorted(videos)

	pool = mp.Pool(workers)
	lengths = list(tqdm.tqdm(pool.imap(get_vid_length, videos), total=len(videos)))

	lengths = dict(lengths)
	with open(out_fn, "wb") as f:
		pickle.dump(lengths, f)

def get_vid_length(vid_fn):
	vid_id = vid_fn.split('/')[-1][:-4]
	vid = cv2.VideoCapture(vid_fn)
	if not vid.isOpened(): return (vid_fn, 0)
	n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	return (vid_id, n_frames)

if __name__ == "__main__":
	main()