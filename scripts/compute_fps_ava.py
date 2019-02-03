import os
import glob
import math
import random
import tqdm
import pickle
import multiprocessing as mp

import cv2
import numpy as np

vid_dir = "/mnt/data2/ava/vid/"
out_fn = "/mnt/data2/ava/meta/vid_fps.pkl"
workers = 16

def main():

	videos = glob.glob(os.path.join(vid_dir, "*"))
	videos = [fn for fn in videos if fn.split('.')[-1] in ["mp4", "mkv", "webm"]]
	videos = sorted(videos)

	pool = mp.Pool(workers)
	data = list(tqdm.tqdm(pool.imap(get_vid_length, videos), total=len(videos)))

	data = dict(data)
	with open(out_fn, "wb") as f:
		pickle.dump(data, f)

def get_vid_length(vid_fn):
	vid_id = vid_fn.split('/')[-1][:-4]
	vid = cv2.VideoCapture(vid_fn)
	if not vid.isOpened(): return (vid_fn, 0)
	fps = int(vid.get(cv2.CAP_PROP_FPS))
	return (vid_id, fps)

if __name__ == "__main__":
	main()