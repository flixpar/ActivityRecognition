import os
import tqdm
import numpy as np
from sklearn import metrics

import torch
from torch import nn
import torch.nn.functional as F

from utils.logger import Logger
from utils.arg_handler import (get_dataset, get_model, get_loss,
		get_train_sampler, get_scheduler, get_optimizer)

from args import Args
args = Args()

primary_device = torch.device("cuda:{}".format(args.device_ids[0]))

def main():

	# datasets
	dataset_class = get_dataset(args)

	train_dataset = dataset_class(split=args.train_split, config=args, n_samples=args.n_train_samples, debug=args.debug)
	tval_dataset  = dataset_class(split=args.train_split, config=args, n_samples=args.n_tval_samples,  debug=args.debug)
	vval_dataset  = dataset_class(split=args.val_split,   config=args, n_samples=args.n_vval_samples,  debug=args.debug)

	# sampling
	train_sampler = get_train_sampler(args, train_dataset)
	shuffle = (train_sampler is None)

	# dataloaders

	train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=shuffle, sampler=train_sampler,
		batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

	tval_loader = torch.utils.data.DataLoader(tval_dataset, shuffle=False,
		batch_size=1, num_workers=args.workers, pin_memory=True)

	vval_loader = torch.utils.data.DataLoader(vval_dataset, shuffle=False,
		batch_size=1, num_workers=args.workers, pin_memory=True)

	# model
	model = get_model(args, train_dataset.n_classes).cuda()
	model = nn.DataParallel(model, device_ids=args.device_ids)
	model.to(primary_device)

	# training
	loss_func = get_loss(args, train_dataset.class_weights).to(primary_device)
	optimizer = get_optimizer(args, model)
	scheduler = get_scheduler(args, optimizer)

	logger = Logger()
	max_score = 0.0

	for epoch in range(1, args.epochs+1):
		logger.print("Epoch {}".format(epoch))
		scheduler.step()
		train(model, train_loader, loss_func, optimizer, logger)
		t_scores = evaluate(model, tval_loader, loss_func, logger, splitname="train")
		v_scores = evaluate(model, vval_loader, loss_func, logger, splitname="val")
		logger.save()
		if v_scores["acc"] > max_score:
			logger.save_model(model.module, epoch)
			max_score = v_scores["acc"]

	logger.save()
	logger.save_model(model, "final")


def train(model, train_loader, loss_func, optimizer, logger):
	model.train()

	losses = []
	for i, (clips, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):

		clips = clips.to(primary_device, dtype=torch.float32, non_blocking=True)
		labels = labels.to(primary_device, dtype=torch.long, non_blocking=True)

		outputs = model(clips)
		loss = loss_func(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		logger.log_loss(loss.item())
		if i % (len(train_loader)//args.log_freq) == 0:
			mean_loss = np.mean(logger.losses[-10:])
			tqdm.tqdm.write("Train loss: {}".format(mean_loss))
			logger.log("Train loss: {}".format(mean_loss))

def evaluate(model, loader, loss_func, logger, splitname="val"):
	model.eval()

	losses = []
	preds = []
	targets = []

	with torch.no_grad():
		for clips, labels in tqdm.tqdm(loader, total=len(loader)):

			clips = clips.to(primary_device, dtype=torch.float32, non_blocking=True)
			labels = labels.to(primary_device, dtype=torch.long, non_blocking=True)

			outputs = model(clips)
			loss = loss_func(outputs, labels).item()

			pred = outputs
			if pred.shape[0] != 1:
				pred = (0.5 * pred[0, :]) + (0.5 * pred[1:, :].mean(axis=0))
				pred = pred[np.newaxis, :]

			pred = torch.softmax(pred, dim=-1).argmax(dim=-1)
			pred = pred.cpu().numpy()

			labels = labels.cpu().numpy().astype(np.int).squeeze()

			losses.append(loss)
			preds.append(pred)
			targets.append(labels)

	targets = np.array(targets).squeeze()
	preds = np.array(preds).squeeze()

	acc = metrics.accuracy_score(targets, preds)
	f1 = metrics.f1_score(targets, preds, average="macro")
	f1_perclass = metrics.f1_score(targets, preds, average=None)
	loss = np.mean(losses)

	logger.print()
	logger.print("Eval - {}".format(splitname))
	logger.print("Loss:", loss)
	logger.print("Accuracy:", acc)
	logger.print("Macro F1:", f1)
	logger.print("Per-Class F1:", f1_perclass)
	logger.print()

	logger.log_eval({f"{splitname}-loss": loss, f"{splitname}-acc": acc, f"{splitname}-f1": f1})

	scores = {
		"acc": acc,
		"f1": f1,
		"f1-perclass": f1_perclass,
		"loss": loss
	}
	return scores

if __name__ == "__main__":
	main()
