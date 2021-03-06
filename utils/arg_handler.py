import torch
from torch import nn
import torch.optim.lr_scheduler
from torch.utils.data import WeightedRandomSampler
from utils.lr_schedule import ConstantLR, PolynomialLR

from models.lstm import ResNetLSTM, ResNetTCN
from models.resnet3d import ResNet3D
from models.i3d import InceptionI3D

from loaders.datasets import KineticsDataset, AVADataset, UCF101Dataset

def get_dataset(args):
	if args.dataset == "kinetics":
		return KineticsDataset
	elif args.dataset == "charades":
		raise NotImplementedError()
	elif args.dataset == "ava":
		raise NotImplementedError()
	elif args.dataset == "ucf101":
		return UCF101Dataset
	else:
		raise ValueError("Invalid dataset selection.")

def get_model(args, n_classes):
	if args.model == "resnet-lstm":
		return ResNetLSTM(n_classes, **args.model_config)
	elif args.model == "resnet-tcn":
		return ResNetTCN(n_classes, **args.model_config)
	elif args.model == "3dresnet":
		n_layers = args.model_config["n_layers"] if "n_layers" in args.model_config else 101
		return ResNet3D(
			layers=n_layers,
			n_classes=n_classes,
			frame_size=args.frame_size,
			n_frames=args.clip_length
		)
	elif args.model == "i3d":
		return InceptionI3D(n_classes=n_classes, **args.model_config)
	else:
		raise ValueError("Invalid model selection.")

def get_loss(args, weights):

	if args.weight_method == "loss":
		if args.weight_mode is not None:
			class_weights = weights
		else:
			class_weights = None
	else:
		class_weights = None

	if args.loss == "crossentropy":
		loss_func = nn.CrossEntropyLoss(weight=class_weights)
	else:
		raise ValueError("Invalid loss function specifier: {}".format(args.loss))

	return loss_func

def get_train_sampler(args, dataset):
	if args.weight_method == "sampling":
		num_samples = args.n_train_samples if args.n_train_samples is not None else len(dataset)
		num_samples = num_samples if not args.debug else 100
		return WeightedRandomSampler(weights=dataset.example_weights, num_samples=num_samples)
	else:
		return None

def get_scheduler(args, optimizer):
	params = args.lr_schedule_params
	if args.lr_schedule == "poly":
		gamma = params["gamma"] if "gamma" in params else 0.9
		max_iter = args.epochs
		decay_iter = 1
		return PolynomialLR(optimizer, max_iter, decay_iter, gamma)
	elif args.lr_schedule == "exp":
		gamma = params["gamma"] if "gamma" in params else 0.9
		return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
	elif args.lr_schedule == "step":
		step_size = params["step_size"] if "step_size" in params else 5
		gamma = params["gamma"] if "gamma" in params else 0.5
		return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
	elif args.lr_schedule == "multistep":
		milestones = params["milestones"] if "milestones" in params else list(range(10, args.epochs, 10))
		gamma = params["gamma"] if "gamma" in params else 0.2
		return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
	elif args.lr_schedule == "cosine":
		T_max = params["period"] // 2 if "period" in params else 10
		max_decay = params["max_decay"] if "max_decay" in params else 50
		eta_min = args.initial_lr / max_decay
		return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
	else:
		return ConstantLR(optimizer)

def get_optimizer(args, model):
	if args.optimizer == "adam":
		return torch.optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
	elif args.optimizer == "sgd":
		return torch.optim.SGD(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
	else:
		raise ValueError("Invalid optimizer selection.")
