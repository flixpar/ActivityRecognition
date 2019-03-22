import torch
from torch import nn
import torchvision

from models.modules import Resnet, TemporalConvNet


class ResNetLSTM(nn.Module):

	def __init__(self, n_classes=600, resnet_layers=101, hidden_size=1024, layers=2):
		super(ResNetLSTM, self).__init__()
		self.cnn = Resnet(layers=resnet_layers, output_dims=512)
		self.lstm = nn.LSTM(
			input_size = 512,
			hidden_size = hidden_size,
			num_layers = 2,
			bidirectional = True,
		)
		self.statesize  = (layers*2, 1, hidden_size)
		self.outputsize = 2 * hidden_size
		self.state = self.init_state(self.statesize, None)
		self.temporal_pooling = "mean"
		self.fc = nn.Linear(self.outputsize, n_classes)

	def forward(self, x):
		n, c, f, h, w = x.shape
		x = x.permute(0, 2, 1, 3, 4) # (n,c,f,h,w) -> (n,f,c,h,w)
		x = x.reshape(n*f, c, h, w)
		x = self.cnn(x)
		x = x.reshape(n, f, -1).permute(1, 0, 2)
		self.lstm.flatten_parameters()
		output, _ = self.lstm(x, self.state)
		if self.temporal_pooling == "mean": output = torch.mean(output, dim=0)
		else: output = output[-1]
		output = self.fc(output)
		return output

	def init_state(self, size, device):
		return torch.zeros(size).to(device), torch.zeros(size).to(device)

class ResNetTCN(nn.Module):

	def __init__(self, n_classes=600, resnet_layers=101, hidden_size=1024, layers=2):
		super(ResNetTCN, self).__init__()
		self.cnn = Resnet(layers=resnet_layers, output_dims=512)
		self.tcn = TemporalConvNet(512, [hidden_size]*layers)
		self.fc = nn.Linear(hidden_size, n_classes)
		self.temporal_pooling = "mean"

	def forward(self, x):
		n, c, f, h, w = x.shape
		x = x.permute(0, 2, 1, 3, 4) # (n,c,f,h,w) -> (n,f,c,h,w)
		x = x.reshape(n*f, c, h, w)
		x = self.cnn(x)
		x = x.reshape(n, f, -1).permute(0, 2, 1)
		output = self.tcn(x)
		if self.temporal_pooling == "mean": output = torch.mean(output, dim=2)
		else: output = output[:, :, -1]
		output = self.fc(output)
		return output
