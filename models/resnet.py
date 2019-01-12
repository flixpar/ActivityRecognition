import torch
from torch import nn
import torchvision

from models.modules import Resnet


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
		self.fc = nn.Linear(hidden_size, n_classes)
		self.statesize  = (layers*2, 1, hidden_size)
		self.outputsize = 2 * hidden_size
		self.state = self.init_state(self.statesize, None)
		self.temporal_pooling = "mean"

	def forward(self, x):
		x = self.cnn(x)
		x = x.unsqueeze(0).permute(1, 0, 2)
		output, _ = self.lstm(x, self.init_state(self.statesize, x.device))
		if self.temporal_pooling == "mean": output = torch.mean(output, dim=0)
		else: output = output[-1]
		output = self.fc(output)
		return output

	def init_state(self, size, device):
		return torch.zeros(size).to(device), torch.zeros(size).to(device)

