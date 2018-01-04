import torch
import torch.nn as nn

class ZSSRNet(nn.Module):
	def __init__(self, input_channels=3, kernel_size=3, channels=64):
		 super(ZSSRNet, self).__init__()
		 
		 self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		 self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		 self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		 self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		 self.conv4 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		 self.conv5 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		 self.conv6 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
		 self.conv7 = nn.Conv2d(channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True)

		 self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv0(x))
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.relu(self.conv4(x))
		x = self.relu(self.conv5(x))
		x = self.relu(self.conv6(x))
		x = self.conv7(x)

		return x