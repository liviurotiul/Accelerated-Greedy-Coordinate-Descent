import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import os
import glob
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import math


def my_argmax(tensor):
	return np.unravel_index(np.argmax(tensor), np.shape(tensor))

def update_theta(theta):
	temp = 1/(theta**2)
	return torch.from_numpy(np.asarray([(-temp + math.sqrt(temp**2 + 4*temp))/2]))





class AGCD(Optimizer):
	def __init__(self, params, theta=1):
		defaults = dict(theta=theta)
		super(AGCD, self).__init__(params, defaults)
		self.x_update_params = deepcopy(self.param_groups)
		self.z_update_params = deepcopy(self.param_groups)
		self.theta_update_params = deepcopy(self.param_groups)
		for group in self.theta_update_params:
			for i in range(len(group['params'])):
				group['params'][i] = torch.ones(1)
	
		

	def __setstate__(self, state):
		super(AGCD, self).__setstate__(state)
			
	
	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss=closure()

	
		for group_y, group_x, group_z, group_theta in zip(self.param_groups,
							self.x_update_params,
							self.z_update_params,
							self.theta_update_params):
			
			for y, x, z, theta in zip(group_y['params'],
							group_x['params'],
							group_z['params'],
							group_theta['params']):
				if y.grad is None:
					continue
			
				d_y = y.grad
				j1 = j2 = my_argmax(d_y)
				dummy = y.clone().detach()
				y.copy_((x*(1-theta).expand_as(x)).add(z*theta.expand_as(z))) #use pytorch functions for math
				temp = torch.zeros(x.size())
				temp[j1] = 2*x.clone()[j1]
				dummy = x.clone().detach()
				x.copy_((y - temp).clone().detach())
				temp = torch.zeros(z.size())
				temp[j2] = 2*z.clone()[j2]
				z.sub_(temp)
				theta.copy_(update_theta(theta))
					
		return loss			

trainset = torchvision.datasets.MNIST('./files', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=False)

testset =   torchvision.datasets.MNIST('./files', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))]))

testloaderm = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

class NetMnist(nn.Module):
    def __init__(self):
        super(NetMnist, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.Conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.Conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.Conv3 = nn.Conv2d(32, 128, kernel_size=3)
        self.BN1 = nn.BatchNorm2d(16)
        self.BN2 = nn.BatchNorm2d(32)
        self.BN3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.BN1(self.pool(F.relu(self.Conv1(x))))
        x = self.BN2(self.pool(F.relu(self.Conv2(x))))
        x = self.BN3(self.pool(F.relu(self.Conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

net = NetMnist()
criterion = nn.CrossEntropyLoss()
optimizer = AGCD(net.parameters())
optimizer2 = optim.Adam(net.parameters())
ITERATIONS = 1000

losses = []

net.train()
inputs, target = iter(trainloader).next()
for idx in range(20000):
	optimizer.zero_grad()
	outputs = net(inputs)
	outputs = outputs.float()
	loss = criterion(outputs, target.long())
	loss.backward()
	optimizer.step()

	if idx % 100 == 99:
		print(loss.item())
		losses.append(loss.item())
		print(torch.max(outputs))
plt.plot(losses)
plt.show()

'''
for iteration in range(ITERATIONS):
	net.train()
	running_loss = 0
	for i, (inputs,target) in enumerate(trainloader):
		optimizer.zero_grad()
		inputs = inputs.float()
		outputs = net(inputs)
		outputs = outputs.float()
		loss = criterion(outputs, target.long())
		loss.backward()
		for idx in range(100):
			optimizer.step()
		running_loss += loss.item()
		if i% 30 == 29:
			print("[", iteration, ",",  running_loss/30, "]")
			#losses.append(running_loss/30)
			running_loss = 0.0

'''

'''
for epoch in range(2):
	running_loss = 0.0
    model.train()

    for i, (inputs, target) in enumerate(trainloaderm):
        optimizer.zero_grad()
        inputs = inputs.float()
        outputs = model(inputs)
        loss = criterion(outputs, target.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 30 == 29 :
            print("[", epoch, ",",  running_loss/30, "]")
            losses.append(running_loss/30)
            running_loss = 0.0 
'''
