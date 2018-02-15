import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
from google.protobuf import text_format
import onnx
import numpy as np
import os
from onnx_tf.backend import run_model

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=2)
    self.fc1 = nn.Linear(720, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(-1, 720)
    x = F.relu(self.fc1(x))
    return F.log_softmax(x, dim=1)

def export_pytorch():
  model = Net()

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True,
                                          transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                          ]))
  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True,
                                         transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                         ]))
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                           shuffle=False, num_workers=2)

  # for epoch in range(2):  # loop over the dataset multiple times
  #   continue
  #   running_loss = 0.0
  #   for i, data in enumerate(trainloader):
  #
  #     # get the inputs
  #     inputs, labels = data
  #
  #     # wrap them in Variable
  #     inputs, labels = Variable(inputs), Variable(labels)
  #
  #     # zero the parameter gradients
  #     optimizer.zero_grad()
  #
  #     # forward + backward + optimize
  #     outputs = model(inputs)
  #
  #     loss = criterion(outputs, labels)
  #     loss.backward()
  #     optimizer.step()
  #
  #     # print statistics
  #     running_loss += loss.data[0]
  #     if i % 2000 == 1999:  # print every 2000 mini-batches
  #       print('[%d, %5d] loss: %.3f' %
  #             (epoch + 1, i + 1, running_loss / 2000))
  #       running_loss = 0.0

  # export onnx model
  dummy_input = Variable(torch.randn(1, 3, 32, 32))
  # torch.onnx.export(model, dummy_input, "test.proto", verbose=True)
  torch.onnx.export(model, dummy_input, "pb/torch_gh.pb", verbose=False, export_params=True)
  # with open('test.proto', "rb") as f:
  #   gh_str = text_format.MessageToString(f)
  #   print(gh_str)

# Load the ONNX model
# f = open('str_graph/torch_gh_str.txt', 'r')
# bytes = f.read().encode()
# f_ = open('str_graph/torch_gh_str_temp.txt', 'wb')
# f_.write(bytes)
# f_.close()
# f.close()

# model = onnx.load("pb/torch_gh.pb")
#
# # Check that the IR is well formed
# result = onnx.checker.check_model(model)
#
# # Print a human readable representation of the graph
# content = onnx.helper.printable_graph(model.graph)
# f = open('str_graph/torch_gh_str.txt', 'w')
# f.write(content)
# f.close()

# export_pytorch()

def test():
  model = Net()
  data = np.random.randn(1, 3, 32, 32).astype(np.float32)
  input = Variable(torch.from_numpy(data))
  rs_torch = model(input)
  torch.onnx.export(model, input, "pb/torch_gh.pb", verbose=False, export_params=True)
  print(rs_torch)

  with open(os.path.join('pb/torch_gh.pb'), "rb") as f:
    test = onnx.load(f)
    # data = np.transpose(data, (0, 2, 3, 1))
    rs_onnx = run_model(test, [data])
    print(rs_onnx)

  np.allclose(rs_torch, rs_onnx)

test()