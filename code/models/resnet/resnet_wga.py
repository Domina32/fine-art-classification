import os

import numpy as np
import torch
from torch.cuda import device
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50

DATA_PATH = "./data/wga/"

tensor_x_train = torch.Tensor(np.load(os.path.join(DATA_PATH, "wga_X_train.npy")))
tensor_y_train = torch.Tensor(np.load(os.path.join(DATA_PATH, "wga_y_train.npy")))

train_dataloader = DataLoader(TensorDataset(tensor_x_train, tensor_y_train))

print(tensor_x_train.shape)

device = device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = resnet50(pretrained=True)