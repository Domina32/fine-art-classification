import torch
from torch.cuda import device
from torchvision.models import ResNet50_Weights, resnet50

DATA_PATH = "./data/wga"

device = device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
