import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import numpy as np
from torchmetrics import ConfusionMatrix
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import pickle

class NET:
    def __init__(self, model, clss_task_list, device, index = ""):
        self.task_class = clss_task_list[0]
        self.class_num = sum(clss_task_list)
        self.model = model
        self.device = device
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_before_task = 0
        self.index = index
    
    def load_dataset(self, GData, batchsize=64):
        self.dataset, self.train_set, self.val_set, self.test_set = GData.get_dataset()