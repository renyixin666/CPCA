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
import dgl
import pickle


class NET:
    def __init__(self, model, clss_task_list, device, inter_task_edge = True, mini_batch = True, dataset_name = "CoraFull", index = ""):
        self.task_class = clss_task_list[0]
        self.class_num = sum(clss_task_list)
        self.model = model
        self.device = device
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_before_task = 0
        self.class_labels = []
        self.class_features = []
        self.class_cov = []
        self.inter_task_edge = inter_task_edge
        self.test_data_uptonow_batch = []
        self.test_data_uptonow_n = []
        self.mini_batch = mini_batch

        self.dataset_name = dataset_name
        self.index = index

    def load_dataset(self, NData, clss_task_list):
        self.NData = NData
        n_cls_all = sum(clss_task_list)
        cls_all = list(range(n_cls_all))
        # self.graph, self.ids_per_cls_all, [self.train_ids, self.valid_ids, \
        #                                     self.test_ids] = self.NData.get_graph(cls_all)
        
    def load_task_dataset(self, task_class, task_id):
        self.task_class = task_class
        if self.inter_task_edge == False:
            ### task_now train/val/test ###
            # print([self.class_before_task, self.class_before_task + self.task_class])
            cls_retain = list(range(self.class_before_task, self.class_before_task + self.task_class))
            self.task_subgraph, self.task_ids_per_cls, [self.task_train_ids, \
                    self.task_valid_ids, self.task_test_ids] = self.NData.get_graph(tasks_to_retain=cls_retain)
            with open(f'./dataset_split/{self.dataset_name}_{task_id}.pkl','wb') as f:
                pickle.dump([self.task_subgraph, self.task_ids_per_cls, [self.task_train_ids, self.task_valid_ids, self.task_test_ids]], f)

    def end_task(self):
        self.class_before_task += self.task_class