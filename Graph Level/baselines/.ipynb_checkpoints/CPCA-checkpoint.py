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

class cpca:
    def __init__(self, model, class_num_first, device, index = ""):
        self.task_class = class_num_first
        self.aug_class = class_num_first
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
        self.index = index


    def load_dataset(self, GData, batchsize=64):
        self.dataset, self.train_set, self.val_set, self.test_set = GData.get_dataset()
        self.train_loader, self.val_loader, self.test_loader = GData.get_dataloader(batchsize)

    def load_task_dataset(self,task_id, task_class):
        self.task_class = task_class
        self.task_train_loader = self.train_loader[task_id]
        self.task_val_loader = self.val_loader[task_id]
        self.task_test_loader = self.test_loader[task_id]

    def before_train(self, task_now):
        self.aug_class = self.class_before_task + self.task_class + int(self.task_class*(self.task_class-1)/2)
        self.model.Incremental_learning(self.class_before_task, self.aug_class)
        self.model.train()
        self.model.to(self.device)
    
        
    def generate_label_batch(self, y_a, y_b):
        y_a = y_a.reshape(-1,1)
        y_b = y_b.reshape(-1,1)
        y_cat = torch.cat((y_a,y_b),1)
        y_cat,_ = torch.sort(y_cat,descending=False)
        y_cat = y_cat - self.class_before_task
        y_a_new = y_cat[:,0]
        y_b_new = y_cat[:,1]
        label_index = self.class_before_task + self.task_class + (((2*self.task_class-y_a_new-1)*y_a_new)/2).long() + y_b_new - y_a_new - 1
        return label_index

    def model_train(self, task_now, debug=True, if_eval=True, n_epoch=100, lamb_kd=0.1, lamb_se = 1, use_mu = True):
        self.use_mu = use_mu
        lamb_kd = lamb_kd
        lamb_se = lamb_se
        if task_now==0:
            opt = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
            scheduler = StepLR(opt, step_size=50, gamma=0.95)
        else:
            opt = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
            scheduler = StepLR(opt, step_size=50, gamma=0.9)
        loss_train = []

        for epoch in range(n_epoch):
            self.model.train()
            loss_train = []
            for batch_id_train, batch_data in enumerate(self.task_train_loader):
                smiles, bg, labels, masks = batch_data
                node_feats = bg.ndata['h']
                
                labels = labels.long()
                
                alpha = 20.0
                mix_time = 4
                emb_x = self.model.feature_extractor(bg.to(self.device),node_feats.to(self.device))
                emb_y = labels.to(self.device)
                mix_emb = None
                mix_target = None
                bz = len(emb_y)

                lam_bz = np.random.beta(alpha,alpha,bz*mix_time)
                lam_bz[np.abs(lam_bz-0.5)>0.1] = 0.5
                lam_bz = torch.tensor(lam_bz, dtype=torch.float32).to(self.device)
                for mix in range(mix_time):
                    index_shuffle = np.random.permutation(bz)
                    x_reb = emb_x[index_shuffle,:]
                    y_reb = emb_y[index_shuffle]
                    ind = (y_reb == emb_y).reshape(-1)
                    x_mix_batch = x_reb[ind == False,:]
                    y_mix_batch = y_reb[ind == False]
                    emb_x_batch = emb_x[ind == False,:]
                    emb_y_batch = emb_y[ind == False]

                    mix_label = self.generate_label_batch(y_mix_batch, emb_y_batch)
                    l = len(x_mix_batch)
                    lam = lam_bz[mix*bz:mix*bz+l].reshape(-1,1)
                    if mix_emb is None:
                        mix_emb = lam*emb_x_batch +(1-lam)*x_mix_batch
                        mix_target = mix_label
                    else:
                        mix_emb = torch.cat((mix_emb, lam*emb_x_batch +(1-lam)*x_mix_batch),0)
                        mix_target = torch.cat((mix_target, mix_label),0)

                # new_target = torch.Tensor(mix_target).reshape(-1,1)
                y = torch.cat((emb_y, mix_target.long()),0)

                x = torch.cat((emb_x, mix_emb),0)
                loss = F.cross_entropy(self.model.fc(x), y)

                if task_now > 0:
                    old_feature = self.old_model.feature_extractor(bg.to(self.device),node_feats.to(self.device))
                    new_feature = self.model.feature_extractor(bg.to(self.device),node_feats.to(self.device))
                    loss_kd = torch.dist(new_feature, old_feature, 2)

                    feature_select = []
                    label_select =[]
                    # index = list(range(self.class_before_task))
                    index = list(range(len(self.class_features)))

                    col_n = len(self.class_features[0])

                    bz = 64
                    for _ in range(bz):
                        np.random.shuffle(index)
                        index_select = index[0]
                        label_select.append(self.class_labels[index_select])
                        if use_mu:
                            feature_select.append(self.class_features[index_select])
                        else:
                            feature_select.append(self.class_features[index_select] + np.random.normal(0, 1, col_n) * self.class_cov[index_select])


                    label_tensor = torch.from_numpy(np.float32(np.asarray(label_select))).type(torch.LongTensor).to(self.device) 
                    feature_tensor = torch.from_numpy((np.asarray(feature_select))).type(torch.FloatTensor).to(self.device) 

                    loss_se = F.cross_entropy(self.model.fc(feature_tensor),label_tensor)
                    
                    loss += lamb_kd * loss_kd + lamb_se * loss_se
                    
                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step()

#                 if if_eval:
#                     if batch_id_train%20==0 and epoch%30==0:
#                         self.model.eval()
#                         acc = 0
#                         num = 0
#                         for _ , batch_data in enumerate(self.task_test_loader):
#                             smiles, bg, labels, masks = batch_data
#                             node_feats = bg.ndata['h']
#                             logits = self.model(bg.to(self.device),node_feats.to(self.device))[0]
#                             pred = logits.argmax(1)
#                             acc_b = torch.sum((labels.to(self.device)==pred)).item()
#                             acc += acc_b
#                             num += len(labels)
#                         print("CL step:", "epoch ",epoch, "test_acc ", acc/num)

#                     if batch_id_train%20==0 and epoch%30==0:
#                         self.model.eval()
#                         acc_uptonow = 0
#                         num_uptonow = 0
#                         for test_loader_now in self.test_loader[:task_now+1]:
#                             for _ , batch_data in enumerate(test_loader_now):
#                                 smiles, bg, labels, masks = batch_data
#                                 node_feats = bg.ndata['h']
#                                 logits = self.model(bg.to(self.device),node_feats.to(self.device))[0]
#                                 pred = logits.argmax(1)
#                                 acc_b = torch.sum((labels.to(self.device)==pred)).item()
#                                 acc_uptonow += acc_b
#                                 num_uptonow += len(labels)
#                         print("CL step Uptonow:", "epoch ",epoch, "test_acc ", acc_uptonow/num_uptonow)

    def after_train(self, task_now):
        self.model.Cut_class(self.class_before_task+self.task_class)
        path_save = './Model_save/'
        filename = path_save + '%d_CPCA.pkl'% task_now
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()

    def cal_feature_class(self, debug=True):
        self.model.eval()

        feature_save = None
        label_save = None
        for batch_id_train, batch_data in enumerate(self.task_train_loader):
            smiles, bg, labels, masks = batch_data
            node_feats = bg.ndata['h']

            labels = labels.long()
            if feature_save is None:
                feature_save = self.model.feature_extractor(bg.to(self.device),node_feats.to(self.device)).cpu().detach().numpy()
                label_save = labels.cpu().detach().numpy()
            else:
                feature_save = np.concatenate((feature_save, self.model.feature_extractor(bg.to(self.device)\
                                            ,node_feats.to(self.device)).cpu().detach().numpy()))
                label_save = np.concatenate((label_save,labels.cpu().detach().numpy()))


        label_set = np.unique(label_save)

        class_feature = []
        class_label = []
        class_cov = []
        for lab in label_set:
            class_label.append(lab)
            index = np.where(lab==label_save)[0]
            feature_classwise = feature_save[index]
            class_feature.append(np.mean(feature_classwise,axis=0))
            if not self.use_mu:
                sigma = np.sqrt(np.var(feature_classwise,axis = 0)).reshape(-1)
                class_cov.append(sigma)

        self.class_features.extend(class_feature)
        self.class_labels.extend(class_label)
        self.class_cov.extend(class_cov)

    def end_task(self):
        self.class_before_task += self.task_class

    def evaluation(self, task_now, clss_task_list):
        self.model.eval()
        if task_now == 0:
            # dim: task_num * class_num ##
            self.eval_result = np.zeros((len(clss_task_list), sum(clss_task_list)))
            self.num_result = np.zeros((len(clss_task_list), sum(clss_task_list)))
            self.performance_matrix = np.zeros((len(clss_task_list),len(clss_task_list)))

        gt = None
        pt = None 
        for test_loader_now in self.test_loader[:task_now+1]:
            for _ , batch_data in enumerate(test_loader_now):
                smiles, bg, labels, masks = batch_data
                node_feats = bg.ndata['h']
                logits = self.model(bg.to(self.device),node_feats.to(self.device))[0]
                logits_con = logits[:,0:self.class_before_task+self.task_class]
                pred = logits_con.argmax(1)
                labels = labels.long()
                if gt is None:
                    gt = labels
                    pt = pred.cpu().detach()
                else:
                    gt = torch.cat((gt,labels))
                    pt = torch.cat((pt,pred.cpu().detach()))
        
        ### plot the confusion matrix for class accuray ###
        # self.plot_confusion_matrix(task_now, sum(clss_task_list), pt, gt)

        ### ac for all class ###
        for i in range(sum(clss_task_list)):
            index_i = np.where(gt==i)[0]
            if len(index_i) == 0:
                acc_i = 0.
            else:
                acc_i = sum(pt[index_i] == gt[index_i])/len(index_i)
            self.eval_result[task_now, i] = acc_i
            self.num_result[task_now, i] = len(index_i)*1.0
        
        left = 0
        right = 0
        for i,clss in enumerate(clss_task_list):
            right = left + clss
            if sum(self.num_result[task_now, left:right])!=0:
                self.performance_matrix[task_now, i] = sum(self.eval_result[task_now, left:right]*\
                    self.num_result[task_now, left:right])/sum(self.num_result[task_now, left:right])
            left = right
        # print(self.eval_result, self.num_result, self.performance_matrix)
        ### At the end plot the performace_matrix ###
        if task_now == len(clss_task_list)-1:
            self.plot_performance_matrix()
            ### calculate AF(Average forget) and AP(Average accuary) ###
            self.cal_AP_AF()

#     def plot_confusion_matrix(self, task_now, num_classes, pt, gt):
#         self.model.eval()
#         cm =  ConfusionMatrix(num_classes = num_classes)
#         label_now = torch.ones(num_classes)
#         labels_not_zero,count_labels = torch.unique(gt,return_counts=True)
#         label_now[labels_not_zero] = count_labels.float()
#         recm = cm(pt,gt)/label_now.reshape(-1,1)
        
#         plt.figure(figsize=(10,10))
#         plt.imshow(recm)
#         plt.clim(vmin=0, vmax=1)
#         fig = plt.gcf()
#         plt.show()

    def plot_performance_matrix(self):
        mask = np.tri(self.performance_matrix.shape[0],k=-1).T
        acc_matrix_mean = np.ma.array(self.performance_matrix, mask = mask)
        plt.figure(figsize=(10,10))
        im = plt.imshow(acc_matrix_mean)
        plt.clim(vmin=0, vmax=1)
        cbar = plt.colorbar(im, ticks=[0, 50, 100])  # , fontsize = 15)
        cbar.ax.tick_params()
        fig = plt.gcf()
        plt.show()

    def cal_AP_AF(self):
        n = len(self.performance_matrix)
        # ap_vec = np.mean(self.performance_matrix,axis = 1)
        ap_vec = np.zeros(n)
        af_vec = np.zeros(n)
        
        for i in range(n):
            ap_vec[i] = np.mean(self.performance_matrix[i,:i+1])

        for i in range(1,n):
            backward = []
            for j in range(i):
                b = self.performance_matrix[i,j]-self.performance_matrix[j,j]
                backward.append(b)
            af_vec[i] = np.mean(backward)
        
        print("End! ","the AP", ap_vec, "the AF", af_vec)









        

