import random
import pickle
import numpy as np
import torch
from torch import Tensor, device, dtype
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from dgl.data import CoraGraphDataset, CoraFullDataset, register_data_args, RedditDataset
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl, Evaluator
import copy
from sklearn.metrics import roc_auc_score, average_precision_score


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)

class incremental_graph_trans_(nn.Module):
    def __init__(self,dataset,n_cls):
        super().__init__()
        self.graph, self.labels = dataset[0]
        #self.graph = dgl.add_reverse_edges(self.graph)
        #self.graph = dgl.add_self_loop(self.graph)
        self.graph.ndata['label'] = self.labels
        self.d_data = self.graph.ndata['feat'].shape[1]
        self.n_cls = n_cls
        self.n_nodes = self.labels.shape[0]
        self.tr_va_te_split = dataset[1]

    def get_graph(self, tasks_to_retain=[], node_ids = None, remove_edges =True):
        node_ids_ = copy.deepcopy(node_ids)
        node_ids_retained = []
        ids_train_old, ids_valid_old, ids_test_old = [], [], []
        if len(tasks_to_retain) > 0:
            for t in tasks_to_retain:
                ids_train_old.extend(self.tr_va_te_split[t][0])
                ids_valid_old.extend(self.tr_va_te_split[t][1])
                ids_test_old.extend(self.tr_va_te_split[t][2])
                node_ids_retained.extend(self.tr_va_te_split[t][0]+self.tr_va_te_split[t][1]+self.tr_va_te_split[t][2])
            subgraph_0 = dgl.node_subgraph(self.graph, node_ids_retained, store_ids=True)
            if node_ids_ is None:
                subgraph = subgraph_0
        if node_ids_ is not None:
            if not isinstance(node_ids_[0],list):
                # if nodes are not divided into different tasks
                subgraph_1 = dgl.node_subgraph(self.graph, node_ids_, store_ids=True)
                if remove_edges:
                    # to facilitate the methods like ER-GNN to only retrieve nodes
                    n_edges = subgraph_1.edges()[0].shape[0]
                    subgraph_1.remove_edges(list(range(n_edges)))

            elif isinstance(node_ids_[0],list):
                # if nodes are diveded into different tasks
                subgraph_1 = dgl.node_subgraph(self.graph, node_ids_[0], store_ids=True) # load the subgraph containing nodes of the first task
                node_ids_.pop(0)
                for ids in node_ids_:
                    # merge the remaining nodes
                    subgraph_1 = dgl.batch([subgraph_1,dgl.node_subgraph(self.graph, ids, store_ids=True)])

            if len(tasks_to_retain)==0:
                subgraph = subgraph_1
                
        if len(tasks_to_retain)>0 and node_ids is not None:
            subgraph = dgl.batch([subgraph_0,subgraph_1])

        old_ids = subgraph.ndata['_ID'].cpu()
        ids_train = [(old_ids == i).nonzero()[0][0].item() for i in ids_train_old]
        ids_val = [(old_ids == i).nonzero()[0][0].item() for i in ids_valid_old]
        ids_test = [(old_ids == i).nonzero()[0][0].item() for i in ids_test_old]
        node_ids_per_task_reordered = []
        for c in tasks_to_retain:
            ids = (subgraph.ndata['label'] == c).nonzero()[:, 0].view(-1).tolist()
            node_ids_per_task_reordered.append(ids)
        subgraph = dgl.add_self_loop(subgraph)

        return subgraph, node_ids_per_task_reordered, [ids_train, ids_val, ids_test]
        
def train_valid_test_split(ids,ratio_valid_test):
    va_te_ratio = sum(ratio_valid_test)
    train_ids, va_te_ids = train_test_split(ids, test_size=va_te_ratio)
    return [train_ids] + train_test_split(va_te_ids, test_size=ratio_valid_test[1]/va_te_ratio)

class NodeLevelDataset(incremental_graph_trans_):
    def __init__(self,name='ogbn-arxiv',IL='class',default_split=False,ratio_valid_test=None,args=None):

        # return an incremental graph instance that can return required subgraph upon request
        if name[0:4] == 'ogbn':
            data = DglNodePropPredDataset(name, root="")
            graph, label = data[0]
        elif name in ['CoraFullDataset', 'CoraFull','corafull', 'CoraFull-CL','Corafull-CL']:
            data = CoraFullDataset()
            graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
        elif name in ['reddit','Reddit','Reddit-CL']:
            data = RedditDataset(self_loop=False)
            graph, label = data.graph, data.labels.view(-1, 1)
        elif name == 'Arxiv-CL':
            data = DglNodePropPredDataset('ogbn-arxiv', root="")
            graph, label = data[0]
        elif name == 'Products-CL':
            data = DglNodePropPredDataset('ogbn-products', root="")
            graph, label = data[0]
        else:
            print('invalid data name')
        n_cls = data.num_classes
        cls = [i for i in range(n_cls)]
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls}
        cls_sizes = {c: len(cls_id_map[c]) for c in cls_id_map}
        self.cls_sizes = cls_sizes
        for c in cls_sizes:
            if cls_sizes[c] < 2:
                cls.remove(c) # remove classes with less than 2 examples, which cannot be split into train, val, test sets
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls}
        n_cls = len(cls)
        if default_split:
            split_idx = data.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"].tolist(), split_idx["valid"].tolist(), split_idx[
                "test"].tolist()
            print(len(train_idx), len(valid_idx), len(test_idx))
            tr_va_te_split = {c: [list(set(cls_id_map[c]).intersection(set(train_idx))),
                                  list(set(cls_id_map[c]).intersection(set(valid_idx))),
                                  list(set(cls_id_map[c]).intersection(set(test_idx)))] for c in cls}

        elif not default_split:
            # print("not default_split")
            # split_name = f'{args.data_path}/tr{round(1-ratio_valid_test[0]-ratio_valid_test[1],2)}_va{ratio_valid_test[0]}_te{ratio_valid_test[1]}_split_{name}.pkl'
            # try:
            #     tr_va_te_split = pickle.load(open(split_name, 'rb')) # could use same split across different experiments for consistency
            # except:
            if ratio_valid_test[1] > 0:
                tr_va_te_split = {c: train_valid_test_split(cls_id_map[c], ratio_valid_test=ratio_valid_test[1:])
                                  for c in
                                  cls}
                print(f'splitting is {ratio_valid_test}')
            elif ratio_valid_test[1] == 0:
                tr_va_te_split = {}
                for c in cls:
                    train_ids, test_ids = train_test_split(cls_id_map[c], test_size=ratio_valid_test[2])
                    tr_va_te_split[c] = [train_ids,[],test_ids]
                # with open(split_name, 'wb') as f:
                #     pickle.dump(tr_va_te_split, f)
        super().__init__([[graph, label], tr_va_te_split], n_cls)


