import dgl
from dgl.data.utils import Subset
from dgllife.data import PubChemBioAssayAromaticity
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import CanonicalAtomFeaturizer
from dgllife.utils.splitters import RandomSplitter
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from functools import partial
from itertools import accumulate
import pickle


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


class GraphLevelDataset():
    def __init__(self,frac_list,clss_task_list,m):
        dataset = PubChemBioAssayAromaticity(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                            node_featurizer=CanonicalAtomFeaturizer(),
#                                           edge_featurizer=args.get('edge_featurizer', None), load=True
                                            )
        dataset.labels = dataset.labels.view(-1)
        n_per_cls = np.array([(dataset.labels == i).sum() for i in range((dataset.labels.max().int().item() + 1))])
        selected_clss = (n_per_cls > 3).nonzero()[
            0].tolist()  # at least 3 examples for train val test splittings
        ids_per_cls = {i: (dataset.labels == i).nonzero().view(-1).tolist() for i in selected_clss}
        self.n_per_cls = {i: len(ids_per_cls[i]) for i in ids_per_cls}
        label_key = list(ids_per_cls.keys())
        
        if m=="data":
            train_ids_task, val_ids_task, test_ids_task = [], [], []
            train_set = []
            val_set = []
            test_set = []
            clss_cumsum = np.cumsum(clss_task_list)
            a = [0,*clss_cumsum]
            task_seq = [list(range(0,a[i+1])) for i in range(len(a)-1)]
            for clss in task_seq:
                train_ids_task = []
                val_ids_task = []
                test_ids_task = []
                for cls in clss:
                    if cls not in label_key:
                        continue
                    ids = ids_per_cls[cls]
                    np.random.shuffle(ids)
                    assert np.allclose(np.sum(frac_list), 1.), \
                        'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))
                    num_data = len(ids)
                    lengths = (num_data * frac_list).astype(int)
                    for i in range(len(lengths) - 1, 0, -1):
                        lengths[i] = max(1, lengths[i])  # ensure at least one example for test and val
                    lengths[0] = num_data - np.sum(lengths[1:])
                    split = [ids[offset - length:offset] for offset, length in zip(accumulate(lengths), lengths)]
                    train_ids_task.extend(split[0])
                    val_ids_task.extend(split[1])
                    test_ids_task.extend(split[2])
            #         print(dataset.labels[train_ids_task])
                train_set.append(Subset(dataset, train_ids_task))
                val_set.append(Subset(dataset, val_ids_task))
                test_set.append(Subset(dataset, test_ids_task))

            with open(f'./dataset_split/dataset_split_CGL.pkl','wb') as f:
                pickle.dump([dataset, train_set, val_set, test_set], f)
            
            train_ids_task, val_ids_task, test_ids_task = [], [], []
            train_set = []
            val_set = []
            test_set = []
            clss_cumsum = np.cumsum(clss_task_list)
            a = [0,*clss_cumsum]
            task_seq = [list(range(a[i],a[i+1])) for i in range(len(a)-1)]
            for clss in task_seq:
                train_ids_task = []
                val_ids_task = []
                test_ids_task = []
                for cls in clss:
                    if cls not in label_key:
                        continue
                    ids = ids_per_cls[cls]
                    np.random.shuffle(ids)
                    assert np.allclose(np.sum(frac_list), 1.), \
                        'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))
                    num_data = len(ids)
                    lengths = (num_data * frac_list).astype(int)
                    for i in range(len(lengths) - 1, 0, -1):
                        lengths[i] = max(1, lengths[i])  # ensure at least one example for test and val
                    lengths[0] = num_data - np.sum(lengths[1:])
                    split = [ids[offset - length:offset] for offset, length in zip(accumulate(lengths), lengths)]
                    train_ids_task.extend(split[0])
                    val_ids_task.extend(split[1])
                    test_ids_task.extend(split[2])
            #         print(dataset.labels[train_ids_task])
                train_set.append(Subset(dataset, train_ids_task))
                val_set.append(Subset(dataset, val_ids_task))
                test_set.append(Subset(dataset, test_ids_task))
            
            with open(f'./dataset_split/dataset_split_CGL.pkl','wb') as f:
                pickle.dump([dataset, train_set, val_set, test_set], f)

            print("End Dataset Split!")

        else:
            print("load saved dataset!!!")
            dataset, train_set, val_set, test_set = pickle.load(open(f'./dataset_split/dataset_split_CGL.pkl','rb'))
        
        self.dataset, self.train_set, self.val_set, self.test_set = dataset, train_set, val_set, test_set
    
    def get_dataset(self):
        return self.dataset, self.train_set, self.val_set, self.test_set

    def get_dataloader(self, batchsize, shuffle=True):
        train_loader = [DataLoader(s, batch_size=batchsize, collate_fn=collate_molgraphs, shuffle=shuffle) for s in self.train_set]
        val_loader = [DataLoader(s, batch_size=batchsize, collate_fn=collate_molgraphs, shuffle=shuffle) for s in self.val_set]
        test_loader = [DataLoader(s, batch_size=batchsize, collate_fn=collate_molgraphs, shuffle=shuffle) for s in self.test_set]
        return train_loader, val_loader, test_loader



def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)  
    return smiles, bg, labels, masks

