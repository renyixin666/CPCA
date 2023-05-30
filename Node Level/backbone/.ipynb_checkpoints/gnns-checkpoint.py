from .gnnconv import GATConv, GCNLayer, GINConv
from .layers import PairNorm
from .utils import *
linear_choices = {'nn.Linear':nn.Linear, 'Linear_IL':Linear_IL}

class GIN(nn.Module):
    def __init__(self,
                 args,):
        super(GIN, self).__init__()
        dims = [args.d_data] + args.GIN_args['h_dims'] + [args.n_cls]
        # self.dropout = args.GIN_args['dropout']
        self.dropout = 0
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            lin = torch.nn.Linear(dims[l], dims[l+1])
            self.gat_layers.append(GINConv(lin, 'sum'))


    def forward(self, g, features):
        e_list = []
        h, e = self.gat_layers[0](g, features)
        x = F.relu(h)
        logits, e = self.gat_layers[1](g, x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h, e = self.gat_layers[0].forward_batch(blocks[0], features)
        x = F.relu(h)
        logits, e = self.gat_layers[1].forward_batch(blocks[1], x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GIN_original(nn.Module):
    def __init__(self, args, ):
        super().__init__()
        dims = [args.d_data] + args.GIN_args['h_dims'] + [args.n_cls]
        # self.dropout = args.GIN_args['dropout']
        self.dropout = 0
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims) - 1):
            lin = torch.nn.Linear(dims[l], dims[l + 1])
            self.gat_layers.append(GINConv(lin, 'sum'))

    def forward(self, g, features):
        e_list = []
        h, e = self.gat_layers[0](g, features)
        x = F.relu(h)
        # e_list = e_list + e
        logits, e = self.gat_layers[1](g, x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h, e = self.gat_layers[0].forward_batch(blocks[0], features)
        x = F.relu(h)
        logits, e = self.gat_layers[1].forward_batch(blocks[1], x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GCN(nn.Module):
    def __init__(self,
                 args):
        super(GCN, self).__init__()
        dims = [args['d_data']] + args['h_dims'] + [args['n_cls']]
        # self.dropout = args.GCN_args['dropout']
        self.dropout = 0
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            self.gat_layers.append(GCNLayer(dims[l], dims[l+1]))

    def forward(self, g, features):
        e_list = []
        h = features
        for layer in self.gat_layers[:-1]:
            h, e = layer(g, h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits, e = self.gat_layers[-1](g, h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

from dgl.nn import GraphConv

class GCN_Increment(nn.Module):
    def __init__(self,
                args):
        super(GCN_Increment, self).__init__()
        dims = [args['d_data']] + args['h_dims'] + [args['n_cls']]
        # self.conv1 = GraphConv(dims[0], dims[1])
        # self.conv2 = GraphConv(dims[1], dims[1])
        self.conv1 = GCNLayer(dims[0], dims[1])
        self.conv2 = GCNLayer(dims[1], dims[1])
        self.fc = nn.Linear(dims[1], dims[-1], bias=True)
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(self.conv1)
        self.gat_layers.append(self.conv2)
        
        self.dropout = 0
        
    def forward(self, g, features):
        # h = self.conv1(g, in_feat)
        # h = F.relu(h)
        # h = self.conv2(g, h)
        # logits = self.fc(h)
        # return logits
        e_list = []
        h = features
        for layer in self.gat_layers[:-1]:
            h, e = layer(g, h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        emb, e = self.gat_layers[-1](g, h)
        e_list = e_list + e
        logits = self.fc(emb)
        return logits, e_list
    
    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        emb, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        e_list = e_list + e
        logits = self.fc(emb)
        return logits, e_list
    
    def Incremental_learning(self, old_class, new_class):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = old_class

        self.fc = nn.Linear(in_feature, new_class, bias=True)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]
    
    def feature_extractor(self, g, features):
        # h = self.conv1(g, in_feat)
        # h = F.relu(h)
        # h = self.conv2(g, h)
        e_list = []
        h = features
        for layer in self.gat_layers[:-1]:
            h, e = layer(g, h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        emb, e = self.gat_layers[-1](g, h)
        e_list = e_list + e
        return emb
    
    def feature_extractor_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        emb, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        return emb

class GAT(nn.Module):
    def __init__(self,
                 args,
                 heads,
                 activation):
        super(GAT, self).__init__()
        #self.g = g
        self.num_layers = args.GAT_args['num_layers']
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            args.d_data, args.GAT_args['num_hidden'], heads[0],
            args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], False, None))
        # self.norm_layers.append(nn.BatchNorm1d(num_hidden*heads[0]))
        self.norm_layers.append(PairNorm())
        
        # hidden layers
        for l in range(1, args.GAT_args['num_layers']):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                args.GAT_args['num_hidden'] * heads[l-1], args.GAT_args['num_hidden'], heads[l],
                args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], args.GAT_args['residual'], self.activation))
            # self.norm_layers.append(nn.BatchNorm1d(num_hidden*heads[l]))
            self.norm_layers.append(PairNorm())
        # output projection

        self.gat_layers.append(GATConv(
            args.GAT_args['num_hidden'] * heads[-2], args.n_cls, heads[-1],
            args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], args.GAT_args['residual'], None))

    def forward(self, g, inputs, save_logit_name = None):
        h = inputs
        e_list = []
        for l in range(self.num_layers):
            h, e = self.gat_layers[l](g, h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        # store for ergnn
        self.second_last_h = h
        # output projection
        logits, e = self.gat_layers[-1](g, h)
        #self.second_last_h = logits if len(self.gat_layers) == 1 else h
        logits = logits.mean(1)
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        logits = logits.mean(1)
        e_list = e_list + e
        return logits, e_list


    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()