import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd


class SubgraphEncoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_vertices, num_subvertices, dropout=0.2, is_training=False):
        super(SubgraphEncoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(num_subvertices * hidden_dim, hidden_dim)
        self.is_training = is_training
        self.dropout = 0.2
        self.num_vertices = num_vertices
        self.num_subvertices = num_subvertices

    def embed(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, edge_index):
        embeddings = self.embed(x,edge_index)
        # x = F.dropout(x, p=self.dropout, training=self.is_training)
        x=embeddings
        x = x.view(x.shape[0]//self.num_subvertices, x.shape[1] * self.num_subvertices) # For every vertex, concatenate every subvertex's embedding (belonging to that vertex) together into one new "vertex" embedding
        x = self.linear(x)                                                              # reduce the concatenated vertex embedding dimension back down to hidden_dim
        return x, embeddings
    


class GraphEncoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.2, is_training=False):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.is_training = is_training
        self.dropout = 0.2

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        # x = F.dropout(x, p=self.dropout, training=self.is_training)
        return x


class InnerProductDecoder(torch.nn.Module):
    r"""Pytorch Geometrit Inner Product Decoder Implementation"""
    
    def forward(self, z, edge_index,sigmoid = True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


    # def forward_all(self, z, sigmoid = True):
    #     adj = torch.matmul(z, z.t())
    #     return torch.sigmoid(adj) if sigmoid else adj


class MultiviewEncoder(torch.nn.Module):
    def __init__(self, SubgraphEncoder, GraphEncoder, is_training=False):
        super(MultiviewEncoder, self).__init__()
        self.encoder_g = SubgraphEncoder                            #  encoder for gene level graph
        self.encoder_c = GraphEncoder                               #  encoder for cell level graph
        self.is_training = is_training
        # self.multihead_attn = nn.MultiheadAttention(hidden_dim, 2, average_attn_weights=True)


    def forward(self, x_c, x_g, edge_index_c, edge_index_g):
        Z_g, gene_embeddings = self.encoder_g(x_g, edge_index_g)
        Z_c = self.encoder_c(x_c, edge_index_c)
        Z = torch.cat((Z_c,Z_g),dim=1)
        # print(torch.max(Z), torch.min(Z))
        # print(Z)

        assert Z.shape[1] == 2 * Z_g.shape[1]

        return Z, Z_c, Z_g, gene_embeddings
    


class MultiviewGAE(torch.nn.Module):
    def __init__(self, SubgraphEncoder, GraphEncoder, decoder = InnerProductDecoder(), is_training=False):
        super(MultiviewGAE, self).__init__()
        self.encoder_g = SubgraphEncoder                            #  encoder for gene level graph
        self.encoder_c = GraphEncoder                               #  encoder for cell level graph
        self.decoder = decoder                                      #  decoder (default is inner product)
        self.is_training = is_training
        # self.multihead_attn = nn.MultiheadAttention(hidden_dim, 2, average_attn_weights=True)


    def forward(self, x_c, x_g, edge_index_c, edge_index_g):
        Z_gg = self.encoder_g(x_g, edge_index_g)
        Z_cc = self.encoder_c(x_c, edge_index_c)
        Z = torch.cat((Z_cc,Z_gg),dim=1)
        # print(torch.max(Z), torch.min(Z))
        # print(Z)

        assert Z.shape[1] == 2 * Z_gg.shape[1]

        reconstructed_A = self.decoder(Z)
        reconstructed_Ag = self.decoder(Z_gg)

        return reconstructed_A, reconstructed_Ag
    
    
