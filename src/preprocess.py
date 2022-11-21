import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from umap import UMAP
import sys
sys.path.append('./submodules/CeSpGRN/src/')
from submodules.CeSpGRN.src import *
from submodules.CeSpGRN.src import kernel
from submodules.CeSpGRN.src import g_admm as CeSpGRN
import time
from rich.progress import track
from rich.table import Table
from rich.console import Console

import random


def convert_adjacencylist2edgelist(adj_list): # do we need to account for repeat edges?
    '''
    Converts adjacency list to edge list format for pytorch geometric processing.
    
    :adj_list: nd.array(shape=(numnodes, numneighbors)) : adjacency list representation
    
    :return: nd.array(shape=(2, numnodes * numneighbors))
    '''
    
    edge_list = []
    
    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            edge_list.append([node, neighbor])
            
    return np.array(edge_list).T



def select_LRgenes(data_df, num_genespercell, lr_database = 0):
    '''
    Selects LR and relevant background genes to be included for GRN inference.
    
    :data_df: pd.DataFrame : represents the spatial data and contains the following columns ["Cell_ID", "X", "Y", "Cell_Type", "Gene 1", ..., "Gene n"]
    :num_genespercell: int : represents the number of genes to be included for each Cell-Specific GRN
    :lr_database: int: 0/1/2 corresponding to LR database (0: CellTalkDB)
    
    :return : pd.DataFrame with relevant gene columns preserved
    '''
    if lr_database == 0:
        sample_counts = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis = 1)
        lr_df = pd.read_csv("../data/celltalk_human_lr_pair.txt", sep="\t")
        
        receptors = set(lr_df["receptor_gene_symbol"].to_list())
        ligands = set(lr_df["ligand_gene_symbol"].to_list())
        real2uppercase = {x:x.upper() for x in sample_counts.columns}
        uppercase2real = {upper:real for real,upper in real2uppercase.items()}
        candidate_genes = set(np.vectorize(real2uppercase.get)(sample_counts.columns.to_numpy()))
        
        selected_ligands = candidate_genes.intersection(ligands)
        selected_receptors = candidate_genes.intersection(receptors)
        
        print(f"Found {len(selected_ligands)} ligands and {len(selected_receptors)} receptors to be included in the {num_genespercell} selected genes per cell. \n")
        
        
        
        num_genesleft = num_genespercell - len(selected_ligands) - len(selected_receptors)
        candidate_genesleft = candidate_genes - selected_ligands - selected_receptors
        
        selected_randomgenes = set(random.sample(tuple(candidate_genesleft), num_genesleft))
        
        selected_genes = list(selected_randomgenes | selected_ligands | selected_receptors)
        
        selected_columns = ["Cell_ID", "X", "Y", "Cell_Type"] + np.vectorize(uppercase2real.get)(selected_genes).tolist()
        selected_df = data_df[selected_columns]
        
        console = Console()
        table = Table(show_header=True, header_style="bold")
        table.add_column("Selected Ligands", style="cyan")
        table.add_column("Selected Receptors", style="deep_pink3")
        table.add_column("Selected Random Genes", justify="right")
        table.add_row("\n".join(selected_ligands),"\n".join(selected_receptors),"\n".join(selected_randomgenes))
        console.print(table)
        
        return selected_df, selected_ligands|selected_receptors
    else:
        raise Exception("Invalid lr_database type")
    

def infer_initial_grns(data_df, cespgrn_hyperparams):
    
    '''
    Infers the starting cell specific GRNs with CeSpGRN submodule.
    
    :data_df: pd.DataFrame : represents the spatial data and contains the following columns ["Cell_ID", "X", "Y", "Cell_Type", "Gene 1", ..., "Gene n"]
    :cespgrn_hyperparams: dict(): dictionary of CeSpGRN hyperparameters
    
    :return: nd.array(shape=(numcells, numgenespercell, numgenespercell))
    '''
    
    counts = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis = 1).values
    # print(f"GRNs are dimension ({counts.shape[1]} by {counts.shape[1]}) for each of the {counts.shape[0]} cells\n")
    
    # Normalize Counts??? 
    pca_op = PCA(n_components = 20)
    X_pca = pca_op.fit_transform(counts)

    # hyper-parameters
    bandwidth = cespgrn_hyperparams["bandwidth"]
    n_neigh = cespgrn_hyperparams["n_neigh"]
    lamb = cespgrn_hyperparams["lamb"]
    max_iters = cespgrn_hyperparams["max_iters"]

    start_time = time.time()
    # calculate the kernel function
    K, K_trun = kernel.calc_kernel_neigh(X_pca, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = n_neigh)

    # estimate covariance matrix, output is empir_cov of the shape (ncells, ngenes, ngenes)
    empir_cov = CeSpGRN.est_cov(X = counts, K_trun = K_trun, weighted_kt = True)

    # estimate cell-specific GRNs
    cespgrn = CeSpGRN.G_admm_minibatch(X=counts[:, None, :], K=K, pre_cov=empir_cov, batchsize = 120)
    grns = cespgrn.train(max_iters=max_iters, n_intervals=100, lamb=lamb)
    

    # print("Total time inferring GRNs: {:.2f} sec".format(time.time() - start_time))
    # print()

    return grns



def construct_celllevel_graph(data_df, k, get_edges=False):   # Top k closest neighbors for each cell
    '''
    Constructs new cell graph with spatial proximity edges based on kNN.
    
    :data_df: pd.DataFrame : represents the spatial data and contains the following columns ["Cell_ID", "X", "Y"]
    :k: int: Number of nearest neighbors to construct spatial edges for
    :get_edges: boolean:  True to return edge_trace (for visualization purposes)
    
    :return: Cell_level_adjacency, edge list
    '''
    
    adjacency = np.zeros(shape=(len(data_df), k),dtype=int) # shape = (numcells, numneighbors of cell)
    coords = np.vstack([data_df["X"].values,data_df["Y"].values]).T

    edges = None
    edge_x = []
    edge_y = []

    # for i in tqdm(range(len(data_df)), desc=f"2. Constructing Cell-Level Graph from ST Data", colour="cyan", position=1):
    for i in track(range(len(data_df)), description=f"[cyan]2. Constructing Cell-Level Graph from ST Data"):
        cell_id = data_df["Cell_ID"][i]
        x0, y0 = data_df["X"].values[i],data_df["Y"].values[i]
        candidate_cell = coords[i]
        candidate_neighbors = coords
        euclidean_distances = np.linalg.norm(candidate_neighbors - candidate_cell,axis=1)
        neighbors = np.argsort(euclidean_distances)[1:k+1]
        adjacency[i] = neighbors
        assert i not in adjacency[i]
        if get_edges:
            for ncell in adjacency[i]:
                x1, y1 = data_df["X"].values[ncell],data_df["Y"].values[ncell]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
        
    edges=[edge_x,edge_y]
    
        # for cell,adj in enumerate(tqdm(adjacency,desc=f"Getting the {k} edges of each cell", colour="cyan", position=1)):
        #     x0, y0 = coordinate_df["X"].values[cell],coordinate_df["Y"].values[cell]
        #     for ncell in adjacency[cell]:
        #         x1, y1 = coordinate_df["X"].values[ncell],coordinate_df["Y"].values[ncell]
        #         edge_x.append(x0)
        #         edge_x.append(x1)
        #         edge_x.append(None)
        #         edge_y.append(y0)
        #         edge_y.append(y1)
        #         edge_y.append(None)
                
    return adjacency,edges




def construct_genelevel_graph(disjoint_grns, celllevel_adj_list, node_type = "int"):
    '''
    Constructs gene level graph.
    
    :disjoint_grns: nd.array(shape=(numcells, numgenes, numgenes)) : np array of cell-specific GRNs (output of CeSpGRN)
    :celllevel_adj_list: nd.array(shape=(numcells, k)): adjacency list of cell level graph (output of preprocess.construct_celllevel_graph)
    :node_type: str: Either "int" or "str" to set for the node labels
    
    :return: nx.Graph object (gene level), mapping between integer node names to gene names, reverse mapping
    '''
    
    numgenes = disjoint_grns[0].shape[0]
    numcells = disjoint_grns.shape[0]
    num2gene = {}   #dictionary that maps the supergraph integer nodes to the gene name (Cell#_gene#)
    gene2num = {}   #dictionary that maps the gene name (Cell#_gene#) to the supergraph integer node


    grn_graph_list = []
    for cellnum, grn in enumerate(track(disjoint_grns, description=f"[cyan]3a. Combining individual GRNs")):
        G =  nx.from_numpy_matrix(grn)
        grn_graph_list.append(G)
        for i in range(numgenes):
            num2gene[cellnum*numgenes+i] = f"Cell{cellnum}_Gene{i}"
            gene2num[f"Cell{cellnum}_Gene{i}"] = cellnum * numgenes + i

    union_of_grns = nx.disjoint_union_all(grn_graph_list)
    
    gene_level_graph = nx.relabel_nodes(union_of_grns, num2gene)  # relabel nodes to actual gene names

    for cell, neighborhood in enumerate(track(celllevel_adj_list,\
        description=f"[cyan]3b. Constructing Gene-Level Graph")): # for each cell in the ST data
        for genenum1 in range(numgenes):                            # for each gene in the cell                         45
            node1 = f"Cell{cell}_Gene{genenum1}"
            for ncell in neighborhood:                                  # for each neighborhood cell adjacent to cell   5
                candidate_neighbors=np.arange(numgenes,dtype=int)
                np.random.shuffle(candidate_neighbors)
                for genenum2 in candidate_neighbors[:numgenes//3]:          # for each gene in the neighborhood cell    45
                    node2 = f"Cell{ncell}_Gene{genenum2}"
                    gene_level_graph.add_edge(node1, node2)

    
    if node_type == "str":
        gene_level_graph = gene_level_graph
    elif node_type == "int":
        gene_level_graph = nx.convert_node_labels_to_integers(gene_level_graph)

    assert len(union_of_grns.nodes()) == numcells * numgenes

    return gene_level_graph, num2gene, gene2num
        