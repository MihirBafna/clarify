import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
import networkx as nx
from sklearn.model_selection import train_test_split



def construct_celllevel_graph(coordinate_df, k, get_edges=False):   # Top k closest neighbors for each cell
    '''
    Constructs new cell graph with spatial proximity edges based on kNN.
    
    coordinate_df: pd.DataFrame : represents the spatial data and contains the following columns ["Cell_ID", "X", "Y"]
    k: int: Number of nearest neighbors to construct spatial edges for
    get_edges: boolean:  True to return edge_trace (for visualization purposes)
    
    return: Cell_level_adjacency, edge list
    '''
    
    adjacency = np.zeros(shape=(len(coordinate_df), k),dtype=int) # shape = (numcells, numneighbors of cell)
    coords = np.vstack([coordinate_df["X"].values,coordinate_df["Y"].values]).T

    edges = None
    edge_x = []
    edge_y = []

    for i in tqdm(range(len(coordinate_df)), desc=f"2. Construct Cell-Level Graph from ST Data", colour="cyan", position=1):
        cell_id = coordinate_df["Cell_ID"][i]
        x0, y0 = coordinate_df["X"].values[i],coordinate_df["Y"].values[i]
        candidate_cell = coords[i]
        candidate_neighbors = coords
        euclidean_distances = np.linalg.norm(candidate_neighbors - candidate_cell,axis=1)
        neighbors = np.argsort(euclidean_distances)[1:k+1]
        adjacency[i] = neighbors
        assert i not in adjacency[i]
        if get_edges:
            for ncell in adjacency[i]:
                x1, y1 = coordinate_df["X"].values[ncell],coordinate_df["Y"].values[ncell]
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
    
    disjoint_grns: nd.array(shape=(numcells, numgenes, numgenes)) : np array of cell-specific GRNs (output of CeSpGRN)
    celllevel_adj_list: nd.array(shape=(numcells, k)): adjacency list of cell level graph (output of preprocess.construct_celllevel_graph)
    node_type: str: Either "int" or "str" to set for the node labels
    
    return: nx.Graph object (gene level), mapping between integer node names to gene names, reverse mapping
    '''
    
    numgenes = disjoint_grns[0].shape[0]
    numcells = disjoint_grns.shape[0]
    num2gene = {}   #dictionary that maps the supergraph integer nodes to the gene name (Cell#_gene#)
    gene2num = {}   #dictionary that maps the gene name (Cell#_gene#) to the supergraph integer node


    grn_graph_list = []
    for cellnum, grn in enumerate(tqdm(disjoint_grns, desc="3a. Combining individual GRNs", colour="cyan")):
        G =  nx.from_numpy_matrix(grn)
        grn_graph_list.append(G)
        for i in range(numgenes):
            num2gene[cellnum*numgenes+i] = f"Cell{cellnum}_Gene{i}"
            gene2num[f"Cell{cellnum}_Gene{i}"] = cellnum * numgenes + i

    union_of_grns = nx.disjoint_union_all(grn_graph_list)
    
    gene_level_graph = nx.relabel_nodes(union_of_grns, num2gene)  # relabel nodes to actual gene names

    for cell, neighborhood in enumerate(tqdm(celllevel_adj_list,\
        desc="3b. Constructing Gene-Level Graph", colour="cyan")): # for each cell in the ST data
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
        