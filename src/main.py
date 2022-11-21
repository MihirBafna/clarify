import numpy as np
import pandas as pd
import networkx as nx
import os
import argparse
from rich.table import Table
from rich.console import Console
import torch

import preprocessing as preprocessing
import training as training
import visualize as vis


def parse_arguments():
    parser = argparse.ArgumentParser(description='clarifyGAE arguments')
    parser.add_argument("-m", "--mode", type=str, default = "train",
        help="clarifyGAE mode: preprocess|train|test")
    parser.add_argument("-n", "--numgenespercell", type=int, default = 45,
            help="Number of genes in each gene regulatory network")
    parser.add_argument("-k", "--nearestneighbors", type=int, default = 5,
                help="Number of nearest neighbors for each cell")
    parser.add_argument("-i", "--inputdirpath", type=str,
                    help="Input directory path where ST data is stored")
    parser.add_argument("-o", "--outputdirpath", type=str,
                    help="Output directory path where results will be stored ")
    parser.add_argument("-l", "--lrdatabase", type=int, default=0,
                    help="0/1/2 for which Ligand-Receptor Database to use")
    args = parser.parse_args()
    return args



def preprocess(st_data, num_nearestneighbors, cespgrn_hyperparameters):    
        
    # 1. Infer initial GRNs with CeSpGRN
    grns = preprocessing.infer_initial_grns(st_data, cespgrn_hyperparameters) # shape (ncells, ngenes, ngenes)
    # grns = np.load("../out/preprocessing_output/seqfish_grns.npy")

    # 2. Construct Cell-Level Graph from ST Data
    celllevel_adj, _ = preprocessing.construct_celllevel_graph(st_data, num_nearestneighbors, get_edges=False)

    #  3. Construct Gene-Level Graph from ST Data + GRNs 
    gene_level_graph, num2gene, gene2num = preprocessing.construct_genelevel_graph(grns, celllevel_adj, node_type="int")

    return celllevel_adj, gene_level_graph, num2gene, gene2num, grns



def train(data, hyperparams = None):
    num_cells, num_cellfeatures = data[0].x.shape[0], data[0].x.shape[1]
    hidden_dim = hyperparams["concat_hidden_dim"] // 2
    num_genespercell = hyperparams["num_genespercell"]
    
    cellEncoder = training.GraphEncoder(num_cellfeatures, hidden_dim)
    geneEncoder = training.SubgraphEncoder(hidden_dim, num_vertices = num_cells, num_subvertices = num_genespercell)
    multiviewGAE = training.MultiviewGAE(SubgraphEncoder = geneEncoder, GraphEncoder = cellEncoder)
    
    if hyperparams["optimizer"] == "adam":
        hyperparams["optimizer"] = torch.optim.Adam(multiviewGAE.parameters(), lr=0.01, weight_decay=5e-4),
    
    training.train_model(model=multiviewGAE, data=data, hyperparameters = hyperparams)


def main():
    args = parse_arguments()
    
    mode = args.mode
    input_dir_path = args.inputdirpath
    output_dir_path = args.outputdirpath
    num_nearestneighbors = args.nearestneighbors
    num_genespercell = args.numgenespercell
    LR_database = args.lrdatabase
    
    preprocess_output_path = os.path.join(output_dir_path, "preprocessing_output")
    training_output_path = os.path.join(output_dir_path, "training_output")
    evaluation_output_path = os.path.join(output_dir_path, "evaluation_output")

    
    if "preprocess" in mode:
        print("\n#------------------------------ Loading in data/arguments ----------------------------#\n")
        st_data = pd.read_csv(input_dir_path, index_col=None)
        assert {"Cell_ID", "X", "Y", "Cell_Type"}.issubset(set(st_data.columns.to_list()))
        
        numcells, totalnumgenes = st_data.shape[0], st_data.shape[1] - 4
        print(f"{numcells} Cells & {totalnumgenes} Total Genes\n")
        
        cespgrn_hyperparameters = {
            "bandwidth" : 0.1,
            "n_neigh" : 30,
            "lamb" : 0.1,
            "max_iters" : 1000
        }
        
        print(f"Hyperparameters:\n # of Nearest Neighbors: {num_nearestneighbors}\n # of Genes per Cell: {num_genespercell}\n")
        
        selected_st_data, selected_lrs = preprocessing.select_LRgenes(st_data, num_genespercell, LR_database)

        print("\n#------------------------------------ Preprocessing ----------------------------------#\n")
        
        if not os.path.exists(preprocess_output_path):
            os.mkdir(preprocess_output_path)

        # include X,Y in features??
        celllevel_features = st_data.drop(["Cell_ID", "Cell_Type"], axis = 1).values

        celllevel_adj, genelevel_graph, num2gene, gene2num, grns = preprocess(selected_st_data, num_nearestneighbors, cespgrn_hyperparameters)
            
        celllevel_edgelist = preprocessing.convert_adjacencylist2edgelist(celllevel_adj)
        genelevel_edgelist = nx.to_pandas_edgelist(genelevel_graph).drop(["weight"], axis=1).to_numpy().T
        
        assert celllevel_edgelist.shape == (2, celllevel_adj.shape[0] * celllevel_adj.shape[1])
        
        np.save(os.path.join(preprocess_output_path, "celllevel_adjacencylist.npy"),celllevel_adj)
        np.save(os.path.join(preprocess_output_path, "celllevel_edgelist.npy"),celllevel_edgelist)
        np.save(os.path.join(preprocess_output_path, "celllevel_features.npy"),celllevel_features)
        np.save(os.path.join(preprocess_output_path, "genelevel_edgelist.npy"),genelevel_edgelist)
        np.save(file = os.path.join(preprocess_output_path, "initial_grns.npy"), arr = grns) 
        print()
    

    if "train" in mode:
        print("\n#------------------------------ Creating PyG Datasets ----------------------------#\n")

        celllevel_data, genelevel_data = training.create_pyg_data(preprocess_output_path)
        console = Console()
        table = Table(show_header=True, header_style="bold")
        table.add_column("Cell Level PyG Data", style="cyan")
        table.add_column("Gene Level PyG Data", style="deep_pink3")
        table.add_row(str(celllevel_data), str(genelevel_data))
        console.print(table)
        
        if not os.path.exists(training_output_path):
            os.mkdir(training_output_path)


        print("\n#------------------------------- ClarifyGAE Training -----------------------------#\n")

        hyperparameters = {
            "num_genespercell":num_genespercell
            "concat_hidden_dim": 64,
            "optimizer" : "adam",
            "criterion" : torch.nn.MSELoss(),
            "num_epochs": 400
        }

        train_model((celllevel_data, genelevel_data), hyperparameters)
        

    return


if __name__ == "__main__":
    main()