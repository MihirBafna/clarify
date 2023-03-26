import numpy as np
import pandas as pd
import networkx as nx
import os
import argparse
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich.spinner import Spinner

import torch

import time

import preprocessing as preprocessing
import training as training
import models as models
import visualize as vis

from torch_geometric.nn import GAE

debug = False


def parse_arguments():
    parser = argparse.ArgumentParser(description='clarifyGAE arguments')
    parser.add_argument("-m", "--mode", type=str, default = "train",
        help="clarifyGAE mode: preprocess,train,test")
    parser.add_argument("-i", "--inputdirpath", type=str,
                    help="Input directory path where ST data is stored")
    parser.add_argument("-o", "--outputdirpath", type=str,
                    help="Output directory path where results will be stored ")
    parser.add_argument("-s", "--studyname", type=str,
                     help="clarifyGAE study name")
    parser.add_argument("-t", "--split", type=float,
        help="# of test edges [0,1)")
    parser.add_argument("-n", "--numgenespercell", type=int, default = 45,
                help="Number of genes in each gene regulatory network")
    parser.add_argument("-k", "--nearestneighbors", type=int, default = 5,
                help="Number of nearest neighbors for each cell")
    parser.add_argument("-l", "--lrdatabase", type=int, default=0,
                    help="0/1/2 for which Ligand-Receptor Database to use")
    parser.add_argument("--fp", type=float, default=0,
        help="(experimentation only) # of false positive test edges [0,1)")
    parser.add_argument("--fn", type=float, default=0,
        help="(experimentation only) # of false negative test edges [0,1)")
    parser.add_argument("-a", "--ownadjacencypath", type=str,
        help="Using your own cell level adjacency (give path)")
    args = parser.parse_args()
    return args



def preprocess(st_data, num_nearestneighbors, lrgene_ids, cespgrn_hyperparameters, ownadjacencypath = None):    
        
    # 1. Infer initial GRNs with CeSpGRN
    if debug: # skip inferring GRNs while debugging
        print("1. Skipping CeSpGRN inference (for debug mode)")
        grns = np.load("../out/scmultisim_final/1_preprocessing_output/initial_grns.npy")
    else:
        grns = preprocessing.infer_initial_grns(st_data, cespgrn_hyperparameters) # shape (ncells, ngenes, ngenes)

    # 2. Construct Cell-Level Graph from ST Data
    if ownadjacencypath is not None:
        celllevel_adj = np.load(ownadjacencypath)
    else:
        celllevel_adj, _ = preprocessing.construct_celllevel_graph(st_data, num_nearestneighbors, get_edges=False)

    #  3. Construct Gene-Level Graph from ST Data + GRNs 
    gene_level_graph, num2gene, gene2num, grn_components = preprocessing.construct_genelevel_graph(grns, celllevel_adj, node_type="int", lrgenes = lrgene_ids)
    
    # 4. Generate Gene Feature vectors
    # if debug:
    #     gene_features, genefeaturemodel = None,None
    # else:
    #     gene_features, genefeaturemodel = preprocessing.get_gene_features(grn_components, type="node2vec")
    gene_features, genefeaturemodel = preprocessing.get_gene_features(grn_components, type="node2vec")
    

    return celllevel_adj, gene_level_graph, num2gene, gene2num, grns, gene_features, genefeaturemodel



def build_clarifyGAE(data, hyperparams = None):
    num_cells, num_cellfeatures = data[0].x.shape[0], data[0].x.shape[1]
    num_genes, num_genefeatures = data[1].x.shape[0], data[1].x.shape[1]
    hidden_dim = hyperparams["concat_hidden_dim"] // 2
    num_genespercell = hyperparams["num_genespercell"]

    cellEncoder = models.GraphEncoder(num_cellfeatures, hidden_dim)
    geneEncoder = models.SubgraphEncoder(num_features=num_genefeatures, hidden_dim=hidden_dim, num_vertices = num_cells, num_subvertices = num_genespercell)
    
    multiviewGAE = models.MultiviewGAE(SubgraphEncoder = geneEncoder, GraphEncoder = cellEncoder)
    
    if hyperparams["optimizer"] == "adam":
        hyperparams["optimizer"] = torch.optim.Adam(multiviewGAE.parameters(), lr=0.01, weight_decay = 5e-4),    # optionally add weight_decay = 5e-4
    
    return training.train(model=multiviewGAE, data=data, hyperparameters = hyperparams)



def build_clarifyGAE_pytorch(data, hyperparams = None):
    num_cells, num_cellfeatures = data[0][0].x.shape[0], data[0][0].x.shape[1]
    num_genes, num_genefeatures = data[1].x.shape[0], data[1].x.shape[1]
    hidden_dim = hyperparams["concat_hidden_dim"] // 2
    num_genespercell = hyperparams["num_genespercell"]

    cellEncoder = models.GraphEncoder(num_cellfeatures, hidden_dim)
    geneEncoder = models.SubgraphEncoder(num_features=num_genefeatures, hidden_dim=hidden_dim, num_vertices = num_cells, num_subvertices = num_genespercell)
    
    multiviewEncoder = models.MultiviewEncoder(SubgraphEncoder = geneEncoder, GraphEncoder = cellEncoder)
    gae = GAE(multiviewEncoder)

    return gae



def main():
    
    console = Console()
    text = Text.from_ansi(r"""
    --------------------------------------------------------------------------------
     .d8888b.   888              d8888  8888888b.   8888888  8888888888  Y88b   d88P 
    d88P  Y88b  888             d88888  888   Y88b    888    888          Y88b d88P  
    888    888  888            d88P888  888    888    888    888           Y88o88P   
    888         888           d88P 888  888   d88P    888    8888888        Y888P    
    888         888          d88P  888  8888888P"     888    888             888     
    888    888  888         d88P   888  888 T88b      888    888             888     
    Y88b  d88P  888        d8888888888  888  T88b     888    888             888     
      88888P    88888888  d88P     888  888   T88b  8888888  888             888
    --------------------------------------------------------------------------------
        CCI and GRN refinement with spatially resolved transcriptomics and GAEs
                    """)
    text.stylize("cyan")
    console.print(text)
    
    with console.status("[cyan] Clarify booting up ...") as status:
        status.update(spinner="aesthetic", spinner_style="cyan")
        time.sleep(4)
        status.update(status="[cyan] Parsing arguments ...")
        args = parse_arguments()
    
        mode = args.mode
        input_dir_path = args.inputdirpath
        output_dir_path = args.outputdirpath
        num_nearestneighbors = args.nearestneighbors
        num_genespercell = args.numgenespercell
        LR_database = args.lrdatabase
        studyname = args.studyname
        ownadjacencypath = args.ownadjacencypath
        
        preprocess_output_path = os.path.join(output_dir_path, "1_preprocessing_output")
        training_output_path = os.path.join(output_dir_path, "2_training_output")
        evaluation_output_path = os.path.join(output_dir_path, "3_evaluation_output")
        time.sleep(4)
        status.update(status="[cyan] Done")
        time.sleep(2)

        
    if "preprocess" in mode:
        start_time = time.time()
        print("\n#------------------------------------ Loading in data ---------------------------------#\n")
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
        
        selected_st_data, lrgene2id = preprocessing.select_LRgenes(st_data, num_genespercell, LR_database)

        print("\n#------------------------------------ Preprocessing ----------------------------------#\n")
        
        if not os.path.exists(preprocess_output_path):
            os.mkdir(preprocess_output_path)

        # include X,Y in features??
        celllevel_features = st_data.drop(["Cell_ID", "Cell_Type", "X", "Y"], axis = 1).values

        celllevel_adj, genelevel_graph, num2gene, gene2num, grns, genelevel_features, genelevel_feature_model = preprocess(selected_st_data, num_nearestneighbors,lrgene2id.values(), cespgrn_hyperparameters, ownadjacencypath)
            
        celllevel_edgelist = preprocessing.convert_adjacencylist2edgelist(celllevel_adj)
        genelevel_edgelist = nx.to_pandas_edgelist(genelevel_graph).drop(["weight"], axis=1).to_numpy().T
        genelevel_adjmatrix = nx.adjacency_matrix(genelevel_graph, weight=None)
        
        assert celllevel_edgelist.shape == (2, celllevel_adj.shape[0] * celllevel_adj.shape[1])
        
        np.save(os.path.join(preprocess_output_path, "celllevel_adjacencylist.npy"),celllevel_adj)
        np.save(os.path.join(preprocess_output_path, "celllevel_adjacencymatrix.npy"),preprocessing.convert_adjacencylist2adjacencymatrix(celllevel_adj))
        np.save(os.path.join(preprocess_output_path, "celllevel_edgelist.npy"),celllevel_edgelist)
        np.save(os.path.join(preprocess_output_path, "celllevel_features.npy"),celllevel_features)
        np.save(os.path.join(preprocess_output_path, "genelevel_edgelist.npy"),genelevel_edgelist)
        np.save(os.path.join(preprocess_output_path, "genelevel_adjmatrix.npy"),genelevel_adjmatrix)
        np.save(file = os.path.join(preprocess_output_path, "initial_grns.npy"), arr = grns) 
        
        # if not debug:
        np.save(os.path.join(preprocess_output_path, "genelevel_features.npy"), genelevel_features) 
        genelevel_feature_model.save(os.path.join(preprocess_output_path, "genelevel_feature_model")) 

        print(f"Finished preprocessing in {(time.time() - start_time)/60} mins.\n")
    

    if "train" in mode:
        hyperparameters = {
            "num_genespercell": num_genespercell,
            "concat_hidden_dim": 64,
            "optimizer" : "adam",
            "criterion" : torch.nn.BCELoss(),
            "num_epochs": 120,
            "split": args.split,
        }
        
        # add false edges only for experimentation
        false_edges = None if args.fp == 0 and args.fn == 0 else {"fp": args.fp, "fn": args.fn}
        
        
        print("\n#------------------------------ Creating PyG Datasets ----------------------------#\n")

        celllevel_data, genelevel_data = training.create_pyg_data(preprocess_output_path, hyperparameters["split"],false_edges)
        console = Console()
        table = Table(show_header=True, header_style="bold")
        table.add_column("Cell Level PyG Data", style="cyan")
        table.add_column("Gene Level PyG Data", style="deep_pink3")
        
        celllevel_str = str(celllevel_data)

        table.add_row(celllevel_str, "".join(str(genelevel_data).split("\n")))
        console.print(table)
        
        if not os.path.exists(training_output_path):
            os.mkdir(training_output_path)


        print("\n#------------------------------- ClarifyGAE Training -----------------------------#\n")

        data = (celllevel_data, genelevel_data)
        
        # train_clarifyGAE(data, hyperparameters)
        model = build_clarifyGAE_pytorch(data, hyperparameters)
        
        if hyperparameters["optimizer"] == "adam":
            hyperparameters["optimizer"] = torch.optim.Adam(model.parameters(), lr=0.01), 
            
        split = hyperparameters["split"]
    
        trained_model, metrics_df = training.train_gae(model=model, data=data, hyperparameters = hyperparameters)
        
        torch.save(trained_model.state_dict(), os.path.join(training_output_path,f'{studyname}_trained_gae_model.pth'))
        
        metrics_df.to_csv(os.path.join(evaluation_output_path, f"{studyname}_metrics_{split}.csv"))

    return


if __name__ == "__main__":
    main()