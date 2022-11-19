import numpy as np
import pandas as pd
import networkx as nx
import os
import sys
import argparse


import preprocess as pre
import visualize as vis


def parse_arguments():
    parser = argparse.ArgumentParser(description='clarifyGAE arguments')
    parser.add_argument("-n", "--numgenespercell", type=int, default = 45,
            help="Number of genes in each gene regulatory network")
    parser.add_argument("-k", "--nearestneighbors", type=int, default = 5,
                help="Number of nearest neighbors for each cell")
    parser.add_argument("-i", "--inputdirpath", type=str,
                    help="Input directory path where ST data is stored")
    parser.add_argument("-o", "--outputdirpath", type=str,
                    help="Output directory path where results will be stored ")

    args = parser.parse_args()
    return args


def preprocess(st_data, numgenes_percell, numnearestneighbors=5):    
    numcells, totalnumgenes = st_data.shape[0], st_data.shape[1] - 4
    
    print("1. Inferring GRNs with CeSpGRN ...")
    grns = np.load("../out/seqfish_grns.npy") # shape (ncells, ngenes, ngenes)

    #print("2. Construct Cell-Level Graph from ST Data")
    celllevel_adj, edge_coordinates = pre.construct_celllevel_graph(st_data, numnearestneighbors, get_edges=True)

    #print("3. Construct Gene-Level Graph from ST Data + GRNs")
    gene_level_graph, num2gene, gene2num = pre.construct_genelevel_graph(grns, celllevel_adj, node_type="int")

    return celllevel_adj, gene_level_graph, num2gene, gene2num


def main():
    print("\n#------------------------ Loading in data/arguments ----------------------#\n")

    args = parse_arguments()
    
    st_data = pd.read_csv(args.inputdirpath, index_col=None)
    num_nearestneighbors = args.nearestneighbors
    num_genespercell = args.numgenespercell
    
    assert {"Cell_ID", "X", "Y", "Cell_Type"}.issubset(set(st_data.columns.to_list()))
    
    numcells, totalnumgenes = st_data.shape[0], st_data.shape[1] - 4
    print(st_data.head())
    print(f"{numcells} Cells & {totalnumgenes} Total Genes")
    print(f"\nHyperparameters:\n # of Nearest Neighbors: {num_nearestneighbors}\n # of Genes per Cell: {num_genespercell}")
    
    print("\n#------------------------------ Preprocessing ----------------------------#\n")

    celllevel_adj, genelevel_graph, num2gene, gene2num = preprocess(st_data,num_genespercell, num_nearestneighbors)
    
    nx.write_edgelist(genelevel_graph,"../out/gene_level_edge_list.txt",data=False)

if __name__ == "__main__":
    main()