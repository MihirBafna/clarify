import numpy as np
import pandas as pd
import networkx as nx

import preprocess as pre
import visualize as vis


def preprocess(st_data, numgenes_percell, out_dir):
    print("#------------------------------ Preprocessing ----------------------------#")
    
    numcells, totalnumgenes = st_data.shape[0], st_data.shape[1] - 4
    print(f"{numcells} Cells & {totalnumgenes} Genes")
    
    print("1. Inferring GRNs with CeSpGRN ...")
    
    grns = np.load("./grn/thetas_seqFISH_0.1_0.1_30.npy") # shape (ncells, ngenes, ngenes)


    print("2. Construct Cell-Level Graph from ST Data")

    
    celllevel_adj, edges = pre.construct_celllevel_graph(st_data, 5, get_edges=True)

    # nx.write_edgelist(gene_level_graph,"./out/gene_level_edge_list.txt",data=False)


    print("3. Construct Gene-Level Graph from ST Data + GRNs")
    
    gene_level_graph, num2gene, gen2num = pre.construct_genelevel_graph(grns, celllevel_adj, node_type="int")

    nx.write_edgelist(gene_level_graph,f"{out_dir}/gene_level_edge_list.txt",data=False)