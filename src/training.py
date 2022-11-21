import torch
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
import os
import numpy as np



def create_pyg_data(preprocessing_output_folderpath):
    '''
    Construct torch_geometric.Data objects for cell level and gene level graphs
    
    :preprocessing_output_folderpath: str : nd.array of cell-specific GRNs (output of CeSpGRN)
    
    :return: torch_geometric.Data for cell level and torch_geometric.HeteroData for gene level
    '''
    if not os.path.exists(preprocessing_output_folderpath) or \
        not {"celllevel_adjacencylist.npy", "celllevel_edgelist.npy", "genelevel_edgelist.npy", "celllevel_features.npy"}.issubset(set(os.listdir(preprocessing_output_folderpath))):
        
        raise Exception("Data has not been preprocessed yet.")
    
    celllevel_adjacencylist = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_adjacencylist.npy")))
    celllevel_features = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_features.npy")))
    celllevel_edgelist = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_edgelist.npy")))
    genelevel_edgelist = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "genelevel_edgelist.npy")))


    cell_level_data = Data(x=celllevel_features, edge_index = celllevel_edgelist, y = celllevel_adjacencylist)
    gene_level_data = HeteroData(edge_index = genelevel_edgelist)
    
    return cell_level_data, gene_level_data


def train(data, model, hyperparameters):
  wandb.config = hyperparameters
  num_epochs = hyperparameters["num_epochs"]
  optimizer = hyperparameters["optimizer"]
  criterion = hyperparameters["criterion"]

  with trange(num_epochs,desc="") as pbar:
    for epoch in pbar:
      pbar.set_description(f"Epoch {epoch}")
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      recon_A = model(data[0].x,data[1].x, data[0].edge_index, data[1].edge_index)
      preds = (recon_A.detach().numpy() > 0.5).astype(int)
      accuracy = (preds == data[1].y.detach().numpy()).astype(int).sum().item()/ len(data[1].y.flatten())
      auroc = roc_auc_score(data[1].y.detach().numpy(),recon_A.detach().numpy())

      loss = criterion(recon_A, data[1].y.float())
      wandb.log({'Train Accuracy': accuracy, 'Loss': loss.item(), "AUROC":auroc})

      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.

      pbar.set_postfix(loss=loss.item(), accuracy=100. * accuracy)