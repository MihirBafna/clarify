import torch
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
import os
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import normalize
import numpy as np
# import wandb
from tqdm import trange



def create_pyg_data(preprocessing_output_folderpath):
    '''
    Construct torch_geometric.Data objects for cell level and gene level graphs
    
    :preprocessing_output_folderpath: str : nd.array of cell-specific GRNs (output of CeSpGRN)
    
    :return: torch_geometric.Data for cell level and torch_geometric.HeteroData for gene level
    '''
    if not os.path.exists(preprocessing_output_folderpath) or \
        not {"celllevel_adjacencylist.npy","celllevel_adjacencymatrix.npy",  "celllevel_edgelist.npy", "genelevel_edgelist.npy", "celllevel_features.npy", "genelevel_features.npy"}.issubset(set(os.listdir(preprocessing_output_folderpath))):
        
        raise Exception("Proper preprocessing files not found. Please run the 'preprocessing' step.")
    
    # celllevel_adjacencylist = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_adjacencylist.npy"))).type(torch.LongTensor)
    celllevel_adjacencymatrix = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_adjacencymatrix.npy"))).type(torch.LongTensor)
    celllevel_features = torch.from_numpy(normalize(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_features.npy")))).type(torch.float32)
    celllevel_edgelist = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_edgelist.npy"))).type(torch.LongTensor)
    genelevel_edgelist = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "genelevel_edgelist.npy"))).type(torch.LongTensor)
    genelevel_features = torch.from_numpy(normalize(np.load(os.path.join(preprocessing_output_folderpath, "genelevel_features.npy")))).type(torch.float32)


    cell_level_data = Data(x=celllevel_features, edge_index = celllevel_edgelist, y = celllevel_adjacencymatrix)
    gene_level_data = Data(x= genelevel_features, edge_index = genelevel_edgelist)

    return cell_level_data, gene_level_data
  


def train(data, model, hyperparameters):
  # wandb.config = hyperparameters
  num_epochs = hyperparameters["num_epochs"]
  optimizer = hyperparameters["optimizer"][0]
  criterion = hyperparameters["criterion"]

  with trange(num_epochs,desc="") as pbar:
    for epoch in pbar:
      pbar.set_description(f"Epoch {epoch}")
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      recon_Ac, recon_Ag = model(data[0].x,data[1].x, data[0].edge_index, data[1].edge_index)
      # preds = (recon_Ac.detach().numpy() > 0.5).astype(int)
      # print(preds.flatten() == data[0].y.detach().numpy().flatten())
      # accuracy = (preds.flatten == data[0].y.detach().numpy()).astype(int).sum().item()/ len(data[0].y.flatten())
      # auroc = roc_auc_score(data[1].y.detach().numpy(),recon_A.detach().numpy())

      loss = criterion(recon_Ac, data[0].y.float())
      # wandb.log({'Train Accuracy': accuracy, 'Loss': loss.item(), "AUROC":auroc})

      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.

      pbar.set_postfix(loss=loss.item())
      
      
def train_gae(data, model, hyperparameters):
  # wandb.config = hyperparameters
  num_epochs = hyperparameters["num_epochs"]
  optimizer = hyperparameters["optimizer"][0]
  criterion = hyperparameters["criterion"]

  with trange(num_epochs,desc="") as pbar:
    for epoch in pbar:
      pbar.set_description(f"Epoch {epoch}")
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      z = model.encode(data[0].x,data[1].x, data[0].edge_index, data[1].edge_index)
      recon_Ac = torch.sigmoid(torch.matmul(z, z.t())).detach().numpy()
      
      loss = model.recon_loss(z, data[0].edge_index)
      
      auc, ap = roc_auc_score(data[0].y.detach().numpy().flatten(),recon_Ac.flatten() ), average_precision_score(data[0].y.detach().numpy().flatten(),recon_Ac.flatten() )


      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.

      pbar.set_postfix(loss=loss.item(), auc=auc.item(), ap =ap.item())