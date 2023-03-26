# $\color{teal}{\textsf{CLARIFY}}$
$\textsf{Using Graph Autoencoders (GAEs) to clarify:}$

$\textsf{(1.) {\color{teal}{c}}ell cel}\textsf{{\color{teal}{l}} interactions } \textsf{{\color{teal}{a}}nd} \textsf{ (2.) gene {\color{teal}{r}}egulatory network } \textsf{{\color{teal}{i}}nference } \textsf{{\color{teal}{f}}rom } \textsf{spatiall{\color{teal}{y}} resolved transcriptomics.}$


## Installation & Setup

Make sure to clone this repository along with the submodules as follows:

```
git clone --recurse-submodules https://github.com/MihirBafna/clarify.git
cd clarify
```
To install dependencies with conda (recommended), use either the requirements.txt or the environment.yml. Some manual installation (via pip) will also be required.

## Data
Three datasets were utilized for evaluation:

1. seqFISH profile of mouse visual cortex [(Zhu _et al_., 2018)](https://www.nature.com/articles/nbt.4260)
2. MERFISH profile of mouse hypothalamic preoptic region [(Moffitt _et al_., 2018)](https://www.science.org/doi/10.1126/science.aau5324)
3. scMultiSim simulated dataset [(Li _et al_., 2022)](https://www.biorxiv.org/content/10.1101/2022.10.15.512320v1)

All of the preprocessed data are organized into pandas dataframes and are located in the ./data/ folder. These dataframes can be used directly as input to Clarify.

## Demos & Results

To visualize results (from pretrained Clarify models) and play around with demos, use the following interactive notebooks:
1. [preprocessing.ipynb](preprocessing.ipynb)
2. [evaluation.ipynb](evaluation.ipynb)

## Run Clarify

To run Clarify, run main.py and configure parameters based on their definitions below:

```
usage: main.py [-h] [-m MODE] [-i INPUTDIRPATH] [-o OUTPUTDIRPATH] [-s STUDYNAME] [-t SPLIT] 
               [-n NUMGENESPERCELL] [-k NEARESTNEIGHBORS] [-l LRDATABASE] [--fp FP] [--fn FN] [-a OWNADJACENCYPATH]
```
The first row of parameters are necessary

*  `-m MODE, --mode MODE`  clarify mode: preprocess,train (pick one or both separated by a comma)
*  `-i INPUTDIRPATH, --inputdirpath` Input directory path where ST dataframe is stored
*  `-o OUTPUTDIRPATH, --outputdirpath` Output directory path where results will be stored
*  `-s STUDYNAME, --studyname` clarify study name to act as identifier for outputs
*  `-t SPLIT, --split` ratio of test edges [0,1)

This second row of parameters have defaults set and are not needed.

*  `-n NUMGENESPERCELL, --numgenespercell` Number of genes in each gene regulatory network (default 45)
*  `-k NEARESTNEIGHBORS, --nearestneighbors` Number of nearest neighbors for each cell (default 5)
*  `-l LRDATABASE, --lrdatabase` 0/1/2 for which Ligand-Receptor Database to use (default 0 corresponds to mouse DB)
*  `--fp FP`               (experimentation only) add # of fake edges to train set [0,1)
*  `--fn FN`               (experimentation only) remove # of real edges from train set [0,1)
*  `-a OWNADJACENCYPATH, --ownadjacencypath` Using your own cell level adjacency (give path)

