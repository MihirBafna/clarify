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

To visualize results (from pretrained Clarify models) and play around with demos, use the preprocessing.ipynb and evaluation.ipynb notebooks.

## Run Clarify

