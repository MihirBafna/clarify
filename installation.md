# Clarify Installation

First, create a basic conda environment with python 3.8.2
```
conda create --name clarify python=3.8.2
```
Now, activate your environment and utilize the requirements.txt file to install non pytorch dependencies
```
conda activate clarify
pip install -r requirements.txt
```
To install pytorch dependencies, we will use the following pip wheels
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
Finally, we will install the pytorch-geometric dependencies.
```
pip install torch-geometric==2.1.0
pip install torch-cluster==1.6.0 torch-scatter==2.0.9 torch-sparse==0.6.15 torch-spline-conv==1.2.1
```
