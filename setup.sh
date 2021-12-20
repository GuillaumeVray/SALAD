#!/bin/bash

# Download the data, extract it and move it to the data folder
echo "Step 1/2: Dowloading data and Preparing folders"
pip install gdown
mkdir data
mkdir log
cd data
gdown https://drive.google.com/u/4/uc?id=1LmonDb_gNAbbsnyLCsj40rUxbwZZ1ZSp
tar -xvf author.tar.xz

# Download dependencies
echo "Step 2/2: Setuping conda environment"
cd ..
conda env create -f environment.yml

echo "Setup Done !"
