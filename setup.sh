#!/bin/bash

git clone https://github.com/vaventt/smart-image-augmentation.git

# Download the Anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

# Install Anaconda
bash Anaconda3-2022.05-Linux-x86_64.sh -b

# Activate the Conda environment
source ~/anaconda3/bin/activate

# Initialize Conda
conda init

conda create -n da-fusion python=3.8 pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge

conda activate da-fusion

pip install diffusers["torch"] transformers pycocotools pandas matplotlib seaborn scipy

pip install -e ./da-fusion

huggingface-cli login
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

