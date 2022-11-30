# MoE
# 1. Datasets:
## 1.1 IEMOCAP:

Obtain permission to access the dataset from: 

<https://sail.usc.edu/iemocap/iemocap_release.htm>

## 1.2 RAVDESS:

Download with: 

    wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1

Extract and save in a location, such as RAVDESS/speech/

# 2. Setup conda environment
All work has been conducted using Python 3.8 on Ubuntu 18.04.

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh


To create the conda environment: 

    conda env create -f environment.yml

To activate the conda environment:

    conda activate iemocap


# 3. Running the Baseline (Optional)
To run the MACNN code from Xu et al. give permisions to run_original.sh file:

    chmod +x run_original.sh

The code is contained in moe/original_run. This code has been altered to 
run out of the box, for a full list of changes, please refer to 
moe/original_run/changelist.txt


# 4. Run
