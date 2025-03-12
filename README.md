# MoE
# 1. Datasets:
## 1.1 IEMOCAP:

Obtain permission to access the dataset from: 

<https://sail.usc.edu/iemocap/iemocap_release.htm>

## 1.2 RAVDESS:

Download with: 

    wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1

Extract and save in a location, such as RAVDESS/speech/

## 1.3 CMU-MOSEI:

Accessible via:

https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK 


# 2. Setup conda environment
All work has been conducted using Python 3.8 on Ubuntu 18.04.

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh


To create the conda environment: 

    conda env create -f environment.yml

To activate the conda environment:

    conda activate iemocap

# 3. For IEMOCAP: Run handleIEMOCAP.py 

# 4. Running the Baseline (Optional)
To run the MACNN code from Xu et al. give permisions to run_original.sh file:

    chmod +x run_original.sh

The code is contained in moe/original_run. This code has been altered to 
run out of the box, for a full list of changes, please refer to 
moe/original_run/changelist.txt


# 5. Edit config files
Edit the dataset config files under configs/config_iemocap.py etc.

    FEATURE_LOC = "/Path/to/IEMOCAP/features"
    SAVE_LOC = "/Path/to/exp/save/loc"
    DATASET_PATH = "/Path/to/IEMOCAP/Dataset"
    WAV_PATH = glob.glob(os.path.join(DATASET_PATH,
                        "path/to/wav/files/*.wav"))

Edit the experiment config file: configs/config.py
Baseline models are taken from 

MACNN: https://github.com/lessonxmk/head_fusion

    Xu, M., Zhang, F., & Khan, S. U. (2020). Improve accuracy of speech emotion
    recognition with attention head fusion. 2020 10th Annual Computing
    and Communication Workshop and Conference (CCWC), 1058–1064.

and Light-SERNet: https://github.com/AryaAftab/LIGHT-SERNET

    Aftab, A., Morsali, A., Ghaemmaghami, S., & Champagne, B. (2022). LIGHT-
    SERNET: A lightweight fully convolutional neural network for speech
    emotion recognition. ICASSP 2022 - 2022 IEEE International Conference
    on Acoustics, Speech and Signal Processing (ICASSP), 6912–6916.

Config:

    import os
    import torch.nn as nn

    MIXTUREOFEXPERTS = False
    FUSION_LEVEL = 0  # 0 to not use: 1/2/3/4/5/6/-7/-6/-5/-4/-3/-2 -1==4 x params
    SKIP_FINAL_FC = True
    FC_BIAS = True
    SAVE_IND_EXPERTS = False
    LIKE_IEMOCAP = True
    ACTIVATION = nn.Sigmoid()
    
    MODEL_NAME = "test"
    MODEL_TYPE = "MACNN"  # MACNN / LightSERNet
    DATASET = "iemocap"  # iemocap / cmu / ravdess
    NUM_FOLDS = 5
    
    BATCH_SIZE = 32
    EPOCHS = 50
    SEEDS = [111111, 123456, 0, 999999, 987654]

# 6. Run the experiment

    ./run.sh

# 7. Mixture of Experts

The advances from this work are the mixture of experts models and the 
fusion networks. To run these, set:
    
    MIXTUREOFEXPERTS = True

And set fusion level to 1-6 or -7 - -2

    FUSION_LEVEL = 0  # 0 to not use: 1/2/3/4/5/6/-7/-6/-5/-4/-3/-2 -1==4 x params
    