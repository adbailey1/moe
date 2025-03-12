import os
import torch.nn as nn


MIXTUREOFEXPERTS = False
FUSION_LEVEL = 0  # 0 to not use: 1/2/3/4/5/6/-7/-6/-5/-4/-3/-2 -1==4 x params
SKIP_FINAL_FC = True
FC_BIAS = True
SAVE_IND_EXPERTS = False
LIKE_IEMOCAP = False
ACTIVATION = nn.Sigmoid()

MODEL_NAME = "test"
MODEL_TYPE = "MACNN"  # MACNN / LightSERNet
DATASET = "iemocap"  # iemocap / cmu / ravdess
NUM_FOLDS = 5

BATCH_SIZE = 32
EPOCHS = 50
SEEDS = [111111, 123456, 0, 999999, 987654]
USE_WEIGHTS_FOR_LOSS = False
LEARNING_RATE = 1e-3 if MODEL_TYPE == "MACNN" else 1e-4
WEIGHT_DECAY = 1e-6 if MODEL_TYPE == "MACNN" else 0

T_STRIDE = 2
T_OVERLAP = T_STRIDE / 2
VAL_OVERLAP = 1.6

SPEAKER_IND = False
RUN_INFERENCE = "Validation"  # "Validation" or "Test"
SKIP_TRAIN = False
SHOW_T_SNE = False  # For skip training only

ALPHA_MAIN = 1

USE_LOSS_TO_DETERMINE_BEST_EPOCH = False
REPEAT_EXP = len(SEEDS)

attention_head = 4
attention_hidden = 64
acc_type = "average_no_softmax"  # "average_no_softmax" or "average_softmax" or
                                 # "majority_vote"

overlapTime = {
    'neutral': 1,
    'happy': 1,
    'sad': 1,
    'angry': 1,
}

if DATASET == "iemocap":
    from configs import config_iemocap as config_to_use
elif DATASET == "cmu":
    from configs import config_cmu as config_to_use
    SEGMENTED_AUDIO_LOCATION = config_to_use.SEGMENTED_AUDIO_LOCATION
    SEGMENTED_LABELS = config_to_use.SEGMENTED_LABELS
elif DATASET == "ravdess":
    from configs import config_ravdess as config_to_use

DATASET_PATH = config_to_use.DATASET_PATH
FEATURE_LOC = config_to_use.FEATURE_LOC
SAVE_LOC = config_to_use.SAVE_LOC
WAV_PATH = config_to_use.WAV_PATH
EXP_DIR = os.path.join(SAVE_LOC, MODEL_NAME)

RATE = 16000

CLASS_DICT = config_to_use.CLASS_DICT
LABEL_NUM = config_to_use.LABEL_NUM
EMOTIONS_TO_USE = config_to_use.EMOTIONS_TO_USE

FEATURES_TO_USE = 'mfcc'

if DATASET == "iemocap":
    NUM_SESSIONS = config_to_use.get_num_speakers(SPEAKER_IND)
    TOLERANCES = config_to_use.TOLERANCES
    impro_or_script = config_to_use.impro_or_script
elif DATASET == "ravdess":
    NUM_SESSIONS = config_to_use.get_num_speakers()

FEATURESFILENAME = config_to_use.get_feature_file_name(FEATURES_TO_USE)

EMO_THRESHOLD = config_to_use.EMO_THRESHOLD
