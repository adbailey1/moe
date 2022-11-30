import os


MoE = False
FUSION_LEVEL = 0  # 0 to not use: 1/2/3/4/5/6/-7/-6/-5/-4/-3/-2
CLASS_WEIGHTS = False

MODEL_NAME = "modelName"
MODEL_TYPE = "MACNN"  # MACNN or MACNN_x4
DATASET = "iemocap"  # iemocap / ravdess
NUM_FOLDS = 5

BATCH_SIZE = 32
EPOCHS = 50
SEEDS = [111111, 123456, 0, 999999, 987654]
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6

CLIP_LENGTH = 2
TRAINING_OVERLAP = CLIP_LENGTH / 2
VALIDATION_OVERLAP = 1.6

RUN_INFERENCE = "Validation"  # "Validation" or "Test"
SKIP_TRAIN = False  # To run on validation/test set only with a trained model
ATTENTION_HEADS = 4
ATTENTION_HIDDEN = 64

if DATASET == "iemocap":
    from configs import config_iemocap as config_to_use
    SESSION_TYPE = config_to_use.SESSION_TYPE
    NUM_SESSIONS = 10
elif DATASET == "ravdess":
    from configs import config_ravdess as config_to_use
    NUM_SESSIONS = 12

DATASET_PATH = config_to_use.DATASET_PATH
FEATURE_LOC = config_to_use.FEATURE_LOC
SAVE_LOC = config_to_use.SAVE_LOC
WAV_PATH = config_to_use.WAV_PATH
EXP_DIR = os.path.join(SAVE_LOC, MODEL_NAME)

CLASS_DICT = config_to_use.CLASS_DICT
CLASS_DICT_IDX = config_to_use.CLASS_DICT_IDX
EMOTIONS_TO_USE = config_to_use.EMOTIONS_TO_USE

FEATURES_TO_USE = 'mfcc'

RATE = 16000
N_MFCC = 26
WINDOW_SIZE = 2048
HOP_SIZE = 512

FOLD_FILENAME = config_to_use.get_feature_file_name(FEATURES_TO_USE)
