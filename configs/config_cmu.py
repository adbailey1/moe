import glob
import os
from pathlib import Path
import sys
import torch

NUM_JITTER = 100
MODEL_SIZE = "large"
TOL = .5

FPS = 30
DUR = 1/6  #seconds 5.68 took 1.5 hours to do 1%
DUR_FPS = round(FPS * DUR)

DATASET_PATH = "/path/to/Datasets/"
FEATURE_LOC = "/path/to/features/"
SAVE_LOC = "/path/to/save/location"
WAV_PATH = glob.glob("/path/to/wav/files/"
                     "/CMU-MOSEI/Audio/Full/WAV_16000/*.wav")

AUDIO_LOC = "Audio/Full/WAV_16000"
SEGMENTED_LABELS = "Labels/CMU-MOSEI-Seg_Labels.csv"
SDK_LABELS = "Labels/text_audio_labels/TAfiles.pkl"
SDK_SCORES = "Labels/text_audio_labels/TAscores.pkl"
SDK_INTERVALS = "Labels/text_audio_labels/TAintervals.pkl"
SEGMENTED_AUDIO_LOCATION = os.path.join(DATASET_PATH, "Audio/Segmented")

VIDEO_LOC = "Videos/Segmented/Combined/"
VIDEO_FILES = glob.glob(os.path.join(DATASET_PATH, VIDEO_LOC, '*.mp4'))
IMAGE_DIR = os.path.join(DATASET_PATH, Path(VIDEO_LOC).parent, f"Frames{DUR_FPS}f")
ENCODING_DIR = os.path.join(DATASET_PATH, Path(VIDEO_LOC).parent, f"Encoding"
                                                             f"s{DUR_FPS}f")

# In range 0-3: 0=No evidence, 1=weak, 2=confident, 3=high confidence
# neutral should be all 0
EMOTION_LABELS_FROM_SDK = ["sentiment", "happy", "sad", "angry", "surprise",
                           "disgust", "fear"]

EMOTIONS_TO_USE = {
    -1: 'neutral',
    0: 'happy',
    1: 'sad',
    2: 'angry',
    # 3: "surprise",
    # 4: "disgust",
    # 5: "fear"
}


CLASS_DICT = {
    'neutral': torch.Tensor([0]),
    'happy': torch.Tensor([1]),
    'sad': torch.Tensor([2]),
    'angry': torch.Tensor([3]),
    # 'surprise': torch.Tensor([4]),
    # 'disgust': torch.Tensor([5]),
    # 'fear': torch.Tensor([6]),
}

LABEL_NUM = {
    'neutral': 0,
    'happy': 0,
    'sad': 0,
    'angry': 0,
    # 'surprise': 0,
    # 'disgust': 0,
    # 'fear': 0
}
EMO_THRESHOLD = .5  # choices 0.3, 0.5, or 1.0


def get_feature_file_name(USE_FOLDS, DATA_SPLIT, FEATURES_TO_USE):
    if not USE_FOLDS:
        if len(DATA_SPLIT) > 2:
            FEATURESFILENAME = f"features_{FEATURES_TO_USE}_" \
                               f"_train{DATA_SPLIT[0]}_val" \
                               f"{DATA_SPLIT[1]}_test{DATA_SPLIT[2]}_" \
                               f"{len(CLASS_DICT)}emoT{EMO_THRESHOLD}.pkl"
        elif len(DATA_SPLIT) == 2:
            FEATURESFILENAME = f"features_{FEATURES_TO_USE}_" \
                               f"_train{DATA_SPLIT[0]}_val" \
                               f"{DATA_SPLIT[1]}_{len(CLASS_DICT)}emoT{EMO_THRESHOLD}.pkl"
        elif len(DATA_SPLIT) == 1:
            FEATURESFILENAME = f"features_{FEATURES_TO_USE}_" \
                               f"_val{DATA_SPLIT[0]}_{len(CLASS_DICT)}emoT{EMO_THRESHOLD}.pkl"
        else:
            sys.exit("Improper setup detected: USE_FOLDS and DATA_SPLIT")
    else:
        FEATURESFILENAME = f"features_{FEATURES_TO_USE}.pkl"

    return FEATURESFILENAME
