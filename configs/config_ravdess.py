import os

import glob

import torch

DATASET_PATH = "/path/to/Datasets/"
FEATURE_LOC = "/path/to/features/"
SAVE_LOC = "/path/to/save/location"
WAV_PATH = glob.glob(os.path.join(DATASET_PATH, "*/*.wav"))

overlapTime = {
    'neutral': 1,
    'happy': 1,
    'sad': 1,
    'angry': 1,
}

EMOTIONS_TO_USE = {
    '01': 'neutral',
    # '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    # '06': 'fearful',
    # '07': 'disgust',
    # '08': 'surprised'
}

CLASS_DICT = {
    'neutral': torch.Tensor([0]),
    'happy': torch.Tensor([1]),
    'sad': torch.Tensor([2]),
    'angry': torch.Tensor([3]),
}
LABEL_NUM = {
    'neutral': 0,
    'happy': 0,
    'sad': 0,
    'angry': 0,
}

if len(EMOTIONS_TO_USE) == 8:
    CLASS_DICT["calm"] = torch.Tensor([4])
    CLASS_DICT["fearful"] = torch.Tensor([5])
    CLASS_DICT["disgust"] = torch.Tensor([6])
    CLASS_DICT["surprised"] = torch.Tensor([7])
    LABEL_NUM["calm"] = 0
    LABEL_NUM["fearful"] = 0
    LABEL_NUM["disgust"] = 0
    LABEL_NUM["surprised"] = 0
    overlapTime["calm"] = 1
    overlapTime["fearful"] = 1
    overlapTime["disgust"] = 1
    overlapTime["surprised"] = 1


# NAMING_CONVENTION = Modality, Vocal Channel, Emotion, Intensity, Statement,
# Repetition, Actor
def get_num_speakers():
    NUM_SESSIONS = 24
    return NUM_SESSIONS


def get_feature_file_name(FEATURES_TO_USE):

    FEATURESFILENAME = f"features_{FEATURES_TO_USE}.pkl"

    return FEATURESFILENAME

EMO_THRESHOLD = None
