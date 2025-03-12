import os

import torch
import glob

DATASET_PATH = "/path/to/Datasets/"
FEATURE_LOC = "/path/to/features/"
SAVE_LOC = "/path/to/save/location"
WAV_PATH = glob.glob(
    os.path.join(DATASET_PATH,
                 "IEMOCAP_full_release/*/sentences/wav/*/S*.wav"))


def get_num_speakers(speaker_ind):
    if speaker_ind:
        NUM_SPEAKERS = 5
    elif not speaker_ind:
        NUM_SPEAKERS = 10
    return NUM_SPEAKERS

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

EMOTIONS_TO_USE = {
    '01': 'neutral',
    # '02': 'frustration',
    # '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    # '06': 'fearful',
    '07': 'happy',  # excitement->happy
    # '08': 'surprised'
}

TOLERANCES = {"angry": {"v": [1, 3],
                        "a": [4, 5]},
              "happy": {"v": [4, 5],
                        "a": [3, 5]},
              "neutral": {"v": [2, 4],
                          "a": [2, 4]},
              "sad": {"v": [1, 2],
                      "a": [1, 5]}}

impro_or_script = 'impro'


def get_feature_file_name(FEATURES_TO_USE):
    FEATURESFILENAME = f"features_{FEATURES_TO_USE}_" \
                       f"{impro_or_script}.pkl"

    if "03" in EMOTIONS_TO_USE and "07" in EMOTIONS_TO_USE:
        FEATURESFILENAME = FEATURESFILENAME.replace(".pkl",
                                                    "_hapexc.pkl")
    elif "03" in EMOTIONS_TO_USE:
        FEATURESFILENAME = FEATURESFILENAME.replace(".pkl",
                                                    "_hap.pkl")
    elif "07" in EMOTIONS_TO_USE:
        FEATURESFILENAME = FEATURESFILENAME.replace(".pkl",
                                                    "_exc.pkl")

    return FEATURESFILENAME

EMO_THRESHOLD = None
