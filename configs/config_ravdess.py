import glob

DATASET_PATH = "/home/andrew/Data/Datasets/RAVDESS2/"
FEATURE_LOC = f"/home/andrew/Data/RAVDESS/"
SAVE_LOC = "/home/andrew/Data/RAVDESS/MoE/"
WAV_PATH = glob.glob("/home/andrew/Data/Datasets/RAVDESS2"
                     "/speech/*/*.wav")

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

CLASS_DICT = {EMOTIONS_TO_USE[e]: i for i, e in enumerate(EMOTIONS_TO_USE)}
CLASS_DICT_IDX = {CLASS_DICT[j]: j for j in CLASS_DICT}


def get_feature_file_name(FEATURES_TO_USE):
    FOLD_FILENAME = f"{FEATURES_TO_USE}.pkl"

    return FOLD_FILENAME

