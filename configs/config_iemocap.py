import glob
import os


FEATURE_LOC = "/home/andrew/Data/IEMOCAP/"
SAVE_LOC = "/home/andrew/Data/IEMOCAP/MoE/"
DATASET_PATH = "/home/andrew/Data/Datasets/IEMOCAP_stripped"
WAV_PATH = glob.glob(
    os.path.join(DATASET_PATH,
                 "IEMOCAP_full_release/*/sentences/wav/*/S*.wav"))


EMOTIONS_TO_USE = {
    '01': 'neutral',
    # '02': 'frustration',
    # '03': 'happiness',
    '04': 'excited',
    '05': 'sadness',
    '06': 'angry',
    # '07': 'fear',
    # '08': 'surprise',
    # '09': 'disgust',
    # '10': 'other',
    # '11': 'xxx'
}

CLASS_DICT = {EMOTIONS_TO_USE[e]: i for i, e in enumerate(EMOTIONS_TO_USE)}
CLASS_DICT_IDX = {CLASS_DICT[j]: j for j in CLASS_DICT}


DATABASE_EMO_IDX = {
    '01': 'neu',  # neutral
    '02': 'fru',  # frustration
    '03': 'hap',  # happiness
    '04': 'exc',  # excited
    '05': 'sad',  # sadness
    '06': 'ang',  # angry
    '07': 'fea',  # fear
    '08': 'sur',  # surprise
    '09': 'dis',  # disgust
    '10': 'oth',  # other
    '11': 'xxx'   # no agreement from annotators
}

IDX_EMO_DATABASE = {DATABASE_EMO_IDX[i]: i for i in DATABASE_EMO_IDX}
# 'impro' / 'script' / 'both' NOTE we only consider 'impro' for this work
SESSION_TYPE = 'impro'


def get_feature_file_name(FEATURES_TO_USE):
    FOLD_FILENAME = f"{FEATURES_TO_USE}_{SESSION_TYPE}.pkl"

    if "03" in EMOTIONS_TO_USE and "07" in EMOTIONS_TO_USE:
        FOLD_FILENAME = FOLD_FILENAME.replace(".pkl", "_hapexc.pkl")
    elif "03" in EMOTIONS_TO_USE:
        FOLD_FILENAME = FOLD_FILENAME.replace(".pkl", "_hap.pkl")
    elif "07" in EMOTIONS_TO_USE:
        FOLD_FILENAME = FOLD_FILENAME.replace(".pkl", "_exc.pkl")

    return FOLD_FILENAME
