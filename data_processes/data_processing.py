# Adapted from https://github.com/lessonxmk/head_fusion

import sys
from tqdm import tqdm
import os
import numpy as np
import librosa

from utilities import general_utilities as util


def get_mfcc(data, rate=16000, n_mfcc=26):
    """
    Extracts the mfcc for the input data
    Args:
        data: numpy.arrray - data to be converted
        rate: int - the original sampling rate of the wav file
        n_mfcc: int - the number of MFCC coefficients to extract

    Returns:
        mfcc_data: numpy.array - the feature extracted data
    """
    mfcc_data = librosa.feature.mfcc(y=data,
                                     sr=rate,
                                     n_mfcc=n_mfcc,
                                     win_length=2048,
                                     hop_length=512,
                                     center=True,
                                     pad_mode="constant")

    return mfcc_data


def get_features(config, files, overlap_value, clip_length,
                 data_type="train", dataset="iemocap"):
    """
    Gets the filenames from the dataset and checks if they meet the current
    configuration (emotions to use, length of the file etc.). The data and
    their corresponding labels are added to a dictionary. If this is
    training data, convert the dictionary to a numpy.array. In both cases,
    extract features requested from config file.
    Args:
        config: config file holding experimental settings
        files: list - the paths of the files in the dataset
        overlap_value: float - how much overlap when segmenting the wav files
        clip_length: float - how long each of the segments of the wav files
            will be
        dataset: str - set to the dataset in current use (naming conventions
            differ depending on dataset)

    Returns:
        feat_dict: dict - when data_type == "validation" - keys: data (
            features extracted), target
        data: numpy.array - when data_type == "train" - extracted features
        target: numpy.array - when data_type == "train"

    """
    clip_length_frames = clip_length * config.RATE
    overlap_frames = overlap_value * config.RATE
    hop_frames = clip_length_frames - overlap_frames
    # Length of the spectral feature (e.g. raw audio -> MFCC)
    segment_length = (clip_length_frames // config.HOP_SIZE) + 1
    data_dict = {}
    counter_for_final_array = 0
    for i, wav_file in enumerate(tqdm(files)):
        if dataset == "iemocap":
            # NAMING CONVENTION = Ses0X_impro01_XX_F000 or
            #                     Ses0X_script01_X_XX_F000
            label = wav_file.split("_")[-2]
            if label not in config.EMOTIONS_TO_USE:
                continue
        else:
            # NAMING_CONVENTION = Modality, Vocal Channel, Emotion, Intensity,
            #                     Statement, Repetition, Actor
            split_name = wav_file.split("/")[-1].split(".")[0].split("-")

            label = split_name[2]
            if label not in config.EMOTIONS_TO_USE:
                continue
        label = int(config.CLASS_DICT[config.EMOTIONS_TO_USE[label]])

        wav_data, _ = librosa.load(path=wav_file,
                                   sr=config.RATE)

        if clip_length_frames > wav_data.shape[0]:
            continue

        targets = []
        index = 0
        iterations = (wav_data.shape[0] - clip_length_frames) // hop_frames + 1
        iterations = int(iterations)
        counter_for_final_array += iterations
        total_data = np.zeros((iterations, config.N_MFCC, segment_length))
        for j in range(iterations):
            temp_data = wav_data[int(index):int(index + clip_length_frames)]
            features = get_mfcc(data=temp_data,
                                rate=config.RATE,
                                n_mfcc=config.N_MFCC)
            total_data[j, :, :] = features
            targets.append(label)
            index += hop_frames
        assert wav_data[int(index):int(index + clip_length_frames)].shape[0]\
               < clip_length_frames

        data_dict[i] = {
            "data": total_data,
            "targets": targets,
        }

    if data_type == "train":
        comp_data = np.zeros((counter_for_final_array,
                              config.N_MFCC,
                              segment_length))
        comp_labels = []
        counter = 0
        for i in data_dict:
            cur_size = data_dict[i]["data"].shape[0]
            comp_data[counter:counter+cur_size, :, :] = data_dict[i]["data"]
            comp_labels = comp_labels + data_dict[i]["targets"]
            counter += cur_size
        print(f"\nSize of training fold: {comp_data.shape}")
        return comp_data, comp_labels
    else:
        return data_dict


def process_data(config, wav_file_paths, clip_length=2, training_overlap=1,
                 validation_overlap=1.6, dataset="iemocap"):
    """
    Sort the dataset into folds and for each fold, extract features and
    create a training-based fold (numpy.array) and validation-based fold (dict)
    Args:
        config: config file holding experimental information
        wav_file_paths: list - paths to all the wav files in the dataset
        clip_length: float - how long each of the segments of the wav files
            will be
        training_overlap: float - how much overlap when segmenting the wav
            files for the training set
        validation_overlap: float - how much overlap when segmenting the wav
            files for the validation set
        dataset: str - set to the dataset in current use (naming conventions
            differ depending on dataset)

    Returns: None

    """
    total_folds = [[] for _ in range(config.NUM_FOLDS)]
    counter = 0
    if config.DATASET == "iemocap":
        if config.SESSION_TYPE != "both":
            if config.SESSION_TYPE == "script":
                wav_file_paths = list(filter(lambda x: "script" in x,
                                             wav_file_paths))
            else:
                wav_file_paths = list(filter(lambda x: "impro" in x,
                                             wav_file_paths))
    dataset_size = len(wav_file_paths)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    for current_ind in list(indices):
        total_folds[counter].append(wav_file_paths[current_ind])
        counter += 1
        if counter == config.NUM_FOLDS:
            counter = 0

    for p, fold in enumerate(total_folds):
        print(f"\nCreating fold {p} training data")
        fold_data, fold_targets = get_features(
            config=config,
            files=fold,
            overlap_value=training_overlap,
            clip_length=clip_length,
            data_type="train",
            dataset=dataset)
        print(f"\nCreating fold {p} validation data")
        valid_data = get_features(
            config=config,
            files=fold,
            overlap_value=validation_overlap,
            clip_length=clip_length,
            data_type="val",
            dataset=dataset)

        fold_dict = {f"train_data_{str(p)}": fold_data,
                     f"train_targets_{str(p)}": fold_targets,
                     "valid_data": valid_data}

        save_name = f"fold_{str(p)}_{config.FOLD_FILENAME}"
        util.save_pickle(location=os.path.join(config.FEATURE_LOC, save_name),
                         data=fold_dict)


def load_fold(config, fold):
    """
    Loads a fold, contains training data, training labels, and validation info
    Args:
        config: config file holding experimental settings
        fold: int - the fold to load

    Returns:
        dictionary: keys (str) train_data: numpy.array, train_target:
            numpy.array, val_dict: keys (str) data (numpy.array), target (
            numpy.array)

    """
    loc = f"fold_{str(fold)}_{config.FOLD_FILENAME}"
    loc = os.path.join(config.FEATURE_LOC, loc)
    if not os.path.exists(loc):
        sys.exit(f"The location of the fold does not exist, check SKIP_TRAIN "
                 f"is not True: {loc}")
    return util.load_pickle(location=loc)


def generate_fold_data(config, exp_fold):
    """
    Generates the data to be used for training or testing by loading and
    organising the respective folds of data.
    Args:
        config: config file holding experimental information
        exp_fold: int - current fold of the experiment

    Returns:
        train_data_features: numpy.array - the training data from the
            various folds
        train_targets: numpy.array - the relative labels for the training data
        valid_features_dict: dict: key: data (numpy.array),
            target (numpy.array)
    """
    train_data = np.empty(0)
    for current_fold in range(config.NUM_FOLDS - 1):
        fold_data = load_fold(config=config,
                              fold=current_fold)
        if exp_fold == current_fold:
            valid_data = fold_data["valid_data"]
        else:
            temp_data = fold_data[f"train_data_{current_fold}"]
            temp_targets = fold_data[f"train_targets_{current_fold}"]
            if len(train_data) == 0:
                train_data = temp_data
                train_targets = temp_targets
            else:
                train_data = np.concatenate((train_data, temp_data))
                train_targets = np.concatenate((train_targets, temp_targets))

    dataset_dict = {"train_data": train_data,
                    "train_targets": train_targets,
                    "valid_data": valid_data}

    return dataset_dict


def data_preprocessing(config, features_exist, exp_fold=0, dataset="iemocap"):
    """
    If the data folds do not exist for the experiment, create them. Load the
    data folds and separate them into training set and validation set.
    Args:
        config: config file holding experimental information
        features_exist: bool - set True if the extracted data folds exist
        exp_fold: int - the current experimental fold
        dataset: str - set to the dataset in current use (naming conventions
            differ depending on dataset)

    Returns:
        feat_dict: dict: keys - train_data (numpy.array),
            train_targets (numpy.array), val_dict (dict)
    """
    if not features_exist:
        process_data(config=config,
                     wav_file_paths=config.WAV_PATH,
                     clip_length=config.CLIP_LENGTH,
                     training_overlap=config.TRAINING_OVERLAP,
                     validation_overlap=config.VALIDATION_OVERLAP,
                     dataset=dataset)
        print("Finished creating dataset")
    dataset_dict = generate_fold_data(config=config,
                                      exp_fold=exp_fold)

    return dataset_dict
