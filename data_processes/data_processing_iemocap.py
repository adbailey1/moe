import sys
from tqdm import tqdm
import os
import numpy as np
import librosa
import random
from utilities import general_utilities as util

from data_processes.features import FeatureExtractor


def main_feature_extractor(config, files, label_dict, overlap, t,
                           feat_type='train'):
    meta_dict = {}
    for i, wav_file in enumerate(tqdm(files)):
        split_name = wav_file.split("/")[-1]
        session_num = int(split_name[4]) - 1
        gender = 0 if split_name.split("-")[-1][0] == "F" else 1
        # session 0 gender 0 -> 0; session 0 gender 1 -> 1
        # session 1 gender 0 -> 2; session 1 gender 1 -> 3
        speaker_id = (session_num * 2) + gender

        label = str(os.path.basename(wav_file).split('-')[2])
        if label not in label_dict:
            continue
        if (config.impro_or_script != 'all' and config.impro_or_script not in
                wav_file):
            continue
        label = label_dict[label]

        wav_data, _ = librosa.load(wav_file, sr=config.RATE)
        data = []
        labels_list = []
        spkr_list = []
        index = 0
        if t * config.RATE >= len(wav_data):
            continue

        while index + t * config.RATE < len(wav_data):
            data.append(wav_data[int(index):int(index + t * config.RATE)])
            labels_list.append(label)
            spkr_list.append(speaker_id)
            if feat_type == 'train':
                assert t - overlap > 0
                index += int((t - overlap) * config.RATE /
                             config.overlapTime[label])
                config.LABEL_NUM[label] += 1
            else:
                index += int((t - overlap) * config.RATE)
        data = np.array(data)
        meta_dict[i] = {
            "data": data,
            "labels": labels_list,
            "spkr": spkr_list,
            "path": wav_file
        }
    return meta_dict


def get_train_features(config, train_files, label_dict, train_overlap, t):
    meta_dict = main_feature_extractor(config, train_files, label_dict,
                                       train_overlap, t, feat_type='train')

    print("building data and labels...")
    train_data = []
    train_labels = []
    train_spkr = []
    for k in meta_dict:
        train_data.append(meta_dict[k]["data"])
        train_labels += meta_dict[k]["labels"]
        train_spkr += meta_dict[k]["spkr"]
    train_data = np.row_stack(train_data)
    train_labels = np.array(train_labels)
    train_spkr = np.array(train_spkr)

    assert len(train_data) == len(train_labels), \
        f"data length and label length must match! data shape: " \
        f"{train_data.shape}, labels length: {train_labels.shape}"

    return train_data, (train_labels, train_spkr)


def get_valid_features(config, valid_files, label_dict, val_overlap, t):
    if val_overlap >= t:
        val_overlap = t / 2

    val_dict = main_feature_extractor(config, valid_files, label_dict,
                                      val_overlap, t, feat_type='val')

    return val_dict


def process_data(config, path, t=2, train_overlap=1, val_overlap=1.6,
                 rate=16000):
    wav_files = path
    n = len(wav_files)
    if config.SPEAKER_IND:
        total_folds = [[] for _ in range(config.NUM_SESSIONS)]
        for file in wav_files:
            split_name = file.split("/")[-1]
            session_num = int(split_name[4])
            total_folds[session_num - 1].append(file)
    else:
        indices = np.arange(n)
        np.random.shuffle(indices)
        total_folds = [[] for _ in range(config.NUM_FOLDS)]
        counter = 0
        for current_ind in list(indices):
            total_folds[counter].append(wav_files[current_ind])
            counter += 1
            if counter == config.NUM_FOLDS:
                counter = 0

    for p, fold in enumerate(total_folds):
        if config.SPEAKER_IND:
            random.shuffle(fold)
        fold_data, fold_labels = get_train_features(config, fold,
                                                    config.EMOTIONS_TO_USE,
                                                    train_overlap, t)
        print(f"fold_{p}.shape: {fold_data.shape}")

        feature_extractor = FeatureExtractor(rate=rate)
        fold_data_features = feature_extractor.get_features(
            config.FEATURES_TO_USE, fold_data)
        fold_data_valid = get_valid_features(config, fold,
                                             config.EMOTIONS_TO_USE,
                                             val_overlap, t)
        valid_features_dict = generate_validation_feat(config,
                                                       fold_data_valid,
                                                       feature_extractor)

        feat_dict = {f"train_data_{str(p)}": fold_data_features,
                     f"train_labels_{str(p)}": fold_labels,
                     "val_dict": valid_features_dict}
        if config.SPEAKER_IND:
            save_name = f"session_{str(p)}_{config.FEATURESFILENAME}"
        else:
            save_name = f"fold_{str(p)}_{config.FEATURESFILENAME}"
        util.save_pickle(os.path.join(config.FEATURE_LOC, save_name),
                         feat_dict)


def generate_validation_feat(config, data_dict, feature_extractor):
    feat_dict = {}
    for i in data_dict:
        data = feature_extractor.get_features(
            config.FEATURES_TO_USE, data_dict[i]["data"])
        feat_dict[i] = {
            "data": data,
            "label": data_dict[i]["labels"],
            "spkr": data_dict[i]["spkr"],
        }

    return feat_dict


def load_fold(config, fold):
    if config.SPEAKER_IND:
        loc = f"session_{str(fold)}_{config.FEATURESFILENAME}"
        loc = os.path.join(config.FEATURE_LOC, loc)
    else:
        loc = f"fold_{str(fold)}_{config.FEATURESFILENAME}"
        loc = os.path.join(config.FEATURE_LOC, loc)
    if not os.path.exists(loc):
        sys.exit(f"The location of the fold does not exist, check SKIP_TRAIN "
                 f"is not True: {loc}")
    return util.load_pickle(loc)


def generate_fold_data(config, exp_fold):
    train_data_features = np.empty(0)
    for current_fold in range(config.NUM_FOLDS - 1):
        feats = load_fold(config, current_fold)
        if exp_fold == current_fold:
            valid_features_dict = feats["val_dict"]
        else:
            temp_train_data_features = feats[f"train_data_{current_fold}"]
            temp_train_labels = feats[f"train_labels_{current_fold}"]
            if len(temp_train_labels) == 2:
                temp_label, temp_spkr_id = temp_train_labels
            if len(train_data_features) == 0:
                train_data_features = temp_train_data_features
                train_labels = temp_label
                train_spkr_id = temp_spkr_id
            else:
                train_data_features = np.concatenate((
                    train_data_features, temp_train_data_features))
                train_labels = np.concatenate((train_labels, temp_label))
                train_spkr_id = np.concatenate((train_spkr_id, temp_spkr_id))

    return train_data_features, (train_labels, train_spkr_id), \
           valid_features_dict


def data_preprocessing(config, logger, features_exist, exp_fold=0):
    if not features_exist:
        logger.info("creating meta dict...")
        process_data(config, config.WAV_PATH, t=config.T_STRIDE,
                     train_overlap=config.T_OVERLAP)

    train_data_features, train_labels, valid_features_dict = \
        generate_fold_data(config, exp_fold)

    feat_dict = {"train_data": train_data_features,
                 "train_labels": train_labels[0],
                 "train_spkr_id": train_labels[1],
                 "val_dict": valid_features_dict}

    return feat_dict
