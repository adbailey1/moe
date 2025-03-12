import glob
import sys
from tqdm import tqdm
import os
import numpy as np
import random
import librosa
import pandas as pd
import pickle
import csv
import soundfile as sf
from utilities import general_utilities as util

from data_processes.features import FeatureExtractor


EPS = 1e-6

def get_train_features(config, train_files, label_dict, train_overlap, t):
    meta_dict = {}
    total_counter = skip_counter = 0
    for i, wav_file in enumerate(tqdm(train_files)):
        special_case = False
        if not config.GENERATE_GENDER_LABELS:
            labels = wav_file[1:]
            wav_file = wav_file[0]
            label = [i for i, d in enumerate(labels) if float(d) >
                     config.EMO_THRESHOLD]
            if len(label) == 0:
                # only set neutral if at least 2/3 annotators set all
                # emotions to 0
                label = -1 if max(float(d) for d in labels) < .3 else -2
                # label = -1
            else:
                if len(label) > 1:
                    in_label_dict = [i for i, d in enumerate(label) if d in label_dict]
                    if len(in_label_dict) == 0:
                        continue
                    elif len(in_label_dict) == 1:
                        label = label[in_label_dict[0]]
                    elif len(in_label_dict) > 1:
                        if config.LIKE_IEMOCAP:
                            if labels[label[in_label_dict[0]]] == labels[
                                label[in_label_dict[1]]]:
                                continue
                            elif labels[label[in_label_dict[0]]] > labels[
                                label[in_label_dict[1]]]:
                                label = label[in_label_dict[0]]
                            else:
                                label = label[in_label_dict[1]]
                        else:
                            special_case = True
                            label = [label_dict[label[ind]] for ind in in_label_dict]
                            label = "_".join(label)
                            # maximum = 0
                            # for index, d in enumerate(in_label_dict):
                            #     if float(labels[label[d]]) > maximum:
                            #         maximum = float(labels[label[d]])
                            #         to_use = index
                            #         skip = False
                            #     elif float(labels[label[d]]) == maximum:
                            #         skip = True
                            # if skip:
                            #     continue
                            # else:
                            #     label = label[in_label_dict[to_use]]
                else:
                    label = label[0]

            if label not in label_dict and not special_case:
                continue
            if not special_case:
                label = label_dict[label]

        wav_data, _ = librosa.load(wav_file, sr=config.RATE)
        X1 = []
        y1 = []

        index = 0
        if (t * config.RATE >= len(wav_data)):
            continue
        if config.GENERATE_GENDER_LABELS:
            label = label_dict[wav_file]
        while (index + t * config.RATE < len(wav_data)):
            X1.append(wav_data[int(index):int(index + t * config.RATE)])
            y1.append(label)
            # g1.append(gender)
            # s1.append(speaker_id)
            # v1.append(valence)
            # a1.append(arousal)
            assert t - train_overlap > 0
            index += int((t - train_overlap) * config.RATE)
            if special_case:
                current_labels = label.split("_")
                for cl in current_labels:
                    config.LABEL_NUM[cl] += 1
            else:
                config.LABEL_NUM[label] += 1
        X1 = np.array(X1)
        if np.any(np.isnan(X1)):
            print("Error")
        meta_dict[i] = {
            "X": X1,
            "y": y1,
            # "g": g1,
            # "s": s1,
            # "v": v1,
            # "a": a1,
            "path": wav_file
        }

    print("building X, y...")
    if config.use_valence_arousal:
        print(f"The total number of files is: {total_counter}, the number "
              f"skipped due to valence/arousal mismatches is: {skip_counter}")
    train_X = []
    train_y = []
    # train_g = []
    # train_s = []
    # train_v = []
    # train_a = []
    for k in tqdm(meta_dict):
        train_X.append(meta_dict[k]["X"])
        train_y += meta_dict[k]["y"]
        # train_g += meta_dict[k]["g"]
        # train_s += meta_dict[k]["s"]
        # train_v += meta_dict[k]["v"]
        # train_a += meta_dict[k]["a"]
    train_X = np.row_stack(train_X)
    train_y = np.array(train_y)
    # train_g = np.array(train_g)
    # train_s = np.array(train_s)
    # train_v = np.array(train_v)
    # train_a = np.array(train_a)
    assert len(train_X) == len(
        train_y), f"X length and y length must match! X shape: " \
                  f"{train_X.shape}, y length: {train_y.shape}"

    return train_X, train_y


def get_valid_features(config, label_dict, valid_files, val_overlap, t):
    val_dict = {}
    val_map = {}
    total_counter = skip_counter = 0
    if val_overlap >= t:
        val_overlap = t / 2
    for i, wav_file in enumerate(tqdm(valid_files)):
        special_case = False
        if not config.GENERATE_GENDER_LABELS:
            labels = wav_file[1:]
            wav_file = wav_file[0]
            label = [i for i, d in enumerate(labels) if float(d) > config.EMO_THRESHOLD]
            if len(label) == 0:
                label = -1 if max(float(d) for d in labels) < .3 else -2
                # label = -1
            else:
                if len(label) > 1:
                    in_label_dict = [i for i, d in enumerate(label) if d in label_dict]
                    if len(in_label_dict) == 0:
                        continue
                    elif len(in_label_dict) == 1:
                        label = label[in_label_dict[0]]
                    elif len(in_label_dict) > 1:
                        if config.LIKE_IEMOCAP:
                            if labels[label[in_label_dict[0]]] == labels[
                                label[in_label_dict[1]]]:
                                continue
                            elif labels[label[in_label_dict[0]]] > labels[
                                label[in_label_dict[1]]]:
                                label = label[in_label_dict[0]]
                            else:
                                label = label[in_label_dict[1]]
                        else:
                            label = [label_dict[label[ind]] for ind in in_label_dict]
                            label = "_".join(label)
                            special_case = True
                            # maximum = 0
                            # for index, d in enumerate(in_label_dict):
                            #     if float(labels[label[d]]) > maximum:
                            #         maximum = float(labels[label[d]])
                            #         to_use = index
                            #         skip = False
                            #     elif float(labels[label[d]]) == maximum:
                            #         skip = True
                            # if skip:
                            #     continue
                            # else:
                            #     label = label[in_label_dict[to_use]]
                else:
                    label = label[0]

            if label not in label_dict and not special_case:
                continue
            if not special_case:
                label = label_dict[label]
        name = wav_file.split("/")[-1]
        val_map[i] = name
        wav_data, _ = librosa.load(wav_file, sr=config.RATE)

        X1 = []
        y1 = []
        # g1 = []
        # s1 = []
        # v1 = []
        # a1 = []
        if config.GENERATE_GENDER_LABELS:
            label = label_dict[wav_file]
        index = 0
        if (t * config.RATE >= len(wav_data)):
            continue
        while (index + t * config.RATE < len(wav_data)):
            X1.append(wav_data[int(index):int(index + t * config.RATE)])
            y1.append(label)
            # g1.append(gender)
            # s1.append(speaker_id)
            # v1.append(valence)
            # a1.append(arousal)
            index += int((t - val_overlap) * config.RATE)

        X1 = np.array(X1)
        if np.any(np.isnan(X1)):
            print("Error")
        # feature_extractor = FeatureExtractor(rate=config.RATE)
        # X1 = feature_extractor.get_features(
        #     config.FEATURES_TO_USE, X1)
        val_dict[i] = {
            "X": X1,
            "y": y1,
            # "g": g1,
            # "s": s1,
            # "v": v1,
            # "a": a1,
            "path": wav_file
        }

    with open(os.path.join(config.EXP_DIR, "val_mapper.pkl"), "wb") as f:
        pickle.dump(val_map, f)
    return val_dict


def process_data(config, path, logger, t=2, train_overlap=1, val_overlap=1.6,
                 rate=16000, for_gender_labels=False):
    wav_files = path
    train_files = []
    valid_files = []
    test_files = []
    wav_files = glob.glob(config.SEGMENTED_AUDIO_LOCATION + "/*.wav")
    if for_gender_labels:
        label_loc = os.path.join(config.DATASET,
                                 config.COMPLETE_GENDER_LABEL_FILE)
        labels = pd.read_csv(label_loc)
        files = labels["File"].to_list()
        gender = labels["Gender"].to_list()
        train = {files[i]: gender[i] for i in range(len(gender)) if
                 gender[i] != -1}

        n = len(train)

        if len(config.DATA_SPLIT) > 2:
            train_split, val_split, test_split = config.DATA_SPLIT
        elif len(config.DATA_SPLIT) == 1:
            valid_split = config.DATA_SPLIT[0]
            valid_indices = list(np.random.choice(range(n),
                                                  int(n * valid_split),
                                                  replace=False))
        else:
            train_split, val_split = config.DATA_SPLIT
        if len(config.DATA_SPLIT) > 1:
            train_indices = list(np.random.choice(range(n),
                                                  int(n * train_split),
                                                  replace=False))
            if config.DATA_SPLIT[0] == 1. and config.DATA_SPLIT[0] == 1.:
                valid_indices = train_indices
            else:
                valid_indices = list(set(range(n)) - set(train_indices))
        if len(config.DATA_SPLIT) > 2:
            test_indices = list(np.random.choice(valid_indices,
                                                 len(valid_indices) // 2,
                                                 replace=False))
            valid_indices = list(set(valid_indices) - set(test_indices))
            for ind in test_indices:
                # test_files.append(list(test.keys())[ind])
                test_files.append(list(train.keys())[ind])

        if len(config.DATA_SPLIT) > 1:
            for ind in train_indices:
                train_files.append(list(train.keys())[ind])

        for ind in valid_indices:
            valid_files.append(list(train.keys())[ind])

        print(f"Constructing meta dictionary for {path[0]}...")
        if len(config.DATA_SPLIT) > 1:
            train_X, train_y = get_train_features(config, train_files,
                                                  train,
                                                  train_overlap, t)
            print(f"train_X.shape: {train_X.shape}")

        val_dict = get_valid_features(config, train, valid_files,
                                      val_overlap, t)
        print(f"len(val_dict): {len(val_dict)}")

        print("getting features")
        logger.info("getting features")
        feature_extractor = FeatureExtractor(rate=config.RATE)
        feat_dict = {}
        if len(config.DATA_SPLIT) > 1:
            train_X_features = feature_extractor.get_features(
                config.FEATURES_TO_USE, train_X)
            feat_dict = {"train_X": train_X_features,
                         "train_y": train_y}

        valid_features_dict = generate_validation_feat(config,
                                                       val_dict,
                                                       feature_extractor)

        feat_dict["val_dict"] = valid_features_dict

        if len(config.DATA_SPLIT) > 2:
            # test_dict = get_valid_features(config, test_files,
            #                                test,
            #                                val_overlap, t)
            test_dict = get_valid_features(config, train, test_files,
                                           val_overlap, t)
            print(f"len(test_dict): {len(test_dict)}")
            test_features_dict = generate_validation_feat(config,
                                                          test_dict,
                                                          feature_extractor)
            feat_dict["test_dict"] = test_features_dict

        save_loc = os.path.join(config.FEATURE_LOC, str(config.EMO_THRESHOLD), config.FEATURESFILENAME)
        util.save_pickle(save_loc, feat_dict)
    else:
        if config.USE_FOLDS:
            if len(wav_files) == 0:
                files = util.load_pickle(os.path.join(config.DATASET_PATH,
                                                      config.SDK_LABELS))
                intervals = util.load_pickle(os.path.join(config.DATASET_PATH,
                                                          config.SDK_INTERVALS))
                scores = util.load_pickle(os.path.join(config.DATASET_PATH,
                                                       config.SDK_SCORES))
                if not os.path.exists(config.SEGMENTED_AUDIO_LOCATION):
                    os.makedirs(config.SEGMENTED_AUDIO_LOCATION)

                complete_labels = []
                stats = {"<2": 0, "<4": 0, "<6": 0, ">6": 0}
                for i in tqdm(range(len(files))):
                    name = files[i]
                    print(f"Writing: {name} at locations: {intervals[i]}")
                    temp_wav = os.path.join(config.DATASET,
                                            config.AUDIO_LOC,
                                            name + ".wav")
                    wav_data, _ = librosa.load(temp_wav, sr=rate)
                    for j in range(len(intervals[i])):
                        score = scores[i][j][1:]
                        start = intervals[i][j][0]
                        end = intervals[i][j][1]
                        start_samples = int(start * rate)
                        end_samples = int(end * rate)

                        save_loc = os.path.join(config.SEGMENTED_AUDIO_LOCATION,
                                                name + f"_{str(j)}.wav")
                        new_wav = wav_data[start_samples:end_samples]
                        if end - start < 2:
                            stats["<2"] += 1
                        elif 2 <= end - start < 4:
                            stats["<4"] += 1
                        elif 4 <= end - start < 6:
                            stats["<6"] += 1
                        elif end - start >= 6:
                            stats[">6"] += 1
                        else:
                            print("error")

                        sf.write(save_loc, new_wav, rate)
                        complete_labels.append([save_loc] + list(score))

                header = ["file"] + config.EMOTION_LABELS_FROM_SDK[1:]
                with open(os.path.join(config.DATASET_PATH, config.SEGMENTED_LABELS),
                          "w", encoding="UTF-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(complete_labels)
                with open(os.path.join(config.DATASET, "Labels/stats.pkl"),
                          "wb") as f:
                    pickle.dump(stats, f)
            else:
                complete_labels = []
                with open(os.path.join(config.DATASET_PATH,
                                       config.SEGMENTED_LABELS),
                          newline="") as f:
                    reader = csv.reader(f)
                    for i in reader:
                        complete_labels.append(i)
                complete_labels = complete_labels[1:]
            train = {i[0]: i[1:] for i in complete_labels}
            n = len(train)

            if config.USE_FOLDS and config.SPEAKER_IND:
                pass
                # total_folds = [[] for _ in range(config.NUM_SESSIONS)]
                # for file in wav_files:
                #     split_name = file.split("/")[-1]
                #     session_num = int(split_name[4])
                #     total_folds[session_num - 1].append(file)
            else:
                indices = np.arange(n)
                np.random.shuffle(indices)
                total_folds = [[] for _ in range(config.NUM_FOLDS)]
                counter = 0
                for current_ind in list(indices):
                    temp = wav_files[current_ind].replace("/home/andrew/Data/Datasets",
                                                          "/mnt/6663b3e6-a12f-49e8-b881-421cebf2f8c6/datasets")
                    # total_folds[counter].append([wav_files[current_ind]] +
                    #                             train[wav_files[current_ind]])
                    total_folds[counter].append([wav_files[current_ind]] +
                                                train[temp])
                    counter += 1
                    if counter == config.NUM_FOLDS:
                        counter = 0

            for p, fold in enumerate(total_folds):
                if config.USE_FOLDS and config.SPEAKER_IND:
                    random.shuffle(fold)
                fold_x, fold_y = get_train_features(config, fold,
                                                    config.EMOTIONS_TO_USE,
                                                    train_overlap, t)
                print(f"fold_{p}.shape: {fold_x.shape}")

                feature_extractor = FeatureExtractor(rate=rate)
                fold_x_features = feature_extractor.get_features(
                    config.FEATURES_TO_USE, fold_x)
                fold_x_valid = get_valid_features(
                    config=config, label_dict=config.EMOTIONS_TO_USE,
                    valid_files=fold, val_overlap=val_overlap, t=t)
                valid_features_dict = generate_validation_feat(config,
                                                               fold_x_valid,
                                                               feature_extractor)

                feat_dict = {f"train_x_{str(p)}": fold_x_features,
                             f"train_y_{str(p)}": fold_y,
                             "val_dict": valid_features_dict}
                if config.SPEAKER_IND:
                    save_name = f"session_{str(p)}_{config.FEATURESFILENAME}"
                else:
                    save_name = f"fold_{str(p)}_features_mfcc.pkl"
                if config.LIKE_IEMOCAP:
                    save_loc = os.path.join(config.FEATURE_LOC,
                                            f"LIKE_IEMOCAP_{config.EMO_THRESHOLD}")
                    if not os.path.exists(save_loc):
                        os.mkdir(save_loc)
                    save_loc = os.path.join(save_loc, save_name)

                    util.save_pickle(save_loc, feat_dict)
                else:
                    util.save_pickle(os.path.join(config.FEATURE_LOC,
                                                  str(config.EMO_THRESHOLD),
                                                  save_name), feat_dict)
        else:
            if len(wav_files) == 0:
                files = util.load_pickle(os.path.join(config.DATASET,
                                                      config.SDK_LABELS))
                intervals = util.load_pickle(os.path.join(config.DATASET,
                                                          config.SDK_INTERVALS))
                scores = util.load_pickle(os.path.join(config.DATASET,
                                                       config.SDK_SCORES))
                if not os.path.exists(config.SEGMENTED_AUDIO_LOCATION):
                    os.makedirs(config.SEGMENTED_AUDIO_LOCATION)

                complete_labels = []
                stats = {"<2": 0, "<4": 0, "<6": 0, ">6": 0}
                for i in tqdm(range(len(files))):
                    name = files[i]
                    print(f"Writing: {name} at locations: {intervals[i]}")
                    temp_wav = os.path.join(config.DATASET,
                                            config.AUDIO_LOC,
                                            name + ".wav")
                    wav_data, _ = librosa.load(temp_wav, sr=rate)
                    for j in range(len(intervals[i])):
                        score = scores[i][j][1:]
                        start = intervals[i][j][0]
                        end = intervals[i][j][1]
                        start_samples = int(start * rate)
                        end_samples = int(end * rate)

                        save_loc = os.path.join(config.SEGMENTED_AUDIO_LOCATION,
                                                name + f"_{str(j)}.wav")
                        new_wav = wav_data[start_samples:end_samples]
                        if end - start < 2:
                            stats["<2"] += 1
                        elif 2 <= end - start < 4:
                            stats["<4"] += 1
                        elif 4 <= end - start < 6:
                            stats["<6"] += 1
                        elif end - start >= 6:
                            stats[">6"] += 1
                        else:
                            print("error")

                        sf.write(save_loc, new_wav, rate)
                        complete_labels.append([save_loc] + list(score))

                header = ["file"] + config.EMOTION_LABELS_FROM_SDK[1:]
                with open(os.path.join(config.DATASET,
                                       config.SEGMENTED_LABELS),
                          "w", encoding="UTF-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(complete_labels)
                with open(os.path.join(config.DATASET, "Labels/stats.pkl"),
                          "wb") as f:
                    pickle.dump(stats, f)
            else:
                complete_labels = []
                with open(os.path.join(config.DATASET, config.SEGMENTED_LABELS),
                          newline="") as f:
                    reader = csv.reader(f)
                    for i in reader:
                        complete_labels.append(i)
                complete_labels = complete_labels[1:]
            train = {i[0]: i[1:] for i in complete_labels}

            n = len(train)
            if len(config.DATA_SPLIT) > 2:
                train_split, val_split, test_split = config.DATA_SPLIT
            elif len(config.DATA_SPLIT) == 1:
                valid_split = config.DATA_SPLIT[0]
                valid_indices = list(np.random.choice(range(n),
                                                      int(n * valid_split),
                                                      replace=False))
            else:
                train_split, val_split = config.DATA_SPLIT
            if len(config.DATA_SPLIT) > 1:
                train_indices = list(np.random.choice(range(n),
                                                      int(n * train_split),
                                                      replace=False))
                if config.DATA_SPLIT[0] == 1. and config.DATA_SPLIT[0] == 1.:
                    valid_indices = train_indices
                else:
                    valid_indices = list(set(range(n)) - set(train_indices))
            if len(config.DATA_SPLIT) > 2:
                test_indices = list(np.random.choice(valid_indices,
                                    len(valid_indices) // 2, replace=False))
                valid_indices = list(set(valid_indices) - set(test_indices))
                for ind in test_indices:
                    test_files.append([wav_files[ind]] + train[wav_files[ind]])

            if len(config.DATA_SPLIT) > 1:
                for ind in train_indices:
                    train_files.append([wav_files[ind]] + train[wav_files[ind]])

            for ind in valid_indices:
                valid_files.append([wav_files[ind]] + train[wav_files[ind]])

            print(f"Constructing meta dictionary for {path[0]}...")
            if len(config.DATA_SPLIT) > 1:
                train_X, train_y = get_train_features(config, train_files,
                                                      config.EMOTIONS_TO_USE,
                                                      train_overlap, t)
                print(f"train_X.shape: {train_X.shape}")

            val_dict = get_valid_features(config, config.EMOTIONS_TO_USE,
                                          valid_files, val_overlap, t)
            print(f"len(val_dict): {len(val_dict)}")

            print("getting features")
            logger.info("getting features")
            feature_extractor = FeatureExtractor(rate=config.RATE)
            feat_dict = {}
            if len(config.DATA_SPLIT) > 1:
                train_X_features = feature_extractor.get_features(
                    config.FEATURES_TO_USE, train_X)
                del train_X
                feat_dict = {"train_X": train_X_features,
                             "train_y": train_y}

            valid_features_dict = generate_validation_feat(config,
                                                           val_dict,
                                                           feature_extractor)

            feat_dict["val_dict"] = valid_features_dict

            if len(config.DATA_SPLIT) > 2:
                test_dict = get_valid_features(config, config.EMOTIONS_TO_USE,
                                               test_files, val_overlap, t)
                print(f"len(test_dict): {len(test_dict)}")
                test_features_dict = generate_validation_feat(config,
                                                              test_dict,
                                                              feature_extractor)
                feat_dict["test_dict"] = test_features_dict

            save_loc = os.path.join(config.FEATURE_LOC, config.FEATURESFILENAME)
            util.save_pickle(save_loc, feat_dict)


def generate_validation_feat(config, data_dict, feature_extractor):
    feat_dict = {}
    for i in tqdm(data_dict):
        X1 = feature_extractor.get_features(
            config.FEATURES_TO_USE, data_dict[i]["X"])
        if np.any(np.isnan(X1)):
            print("wait")
        feat_dict[i] = {
            "X": X1,
            "y": data_dict[i]["y"],
            # "g": data_dict[i]["g"],
            # "s": data_dict[i]["s"],
            # "v": data_dict[i]["v"],
            # "a": data_dict[i]["a"]
        }

    return feat_dict


def load_fold(config, fold):
    if config.SPEAKER_IND:
        loc = f"session_{str(fold)}_{config.FEATURESFILENAME}"
    else:
        if config.LIKE_IEMOCAP:
            loc = f"LIKE_IEMOCAP_{config.EMO_THRESHOLD}/fold_{str(fold)}" \
                  f"_{config.FEATURESFILENAME}"
        else:
            loc = f"{config.EMO_THRESHOLD}/fold_{str(fold)}_{config.FEATURESFILENAME}"
        loc = os.path.join(config.FEATURE_LOC, loc)
    if not os.path.exists(loc):
        sys.exit(f"The location of the fold does not exist, check SKIP_TRAIN "
                 f"is not True: {loc}")
    return util.load_pickle(loc)


def generate_fold_data(config, exp_fold):
    train_X_features = np.empty(0)
    for current_fold in range(config.NUM_FOLDS - 1):
        feats = load_fold(config, current_fold)
        if exp_fold == current_fold:
            valid_features_dict = feats["val_dict"]
        else:
            temp_train_X_features = feats[f"train_x_{current_fold}"]
            temp_train_y = feats[f"train_y_{current_fold}"]
            # temp_y, temp_gen, temp_spkr_id, temp_val, temp_arou = temp_train_y
            if len(train_X_features) == 0:
                train_X_features = temp_train_X_features
                train_y = temp_train_y
                # train_y = temp_y
                # train_gen = temp_gen
                # train_spkr_id = temp_spkr_id
                # train_valence = temp_val
                # train_arousal = temp_arou
            else:
                train_X_features = np.concatenate((
                    train_X_features, temp_train_X_features))
                train_y = np.concatenate((train_y, temp_train_y))
                # train_y = np.concatenate((train_y, temp_y))
                # train_gen = np.concatenate((train_gen, temp_gen))
                # train_spkr_id = np.concatenate((train_spkr_id, temp_spkr_id))
                # train_valence = np.concatenate((train_valence, temp_val))
                # train_arousal = np.concatenate((train_arousal, temp_arou))
    return train_X_features, (train_y,), valid_features_dict

    # return train_X_features, (train_y, train_gen, train_spkr_id,
    #                           train_valence, train_arousal), valid_features_dict


def data_preprocessing(config, logger, features_exist, exp_fold=0,
                       for_gender_labels=False):
    if not features_exist:
        logger.info("creating meta dict...")
        process_data(config, config.WAV_PATH, logger, t=config.T_STRIDE,
                     train_overlap=config.T_OVERLAP,
                     for_gender_labels=for_gender_labels)
    if config.USE_FOLDS:
        train_X_features, train_y, valid_features_dict = \
            generate_fold_data(config, exp_fold)
        feat_dict = {"train_X": train_X_features,
                     "train_y": train_y[0],
                     "val_dict": valid_features_dict}

        # if len(train_y) == 5:
        #     feat_dict = {"train_X": train_X_features,
        #                  "train_y": train_y[0],
        #                  "train_gen": train_y[1],
        #                  "train_spkr_id": train_y[2],
        #                  "train_valence": train_y[3],
        #                  "train_arousal": train_y[4],
        #                  "val_dict": valid_features_dict}
        # else:
        #     feat_dict = {"train_X": train_X_features,
        #                  "train_y": train_y[0],
        #                  "train_gen": train_y[1],
        #                  "train_spkr_id": train_y[2],
        #                  "val_dict": valid_features_dict}
    else:
        loc = os.path.join(config.FEATURE_LOC, config.FEATURESFILENAME)
        feat_dict = util.load_pickle(loc)

    return feat_dict
