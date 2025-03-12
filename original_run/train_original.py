# Code taken from https://github.com/lessonxmk/head_fusion
# Adapted slightly to run out of the box and to iterate over 5 seeds as
# referenced in the paper
import glob
import os
import pickle
import random
import time
import math
import logging
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import librosa
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(mod):
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)


# setup_seed(111111)
# setup_seed(123456)
# setup_seed(0)
# setup_seed(999999)
# setup_seed(987654)

# Code adapted for ease of running
seeds = [111111, 123456, 0, 999999, 987654]
WAV_PATH = "/path/to/wav"
CODE_DIR = "/path/to/code"
net_type = "macnn"  # macnn / maccn_x4 / moe
impro_or_script = 'impro'  # We only consider improvised scripts for this work

featuresExist = False

toSaveFeatures = True

import features_original
import model_original
import data_loader_original
# import CapsNet


def process_data(path, t=2, train_overlap=1, val_overlap=1.6, RATE=16000):
    path = path.rstrip('/')
    wav_files = glob.glob(path +
                          '/IEMOCAP_full_release/*/sentences/wav/*/*.wav')
    meta_dict = {}
    val_dict = {}
    LABEL_DICT1 = {
        '01': 'neutral',
        # '02': 'frustration',
        # '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        # '06': 'fearful',
        '07': 'happy',  # excitement->happy
        # '08': 'surprised'
    }

    label_num = {
        'neutral': 0,
        'happy': 0,
        'sad': 0,
        'angry': 0,
    }

    n = len(wav_files)
    train_files = []
    valid_files = []
    train_indices = list(np.random.choice(range(n), int(n * 0.8), replace=False))
    valid_indices = list(set(range(n)) - set(train_indices))
    for i in train_indices:
        train_files.append(wav_files[i])
    for i in valid_indices:
        valid_files.append(wav_files[i])

    print("constructing meta dictionary for {}...".format(path))
    for i, wav_file in enumerate(tqdm(train_files)):
        if len(os.path.basename(wav_file).split('-')) < 3:
            continue
        label = str(os.path.basename(wav_file).split('-')[2])
        if (label not in LABEL_DICT1):
            continue
        if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
            continue
        label = LABEL_DICT1[label]

        # ToDo This has stopped working: fixed by scipy.io
        wav_data, _ = librosa.load(wav_file, sr=RATE)
        # from scipy.io import wavfile
        # _, wav_data = wavfile.read(wav_file)
        X1 = []
        y1 = []
        index = 0
        if (t * RATE >= len(wav_data)):
            continue

        while (index + t * RATE < len(wav_data)):
            X1.append(wav_data[int(index):int(index + t * RATE)])
            y1.append(label)
            assert t - train_overlap > 0
            index += int((t - train_overlap) * RATE / overlapTime[label])
            label_num[label] += 1
        X1 = np.array(X1)
        meta_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }

    print("building X, y...")
    train_X = []
    train_y = []
    for k in meta_dict:
        train_X.append(meta_dict[k]['X'])
        train_y += meta_dict[k]['y']
    train_X = np.row_stack(train_X)
    train_y = np.array(train_y)
    assert len(train_X) == len(train_y), "X length and y length must match! X shape: {}, y length: {}".format(
        train_X.shape, train_y.shape)

    if (val_overlap >= t):
        val_overlap = t / 2
    for i, wav_file in enumerate(tqdm(valid_files)):
        if len(os.path.basename(wav_file).split('-')) < 3:
            continue
        label = str(os.path.basename(wav_file).split('-')[2])
        if (label not in LABEL_DICT1):
            continue
        if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
            continue
        label = LABEL_DICT1[label]
        # ToDo This has stopped working: fixed by scipy.io
        # wav_data, _ = librosa.load(wav_file, sr=RATE)
        from scipy.io import wavfile
        _, wav_data = wavfile.read(wav_file)
        X1 = []
        y1 = []
        index = 0
        if (t * RATE >= len(wav_data)):
            continue
        while (index + t * RATE < len(wav_data)):
            X1.append(wav_data[int(index):int(index + t * RATE)])
            y1.append(label)
            index += int((t - val_overlap) * RATE)

        X1 = np.array(X1)
        val_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }

    return train_X, train_y, val_dict


def process_features(X, u=255):
    X = torch.from_numpy(X)
    max = X.max()
    X = X / max
    X = X.float()
    X = torch.sign(X) * (torch.log(1 + u * torch.abs(X)) / torch.log(torch.Tensor([1 + u])))
    X = X.numpy()
    return X


if __name__ == '__main__':
    for exp_iters in range(0, len(seeds)):
        seed = seeds[exp_iters]
        attention_head = 4
        attention_hidden = 64
        learning_rate = 0.001
        Epochs = 5
        BATCH_SIZE = 32

        T_stride = 2
        T_overlop = T_stride / 2
        overlapTime = {
            'neutral': 1,
            'happy': 1,
            'sad': 1,
            'angry': 1,
        }
        FEATURES_TO_USE = 'mfcc'  # {'mfcc' , 'logfbank','fbank','spectrogram','melspectrogram'}

        RATE = 16000
        MODEL_NAME = 'MyModel_2'
        MODEL_PATH = '{}_{}.pth'.format(MODEL_NAME, FEATURES_TO_USE)
        MODEL_PATH = os.path.join(CODE_DIR, "models", MODEL_PATH)

        dict = {
            'neutral': torch.Tensor([0]),
            'happy': torch.Tensor([1]),
            'sad': torch.Tensor([2]),
            'angry': torch.Tensor([3]),
        }
        label_num = {
            'neutral': 0,
            'happy': 0,
            'sad': 0,
            'angry': 0,
        }
        featuresFileName = 'features_{}_{}_{}.pkl'.format(FEATURES_TO_USE,
                                                          impro_or_script,
                                                          seed)
        featuresFileName = os.path.join(CODE_DIR, "features", featuresFileName)
        setup_seed(seed)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件

        log_name = 'train.log'
        logfile = log_name
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        logger.addHandler(fh)

        if (featuresExist == True):
            with open(featuresFileName, 'rb')as f:
                features_to_load = pickle.load(f)
            train_X_features = features_to_load['train_X']
            train_y = features_to_load['train_y']
            valid_features_dict = features_to_load['val_dict']
        else:
            logging.info("creating meta dict...")
            train_X, train_y, val_dict = process_data(WAV_PATH, t=T_stride, train_overlap=T_overlop)
            print(train_X.shape)
            print(len(val_dict))

            print("getting features")
            logging.info('getting features')
            feature_extractor = features_original.FeatureExtractor(rate=RATE)
            train_X_features = feature_extractor.get_features(FEATURES_TO_USE, train_X)
            valid_features_dict = {}
            for _, i in enumerate(val_dict):
                X1 = feature_extractor.get_features(FEATURES_TO_USE, val_dict[i]['X'])
                valid_features_dict[i] = {
                    'X': X1,
                    'y': val_dict[i]['y']
                }
            if (toSaveFeatures == True):
                features_to_save = {'train_X': train_X_features,
                                    'train_y': train_y,
                                    'val_dict': valid_features_dict}
                with open(featuresFileName, 'wb') as f:
                    pickle.dump(features_to_save, f)

        # logging.info("µ-law expansion")
        # train_X_features = process_features(train_X_features)
        # valid_X_features = process_features(valid_X_features)

        for i in train_y:
            label_num[i] += 1
        weight = torch.Tensor([(sum(label_num.values()) - label_num['neutral']) / sum(label_num.values()),
                               (sum(label_num.values()) - label_num['happy']) / sum(label_num.values()),
                               (sum(label_num.values()) - label_num['sad']) / sum(label_num.values()),
                               (sum(label_num.values()) - label_num['angry']) / sum(label_num.values())]).cuda()

        train_data = data_loader_original.DataSet(train_X_features, train_y)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                                  shuffle=True)

        if net_type == "macnn":
            model_train = model_original.MACNN(attention_head,
                                               attention_hidden)
        elif net_type == "macnn_x4":
            model_train = model_original.MACNN4times(attention_head,
                                                     attention_hidden)
        else:
            model_train = model_original.MACNNOneVSAll(attention_head,
                                                       attention_hidden)

        print(f"Number of Params: {count_parameters(model_train)}")
        if torch.cuda.is_available():
            model_train = model_train.cuda()

        # criterion = nn.CrossEntropyLoss(weight=weight)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model_train.parameters(), lr=learning_rate,
                               weight_decay=1e-6)


        logging.info("training...")
        maxWA = 0
        maxUA = 0
        totalrunningTime = 0
        for i in range(Epochs):
            mat = np.zeros((4, 4))
            startTime = time.time()
            tq = tqdm(total=len(train_y))
            model_train.train()
            print_loss = 0
            for _, data in enumerate(train_loader):
                batch_x, batch_y = data
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                batch_out = model_train(batch_x.unsqueeze(1))
                if net_type == "moe":
                    batch_out, n, h, s, a = batch_out
                    new_y = torch.zeros(batch_y.shape[0], 4).cuda()
                    for index in range(new_y.shape[0]):
                        new_y[index, batch_y[index]] = 1
                    n_y = new_y[:, 0]
                    h_y = new_y[:, 1]
                    s_y = new_y[:, 2]
                    a_y = new_y[:, 3]
                    criterion_bce = nn.BCEWithLogitsLoss()

                    loss_n = criterion_bce(n, n_y.unsqueeze(1).cuda())
                    loss_h = criterion_bce(h, h_y.unsqueeze(1).cuda())
                    loss_s = criterion_bce(s, s_y.unsqueeze(1).cuda())
                    loss_a = criterion_bce(a, a_y.unsqueeze(1).cuda())

                loss = criterion(batch_out, batch_y.squeeze(1))
                if net_type == "moe":
                    loss = loss + loss_n + loss_h + loss_s + loss_a
                print_loss += loss.data.item() * BATCH_SIZE

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                y_pred = batch_out.cpu().data.detach().numpy()
                y_pred = np.argmax(y_pred, axis=1)
                y_true = batch_y.cpu().data.detach().numpy()
                for p, val in enumerate(y_pred):
                    if val == y_true[p]:
                        mat[val, val] += 1
                    else:
                        mat[y_true[p], val] += 1
                tq.update(BATCH_SIZE)

            print(mat)
            tq.close()
            print('epoch: {}, loss: {:.4}'.format(i, print_loss / len(train_X_features)))
            logging.info('epoch: {}, loss: {:.4}'.format(i, print_loss))
            if (i > 0 and i % 10 == 0):
                learning_rate = learning_rate / 10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
            # validation
            endTime = time.time()
            totalrunningTime += endTime - startTime
            print(totalrunningTime)
            with torch.no_grad():
                model_train.eval()
                UA = [0, 0, 0, 0]
                num_correct = 0
                class_total = [0, 0, 0, 0]
                matrix = np.mat(np.zeros((4, 4)), dtype=int)
                for _, vfd in enumerate(valid_features_dict):
                    x, y = valid_features_dict[vfd]['X'], valid_features_dict[vfd]['y']
                    x = torch.from_numpy(x).float()
                    y = dict[y[0]].long()
                    if torch.cuda.is_available():
                        x = x.cuda()
                        y = y.cuda()
                    if (x.size(0) == 1):
                        x = torch.cat((x, x), 0)
                    out = model_train(x.unsqueeze(1))
                    if net_type == "moe":
                        out, n, h, s, a = out
                    pred = torch.Tensor([0, 0, 0, 0]).cuda()
                    for j in range(out.size(0)):
                        pred += out[j]
                    pred = pred / out.size(0)
                    pred = torch.max(pred, 0)[1]
                    if (pred == y):
                        num_correct += 1
                    matrix[int(y), int(pred)] += 1

                for col in range(4):
                    for row in range(4):
                        class_total[col] += matrix[col, row]
                    UA[col] = round(matrix[col, col] / class_total[col], 3)
                WA = num_correct / len(valid_features_dict)
                if (maxWA < WA):
                    maxWA = WA
                    torch.save(model_train.state_dict(), MODEL_PATH)
                if (maxUA < sum(UA) / 4):
                    maxUA = sum(UA) / 4
                print('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
                logging.info('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
                print(matrix)
                logging.info(matrix)
        with open(f"/path/to/code/{seed}.txt", "w") as f:
            f.write(f"maxUA: {maxUA}, maxWA: {maxWA}")