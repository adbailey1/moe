import torch
import numpy as np
import random
import logging
import os
import sys
import pickle
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from tqdm import tqdm

import container
from data_processes.data_processing_ravdess import load_fold
from data_processes import data_loader
from torch.utils.data import DataLoader

import model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_logger(log_name, config):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = os.path.join(config.EXP_DIR, log_name)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger


def load_pickle(location):
    with open(location, "rb") as f:
        return pickle.load(f)


def save_pickle(location, data):
    with open(location, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def divider_checker(a, b):
    return a/b if b else -1.


def calc_ua_wa(matrix, num_correct, len_feats_dict, classes=4):
    UA = [0] * classes
    class_total = [0] * classes
    for row in range(classes):
        for col in range(classes):
            class_total[row] += matrix[row, col]
        UA[row] = round(divider_checker(matrix[row, row], class_total[row]), 3)
    WA = num_correct / len_feats_dict
    if -1 in UA:
        del UA[UA.index(-1)]

    return UA, WA


def update_best_scores(UA, WA, maxUA, maxWA, model_train, path):
    if maxWA < WA:
        maxWA = WA
        torch.save(model_train.state_dict(), path)
    if maxUA < sum(UA) / 4:
        maxUA = sum(UA) / 4

    return maxUA, maxWA


def forward_pass_test_data_cmu(config, model_train, feat_dict, test_container):
    model_train.eval()
    with ((((((((torch.no_grad())))))))):
        print("Passing Validation Data")
        comp_dict = {}
        for data in tqdm(feat_dict):
            if len(data) == 3:
                x, y, fol = data
            else:
                raise ValueError(f"Setup is wrong, expected len(data) to be "
                                 f"3 but got {len(data)}")
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            out = model_train(x.unsqueeze(1))

            for indx, f in enumerate(fol):
                f = int(f)
                if f not in comp_dict:
                    if config.MIXTUREOFEXPERTS and config.FUSION_LEVEL > -1:
                        if config.LIKE_IEMOCAP:
                            comp_dict[f] = {"pred": out[0][indx].view(1, -1),
                                            "y": y[indx]}
                        else:
                            comp_dict[f] = {"pred": out[0][indx].view(1, -1),
                                            "y": y[indx].view(1, -1)}
                        comp_dict[f]["N"] = out[1][indx].view(1, -1)
                        comp_dict[f]["H"] = out[2][indx].view(1, -1)
                        comp_dict[f]["S"] = out[3][indx].view(1, -1)
                        comp_dict[f]["A"] = out[4][indx].view(1, -1)
                    else:
                        if config.LIKE_IEMOCAP:
                            comp_dict[f] = {"pred": out[indx].view(1, -1),
                                            "y": y[indx]}
                        else:
                            comp_dict[f] = {"pred": out[indx].view(1, -1),
                                            "y": y[indx].view(1, -1)}
                else:
                    if config.MIXTUREOFEXPERTS and config.FUSION_LEVEL > -1:
                        comp_dict[f]["pred"] = torch.cat((comp_dict[f]["pred"],
                                                          out[0][indx].view(1,
                                                                            -1)),
                                                         dim=0)
                        comp_dict[f]["N"] = torch.cat((comp_dict[f]["N"],
                                                       out[1][indx].view(1,
                                                                         -1)))
                        comp_dict[f]["H"] = torch.cat((comp_dict[f]["H"],
                                                       out[2][indx].view(1,
                                                                         -1)))
                        comp_dict[f]["S"] = torch.cat((comp_dict[f]["S"],
                                                       out[3][indx].view(1,
                                                                         -1)))
                        comp_dict[f]["A"] = torch.cat((comp_dict[f]["A"],
                                                       out[4][indx].view(1,
                                                                         -1)))
                    else:
                        comp_dict[f]["pred"] = torch.cat((comp_dict[f]["pred"],
                                                          out[indx].view(1,
                                                                         -1)),
                                                         dim=0)
        print("Evaluating Validation Data")
        for fol in tqdm(comp_dict):
            if config.MIXTUREOFEXPERTS and config.FUSION_LEVEL > -1:
                predictions = (comp_dict[fol]["pred"], comp_dict[fol]["N"],
                               comp_dict[fol]["H"], comp_dict[fol]["S"],
                               comp_dict[fol]["A"])
            else:
                predictions = comp_dict[fol]["pred"]
            fin_y = (comp_dict[fol]["y"],)
            test_container.__updater__(predictions, fin_y, squash=True)


def forward_pass_test_data_iemocap(config, model_train, feat_dict,
                                   test_container,
                                   mapper=None, multilabel=False):
    model_train.eval()
    with torch.no_grad():
        for key in feat_dict:
            data, label = feat_dict[key]["data"], feat_dict[key]["label"]
            if "spkr" in feat_dict[key]:
                spkr = feat_dict[key]["spkr"]
                if mapper:
                    spkr = torch.LongTensor([mapper[spkr[0]]])
                else:
                    spkr = torch.LongTensor([spkr[0]])

            data = torch.from_numpy(data).float()
            if multilabel:
                temp_label = torch.zeros(len(config.CLASS_DICT))
                if "_" in label[0]:
                    for i in label[0].split("_"):
                        temp_label[int(config.CLASS_DICT[i])] = 1.
                else:
                    temp_label[int(config.CLASS_DICT[label[0]])] = 1.
                label = temp_label
            else:
                if isinstance(label[0], str):
                    label = config.CLASS_DICT[label[0]].long()
                else:
                    label = torch.Tensor([label[0]])
            if "spkr" in feat_dict[key]:
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()
                    spkr = spkr.cuda()
            else:
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()

            if data.size(0) == 1:
                data = torch.cat((data, data), 0)
            out = model_train(data.unsqueeze(1))

            if "spkr" in feat_dict[key]:
                test_container.__updater__(out, (label, spkr), squash=True)
            else:
                test_container.__updater__(out, (label,), squash=True)


def calc_num_correct(out, y, classes, squash=False, key="main",
                     acc_type="average", multilabel=False):
    matrix = np.matrix(np.zeros((classes, classes)), dtype=int)
    num_correct = 0
    if key == "neu" or key == "hap" or key == "sad" or key == "ang":
        pred = torch.round(out)
    else:
        if multilabel and key == "main":
            sig = torch.nn.Sigmoid()
            if squash:
                out = torch.mean(out, dim=0).reshape(1, -1)
            pred = torch.round(sig(out))
            pred = pred.cpu().data.numpy()
            y = y.cpu().data.numpy()
            num_correct = accuracy_score(y, pred)
            num_correct = num_correct * y.shape[0]
            matrix = multilabel_confusion_matrix(y, pred)
            return (matrix, y, pred), num_correct
        else:
            if squash:
                if acc_type == "average_no_softmax":
                    pred = torch.argmax(torch.sum(out, dim=0)).reshape(-1, 1)
                elif acc_type == "average_softmax":
                    soft = torch.nn.Softmax(dim=-1)
                    out = torch.mean(soft(out), dim=0)
                    pred = torch.argmax(out).reshape(-1, 1)
                else:
                    temp_zeros = torch.zeros((out.shape[0]), classes)
                    if torch.cuda.is_available():
                        temp_zeros = temp_zeros.cuda()
                    for i in range(out.shape[0]):
                        arg_prediction = torch.argmax(out[i])
                        temp_zeros[i, arg_prediction] = 1
                    temp_zeros = torch.mean(temp_zeros, dim=0)
                    pred = torch.argmax(temp_zeros).reshape(-1, 1)
            else:
                pred = torch.argmax(out, dim=1)

    for i, p in enumerate(pred):
        if p == y[i]:
            num_correct = num_correct + 1
        # Ground Truth Vertical, Prediction Horizontal
        matrix[int(y[i]), int(p)] += 1

    return matrix, num_correct



def setup_model(config):
    if config.MODEL_TYPE == "MACNN":
        if config.MIXTUREOFEXPERTS:
            if config.FUSION_LEVEL == 0:
                model_train = model.MACNNMoE(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden,
                    fc_bias=config.FC_BIAS,
                    skip_final_fc=config.SKIP_FINAL_FC,
                    act=config.ACTIVATION)
            elif config.FUSION_LEVEL == 1:
                model_train = model.MACNNMoEFF1(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden,
                    fusion_level=config.FUSION_LEVEL,
                    skip_final_fc=config.SKIP_FINAL_FC)
            elif config.FUSION_LEVEL == 2:
                model_train = model.MACNNMoEFF2(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden,
                    fusion_level=config.FUSION_LEVEL,
                    skip_final_fc=config.SKIP_FINAL_FC)
            elif config.FUSION_LEVEL == 3:
                model_train = model.MACNNMoEFF3(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden,
                    fusion_level=config.FUSION_LEVEL,
                    skip_final_fc=config.SKIP_FINAL_FC)
            elif config.FUSION_LEVEL == 4:
                model_train = model.MACNNMoEFF4(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden,
                    fusion_level=config.FUSION_LEVEL,
                    skip_final_fc=config.SKIP_FINAL_FC)
            elif config.FUSION_LEVEL == 5:
                model_train = model.MACNNMoEFF5(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden,
                    fusion_level=config.FUSION_LEVEL,
                    skip_final_fc=config.SKIP_FINAL_FC)
            elif config.FUSION_LEVEL == 6:
                model_train = model.MACNNMoEFF6(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden,
                    fusion_level=config.FUSION_LEVEL,
                    skip_final_fc=config.SKIP_FINAL_FC)
            elif config.FUSION_LEVEL == -7:
                model_train = model.MACNNMoEBF7(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden)
            elif config.FUSION_LEVEL == -6:
                model_train = model.MACNNMoEBF6(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden,
                    fusion_level=config.FUSION_LEVEL)
            elif config.FUSION_LEVEL == -5:
                model_train = model.MACNNMoEBF5(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden,
                    fusion_level=config.FUSION_LEVEL)
            elif config.FUSION_LEVEL == -4:
                model_train = model.MACNNMoEBF4(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden,
                    fusion_level=config.FUSION_LEVEL)
            elif config.FUSION_LEVEL == -3:
                model_train = model.MACNNMoEBF3(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden,
                    fusion_level=config.FUSION_LEVEL)
            elif config.FUSION_LEVEL == -2:
                model_train = model.MACNNMoEBF2(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden,
                    fusion_level=config.FUSION_LEVEL)
        else:
            if config.FUSION_LEVEL == -1:
                model_train = model.MACNN4timesParams(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden)
            else:
                model_train = model.MACNN(
                    attention_heads=config.attention_head,
                    attention_hidden=config.attention_hidden,
                    num_emo=len(config.CLASS_DICT))
    elif config.MODEL_TYPE == "LightSERNet":
        if config.MIXTUREOFEXPERTS:
            if config.FUSION_LEVEL == 0:
                model_train = model.LightSERNetMoE(
                    config.SKIP_FINAL_FC
                )
            elif config.FUSION_LEVEL == 1:
                model_train = model.LightSERNetMoEFF1(
                    config.SKIP_FINAL_FC
                )
            elif config.FUSION_LEVEL == 2:
                model_train = model.LightSERNetMoEFF2(
                    config.SKIP_FINAL_FC
                )
            elif config.FUSION_LEVEL == 3:
                model_train = model.LightSERNetMoEFF3(
                    config.SKIP_FINAL_FC
                )
            elif config.FUSION_LEVEL == 4:
                model_train = model.LightSERNetMoEFF4(
                    config.SKIP_FINAL_FC
                )
            elif config.FUSION_LEVEL == 5:
                model_train = model.LightSERNetMoEFF5(
                    config.SKIP_FINAL_FC
                )
            elif config.FUSION_LEVEL == 6:
                model_train = model.LightSERNetMoEFF6(
                    config.SKIP_FINAL_FC
                )
            elif config.FUSION_LEVEL == -7:
                model_train = model.LightSERNetMoEBF7()
            elif config.FUSION_LEVEL == -6:
                model_train = model.LightSERNetMoEBF6()
            elif config.FUSION_LEVEL == -5:
                model_train = model.LightSERNetMoEBF5()
            elif config.FUSION_LEVEL == -4:
                model_train = model.LightSERNetMoEBF4()
            elif config.FUSION_LEVEL == -3:
                model_train = model.LightSERNetMoEBF3()
            elif config.FUSION_LEVEL == -2:
                model_train = model.LightSERNetMoEBF2()
        else:
            if config.FUSION_LEVEL == 22:
                model_train = model.LightSERNetX4()
            else:
                model_train = model.LightSERNet()

    return model_train


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_weight(config):
    weight = []
    for emo in config.CLASS_DICT:
        current_weight = (sum(config.LABEL_NUM.values()) -
            config.LABEL_NUM[emo]) / sum(config.LABEL_NUM.values())
        weight.append(current_weight)
    weight = torch.Tensor(weight)
    if torch.cuda.is_available():
        weight = weight.cuda()

    if config.MIXTUREOFEXPERTS:
        moe_weights = []
        for emo in config.LABEL_NUM:
            emo_weight = (sum(config.LABEL_NUM.values()) -
                config.LABEL_NUM[emo]) / sum(config.LABEL_NUM.values())
            rest_weight = config.LABEL_NUM[emo] / \
                          sum(config.LABEL_NUM.values())
            current_weight = torch.Tensor([rest_weight, emo_weight])
            if torch.cuda.is_available():
                current_weight = current_weight.cuda()
            moe_weights.append(current_weight)

        return weight, moe_weights

    return weight


def print_res(config, ua, wa, f1, loss_wa, loss_f1, data_type="Validation"):
    print(f"Total UA: {ua}")
    print(f"Total WA: {wa}")
    print(f"Total F1: {f1}")
    print(f"Total Loss-WA: {loss_wa}")
    print(f"Total Loss-F1: {loss_f1}")
    logger = make_logger(os.path.join(config.EXP_DIR, f"{data_type}.log"),
                         config)
    logger.info(f"Total UA: {ua}")
    logger.info(f"Total WA: {wa}")
    logger.info(f"Total F1: {f1}")
    logger.info(f"Total Loss-WA: {loss_wa}")
    logger.info(f"Total Loss-F1: {loss_f1}")

    print(f'Emotion Average UA: {np.round(np.mean(ua["main"]), 4)}')
    print(f'Emotion Average WA: {np.round(np.mean(wa["main"]), 4)}')
    print(f'Emotion Average F1: {np.round(np.mean(f1["main"]), 4)}')
    print(f'Emotion Average Loss-WA: {np.round(np.mean(loss_wa["main"]), 4)}')
    print(f'Emotion Average Loss-F1: {np.round(np.mean(loss_f1["main"]), 4)}')

    if config.MIXTUREOFEXPERTS:
        if config.FUSION_LEVEL > -1:
            print(f'Average UA (neu, hap, sad, ang):'
                  f' {np.round(np.mean(ua["neu"]), 4)}, '
                  f'{np.round(np.mean(ua["hap"]), 4)}, '
                  f'{np.round(np.mean(ua["sad"]), 4)}, '
                  f'{np.round(np.mean(ua["ang"]), 4)}')
            print(f'Average WA (neu, hap, sad, ang): '
                  f'{np.round(np.mean(wa["neu"]), 4)}, '
                  f'{np.round(np.mean(wa["hap"]), 4)}, '
                  f'{np.round(np.mean(wa["sad"]), 4)}, '
                  f'{np.round(np.mean(wa["ang"]), 4)}')
            print(f'Average Loss-WA (neu, hap, sad, ang): '
                  f'{np.round(np.mean(loss_wa["neu"]), 4)}, '
                  f'{np.round(np.mean(loss_wa["hap"]), 4)}, '
                  f'{np.round(np.mean(loss_wa["sad"]), 4)}, '
                  f'{np.round(np.mean(loss_wa["ang"]), 4)}')
            print(f'Average Loss-F1 (neu, hap, sad, ang): '
                  f'{np.round(np.mean(loss_f1["neu"]), 4)}, '
                  f'{np.round(np.mean(loss_f1["hap"]), 4)}, '
                  f'{np.round(np.mean(loss_f1["sad"]), 4)}, '
                  f'{np.round(np.mean(loss_f1["ang"]), 4)}')

    logger.info(f'Emotion Average UA: {np.round(np.mean(ua["main"]), 4)}')
    logger.info(f'Emotion Average WA: {np.round(np.mean(wa["main"]), 4)}')
    logger.info(f'Emotion Average F1: {np.round(np.mean(f1["main"]), 4)}')
    logger.info(f'Emotion Average Loss-WA:'
                f' {np.round(np.mean(loss_wa["main"]), 4)}')
    logger.info(f'Emotion Average Loss-F1:'
                f' {np.round(np.mean(loss_f1["main"]), 4)}')
    if config.MIXTUREOFEXPERTS:
        if config.FUSION_LEVEL > -1:
            logger.info(f'Average UA (neu, hap, sad, ang):'
                  f' {np.round(np.mean(ua["neu"]), 4)}, '
                  f'{np.round(np.mean(ua["hap"]), 4)},'
                  f'{np.round(np.mean(ua["sad"]), 4)},'
                  f'{np.round(np.mean(ua["ang"]), 4)}')
            logger.info(f'Average WA (neu, hap, sad, ang): '
                  f'{np.round(np.mean(wa["neu"]), 4)}, '
                  f'{np.round(np.mean(wa["hap"]), 4)},'
                  f'{np.round(np.mean(wa["sad"]), 4)},'
                  f'{np.round(np.mean(wa["ang"]), 4)}')
            logger.info(f'Average Loss-WA (neu, hap, sad, ang): '
                  f'{np.round(np.mean(loss_wa["neu"]), 4)}, '
                  f'{np.round(np.mean(loss_wa["hap"]), 4)}, '
                  f'{np.round(np.mean(loss_wa["sad"]), 4)}, '
                  f'{np.round(np.mean(loss_wa["ang"]), 4)}')
            logger.info(f'Average Loss-F1 (neu, hap, sad, ang): '
                  f'{np.round(np.mean(loss_f1["neu"]), 4)}, '
                  f'{np.round(np.mean(loss_f1["hap"]), 4)}, '
                  f'{np.round(np.mean(loss_f1["sad"]), 4)}, '
                  f'{np.round(np.mean(loss_f1["ang"]), 4)}')


def feature_checker(config):
    if config.SPEAKER_IND:
        if not os.path.exists(
                os.path.join(config.FEATURE_LOC,
                             f"session_0_{config.FEATURESFILENAME}")):
            features_exist = False
        else:
            features_exist = True
    else:
        if config.LIKE_IEMOCAP and config.DATASET == "cmu":
            if not config.EMO_THRESHOLD:
                raise ValueError("EMO_THRESHOLD not set")
            else:
                if not os.path.exists(os.path.join(
                        config.FEATURE_LOC,
                        "LIKE_IEMOCAP_"+str(config.EMO_THRESHOLD),
                        f"fold_0_{config.FEATURESFILENAME}")):
                    features_exist = False
                else:
                    features_exist = True
        else:
            if config.EMO_THRESHOLD:
                if not os.path.exists(os.path.join(config.FEATURE_LOC, str(config.EMO_THRESHOLD),
                                                   f"fold_0_{config.FEATURESFILENAME}")):
                    features_exist = False
                else:
                    features_exist = True
            else:
                if not os.path.exists(os.path.join(config.FEATURE_LOC,
                                                   f"fold_0_{config.FEATURESFILENAME}")):
                    features_exist = False
                else:
                    features_exist = True

    return features_exist


def append_scores(config, total_scores_ua, total_scores_wa, total_scores_f1,
                  total_scores_loss_wa, total_scores_loss_f1, best_scores):
    total_scores_ua["main"].append(best_scores["maxUA"]["main"])
    total_scores_wa["main"].append(best_scores["maxWA"]["main"])
    total_scores_f1["main"].append(best_scores["maxF1"]["main"])
    total_scores_loss_wa["main"].append(best_scores["best_loss_WA"]["main"])
    total_scores_loss_f1["main"].append(best_scores["best_loss_F1"]["main"])
    if config.MIXTUREOFEXPERTS:
        if config.FUSION_LEVEL > -1:
            total_scores_ua["neu"].append(best_scores["maxUA"]["neu"])
            total_scores_wa["neu"].append(best_scores["maxWA"]["neu"])
            total_scores_loss_wa["neu"].append(
                best_scores["best_loss_WA"]["neu"])
            total_scores_loss_f1["neu"].append(
                best_scores["best_loss_F1"]["neu"])
            total_scores_ua["hap"].append(best_scores["maxUA"]["hap"])
            total_scores_wa["hap"].append(best_scores["maxWA"]["hap"])
            total_scores_loss_wa["hap"].append(
                best_scores["best_loss_WA"]["hap"])
            total_scores_loss_f1["hap"].append(
                best_scores["best_loss_F1"]["hap"])
            total_scores_ua["sad"].append(best_scores["maxUA"]["sad"])
            total_scores_wa["sad"].append(best_scores["maxWA"]["sad"])
            total_scores_loss_wa["sad"].append(
                best_scores["best_loss_WA"]["sad"])
            total_scores_loss_f1["sad"].append(
                best_scores["best_loss_F1"]["sad"])
            total_scores_ua["ang"].append(best_scores["maxUA"]["ang"])
            total_scores_wa["ang"].append(best_scores["maxWA"]["ang"])
            total_scores_loss_wa["ang"].append(
                best_scores["best_loss_WA"]["ang"])
            total_scores_loss_f1["ang"].append(
                best_scores["best_loss_F1"]["ang"])

    return total_scores_ua, total_scores_wa, total_scores_f1, \
           total_scores_loss_wa, total_scores_loss_f1


def save_info_for_checkpoint(seed, epoch,
                             train_container, test_container, save_dir,
                             model, optimizer, cuda, train_loader,
                             total_scores_ua, total_scores_wa, total_scores_f1,
                             total_scores_loss_wa, total_scores_loss_f1):
    save_location_model_data = os.path.join(save_dir, "model_info.pth")
    save_location_data = os.path.join(save_dir, "data_info.pkl")
    if len(optimizer) > 1:
        if cuda:
            save_out_dict = {'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer[0].state_dict(),
                             'optimizer_N': optimizer[1].state_dict(),
                             'optimizer_H': optimizer[2].state_dict(),
                             'optimizer_S': optimizer[3].state_dict(),
                             'optimizer_A': optimizer[4].state_dict(),
                             'rng_state': torch.get_rng_state(),
                             'cuda_rng_state': torch.cuda.get_rng_state(),
                             'numpy_rng_state': np.random.get_state(),
                             'random_rng_state': random.getstate()}
        else:
            save_out_dict = {'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer[0].state_dict(),
                             'optimizer_N': optimizer[1].state_dict(),
                             'optimizer_H': optimizer[2].state_dict(),
                             'optimizer_S': optimizer[3].state_dict(),
                             'optimizer_A': optimizer[4].state_dict(),
                             'rng_state': torch.get_rng_state(),
                             'numpy_rng_state': np.random.get_state(),
                             'random_rng_state': random.getstate()}
    else:
        if cuda:
            save_out_dict = {'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer[0].state_dict(),
                             'rng_state': torch.get_rng_state(),
                             'cuda_rng_state': torch.cuda.get_rng_state(),
                             'numpy_rng_state': np.random.get_state(),
                             'random_rng_state': random.getstate()}
        else:
            save_out_dict = {'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer[0].state_dict(),
                             'rng_state': torch.get_rng_state(),
                             'numpy_rng_state': np.random.get_state(),
                             'random_rng_state': random.getstate()}
    save_data_dict = {"epoch": epoch,
                      "seed": seed,
                      "train_container": train_container,
                      "test_container": test_container,
                      "train_loader": train_loader,
                      "total_scores_ua": total_scores_ua,
                      "total_scores_wa": total_scores_wa,
                      "total_scores_f1": total_scores_f1,
                      "total_scores_loss_wa": total_scores_loss_wa,
                      "total_scores_loss_f1": total_scores_loss_f1
    }

    torch.save(save_out_dict, save_location_model_data)
    with open(save_location_data, "wb") as f:
        pickle.dump(save_data_dict, f)


def load_checkpoint_info(pth):
    with open(pth, "rb") as f:
        checkpoint_data = pickle.load(f)

    return checkpoint_data


def load_model(checkpoint_path, model, cuda, optimizer=None):
    """
    Loads the model weights along with the current epoch and all the random
    states that are used during the experiment. Also loads the current state
    of the data loader for continuity
    Inputs:
        checkpoint_path: Location of the saved model
        model: The model from current experiment
        optimizer: The current optimiser state
        cuda: bool - Set True to use GPU (set in initial arguments)
    Outputs:
        epoch_iter: Current epoch
        data_saver: Holds information regarding the data loader so that it
            can be restored from a checkpoint. This includes the
            current pointer of ones and zeros and the current list of
            indexes of the ones and zeros
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    if optimizer is not None:
        if len(optimizer) > 1:
            optimizer[0].load_state_dict(checkpoint['optimizer'])
            optimizer[1].load_state_dict(checkpoint['optimizer_N'])
            optimizer[2].load_state_dict(checkpoint['optimizer_H'])
            optimizer[3].load_state_dict(checkpoint['optimizer_S'])
            optimizer[4].load_state_dict(checkpoint['optimizer_A'])
        else:
            optimizer[0].load_state_dict(checkpoint['optimizer'])
    torch.set_rng_state(checkpoint['rng_state'])
    if cuda:
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    random.setstate(checkpoint['random_rng_state'])


def create_test_feats(feat_dict):
    valid_folders = [i for i in feat_dict for j in
                     feat_dict[i]["y"]]
    valid_x_features = np.zeros((len(valid_folders),
                                 feat_dict[list(feat_dict.keys())[0]]["x"].shape[1],
                                 feat_dict[list(feat_dict.keys())[0]]["x"].shape[2]))
    valid_y = []
    counter = 0
    for i in feat_dict:
        temp_data = feat_dict[i]["x"]
        length = temp_data.shape[0]
        valid_x_features[counter:counter + length, :, :] = temp_data
        valid_y = valid_y + feat_dict[i]["y"]
        counter += length

    return valid_x_features, valid_y, valid_folders


def get_fold_score(config, seed, model_path, exp_dir):
    if config.DATASET == "cmu":
        like_iemocap = config.LIKE_IEMOCAP
        if like_iemocap:
            multilabel = False
        else:
            multilabel = True
        test_container_wa = container.ResultsContainer(config,
                                                       mulitlabel=multilabel)
        test_container_ua = container.ResultsContainer(config,
                                                       mulitlabel=multilabel)
        test_container_f1 = container.ResultsContainer(config,
                                                       mulitlabel=multilabel)
        tc = [test_container_wa, test_container_ua, test_container_f1]
        for exp_fold in range(config.NUM_FOLDS - 1):
            feat_dict = load_fold(config, exp_fold)
            data = feat_dict["val_dict"]
            test_X_features, test_y, test_folders = create_test_feats(data)
            test_data = data_loader.DataSet(config=config,
                                            X=test_X_features,
                                            Y=test_y, folder=test_folders,
                                            multilabel=multilabel)

            test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE,
                                     shuffle=False)

            # MODEL_PATH = f"fold_{exp_fold}_seed_{seed}" \
            #              f"_{config.FEATURES_TO_USE}.pth"
            # MODEL_PATH = os.path.join(exp_dir, MODEL_PATH)
            MODEL_PATH_WA = f"fold_{exp_fold}_seed_{seed}" \
                            f"_{config.FEATURES_TO_USE}_wa.pth"
            MODEL_PATH_WA = os.path.join(exp_dir, MODEL_PATH_WA)
            MODEL_PATH_UA = f"fold_{exp_fold}_seed_{seed}" \
                            f"_{config.FEATURES_TO_USE}_ua.pth"
            MODEL_PATH_UA = os.path.join(exp_dir, MODEL_PATH_UA)
            MODEL_PATH_F1 = f"fold_{exp_fold}_seed_{seed}" \
                            f"_{config.FEATURES_TO_USE}_f1.pth"
            MODEL_PATH_F1 = os.path.join(exp_dir, MODEL_PATH_F1)

            test_model = setup_model(config)
            for p, m in enumerate(
                    [MODEL_PATH_WA, MODEL_PATH_UA, MODEL_PATH_F1]):
                test_model.load_state_dict(torch.load(m))

                if torch.cuda.is_available():
                    test_model = test_model.cuda()
                test_model.eval()

                if config.MTL and not config.SPEAKER_IND:
                    save_map = model_path[:-4] + "_mapper.pkl"
                    with open(save_map, "rb") as f:
                        mapper = pickle.load(f)
                else:
                    mapper = None

                forward_pass_test_data_cmu(config=config,
                                           model_train=test_model,
                                           feat_dict=test_loader,
                                           test_container=tc[p],
                                           mapper=mapper,
                                           multilabel=multilabel)

        for p, m in enumerate([MODEL_PATH_WA, MODEL_PATH_UA, MODEL_PATH_F1]):
            tc[p].__end_epoch__(save_model=False,
                                model_to_save=(test_model, model_path))
            temp_scores = tc[p].__getter__()
            if p == 0:
                current_best_scores = temp_scores
            elif p == 1:
                current_best_scores["maxUA"] = temp_scores["maxUA"]
            else:
                current_best_scores["maxF1"] = temp_scores["maxF1"]
        return current_best_scores
    else:
        # comp_matrix = np.mat(np.zeros((4, 4)), dtype=int)
        # comp_num_correct = comp_len_feat_dict = 0
        test_container_wa = container.ResultsContainer(config)
        test_container_ua = container.ResultsContainer(config)
        test_container_f1 = container.ResultsContainer(config)
        tc = [test_container_wa, test_container_ua, test_container_f1]
        for exp_fold in range(config.NUM_FOLDS - 1):
            feat_dict = load_fold(config, exp_fold)
            data = feat_dict["val_dict"]

            MODEL_PATH_WA = f"fold_{exp_fold}_seed_{seed}" \
                            f"_{config.FEATURES_TO_USE}_wa.pth"
            MODEL_PATH_WA = os.path.join(config.EXP_DIR, f"seed_{seed}",
                                         MODEL_PATH_WA)
            MODEL_PATH_UA = f"fold_{exp_fold}_seed_{seed}" \
                            f"_{config.FEATURES_TO_USE}_ua.pth"
            MODEL_PATH_UA = os.path.join(config.EXP_DIR, f"seed_{seed}",
                                         MODEL_PATH_UA)
            MODEL_PATH_F1 = f"fold_{exp_fold}_seed_{seed}" \
                            f"_{config.FEATURES_TO_USE}_f1.pth"
            MODEL_PATH_F1 = os.path.join(config.EXP_DIR, f"seed_{seed}",
                                         MODEL_PATH_F1)

            test_model = setup_model(config)
            for p, m in enumerate([MODEL_PATH_WA, MODEL_PATH_UA, MODEL_PATH_F1]):
                # if os.path.exists(m.replace(".pth", "N.pth")):
                #     if not config.SKIP_FINAL_FC:
                #         test_model.load_state_dict(torch.load(m))
                #     test_model.mod_neutral.load_state_dict(
                #         torch.load(m.replace(".pth", "N.pth")))
                #     test_model.mod_happy.load_state_dict(
                #         torch.load(m.replace(".pth", "H.pth")))
                #     test_model.mod_sad.load_state_dict(
                #         torch.load(m.replace(".pth", "S.pth")))
                #     test_model.mod_angry.load_state_dict(
                #         torch.load(m.replace(".pth", "A.pth")))
                # else:
                test_model.load_state_dict(torch.load(m))

                if torch.cuda.is_available():
                    test_model = test_model.cuda()
                test_model.eval()

                mapper = None

                forward_pass_test_data_iemocap(config=config,
                                               model_train=test_model,
                                               feat_dict=data,
                                               test_container=tc[p],
                                               mapper=mapper)
        for p, m in enumerate([MODEL_PATH_WA, MODEL_PATH_UA, MODEL_PATH_F1]):
            tc[p].__end_epoch__(save_model=False,
                                model_to_save=(test_model, model_path))

            temp_scores = tc[p].__getter__()
            if p == 0:
                current_best_scores = temp_scores
            elif p == 1:
                current_best_scores["maxUA"] = temp_scores["maxUA"]
            else:
                current_best_scores["maxF1"] = temp_scores["maxF1"]
        return current_best_scores
