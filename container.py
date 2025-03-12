import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score

from utilities import general_utilities as util

SEPARATOR\
    ='#######################################################################'


def container_dict(classes=4):
    return {"matrix": np.mat(np.zeros((classes, classes)), dtype=int),
            "best_matrix": np.mat(np.zeros((classes, classes)), dtype=int),
            "num_correct": 0,
            "loss": 0,
            "reported_loss": 0,
            "best_loss": {"WA": 1e3, "F1": 1e3},
            "counter": 0,
            "UA": 0,
            "WA": 0,
            "F1": 0,
            "maxUA": 0,
            "maxWA": 0,
            "maxF1": 0,
            "classes": classes}


class ResultsContainer:
    def __init__(self, config, data_split="Train", use_tsne=False,
                 main_key="main", weights=None, mulitlabel=False):
        self.main_dict = {main_key: container_dict(classes=len(
            config.CLASS_DICT))}
        self.config = config
        if self.config.MIXTUREOFEXPERTS:
            if self.config.FUSION_LEVEL > -1:
                for emo in ["neu", "hap", "sad", "ang"]:
                    self.main_dict[emo] = container_dict(classes=2)
        self.multilabel = mulitlabel
        self.weights = weights
        if mulitlabel:
            self.main_dict["main"]["matrix"] = np.zeros((len(
                config.CLASS_DICT), 2, 2))
            self.main_dict["main"]["best_matrix"] = np.zeros((len(
                config.CLASS_DICT), 2, 2))
            if self.weights is not None:
                self.criterion = nn.BCEWithLogitsLoss(weight=self.weights[0])
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            if self.weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=self.weights[0])
            else:
                self.criterion = nn.CrossEntropyLoss()
        if self.config.MIXTUREOFEXPERTS and weights is not None:
            self.ind_weights = {"neu": self.weights[1][0],
                                "hap": self.weights[1][1],
                                "sad": self.weights[1][2],
                                "ang": self.weights[1][3]}
            if len(self.config.CLASS_DICT) == 7:
                self.ind_weights["sur"] = self.weights[1][4]
                self.ind_weights["dis"] = self.weights[1][5]
                self.ind_weights["fea"] = self.weights[1][6]
        self.sigmoid = nn.Sigmoid()
        self.unique_label = {}
        self.data_split = data_split
        self.prediction_counter = 0
        self.use_tsne = use_tsne
        self.prediction_holder = []

    def get_bce_loss(self, pred, tar, emo):
        if self.weights is not None:
            current_weights = torch.zeros_like(tar)
            for w in range(current_weights.shape[0]):
                current_weights[w] = self.ind_weights[emo][int(tar[w])]

            bce = nn.BCEWithLogitsLoss(weight=current_weights)
        else:
            bce = nn.BCEWithLogitsLoss()
        return bce(pred, tar)

    def checker(self, inpt, label):

        if self.config.MIXTUREOFEXPERTS:
            if self.config.FUSION_LEVEL > -1:
                inpt, neu, hap, sad, ang = inpt
                inpt = [inpt, neu, hap, sad, ang]
                label = [label[0]]
                if self.multilabel and label[0].dim() > 1:
                    for i, emo in enumerate(["neutral", "happy", "sad", "angry"]):
                        temp_zeros = torch.zeros((inpt[0].shape[0], 1))
                        temp_zeros[torch.where(label[0][:, int(
                            self.config.CLASS_DICT[emo])].cpu())] = \
                            torch.Tensor([1])
                        if torch.cuda.is_available():
                            temp_zeros = temp_zeros.cuda()
                        label.append(temp_zeros)
                elif self.multilabel and label[0].dim() == 1:
                    for i in range(len(["neutral", "happy", "sad", "angry"])):
                        temp_label = label[0][i].view(-1)
                        if torch.cuda.is_available():
                            temp_label = temp_label.cuda()
                        label.append([temp_label])
                else:
                    if label[0].dim() > 1:
                        for emo in ["neutral", "happy", "sad", "angry"]:
                            temp_zeros = torch.zeros((inpt[0].shape[0], 1))
                            temp_zeros[torch.where(label[0].cpu() == self.config.CLASS_DICT[emo])[0]] = torch.Tensor([1])
                            if torch.cuda.is_available():
                                temp_zeros = temp_zeros.cuda()
                            label.append(temp_zeros)
                    else:
                        for i in range(len(["neutral", "happy", "sad", "angry"])):
                            if i != int(label[0]):
                                temp_label = torch.Tensor([0.])
                            else:
                                temp_label = torch.Tensor([1.])
                            if torch.cuda.is_available():
                                temp_label = temp_label.cuda()
                            label.append(temp_label)
            else:
                inpt = [inpt]
                label = [label[0]]
        else:
            inpt = [inpt]
            label = [label[0]]
        return inpt, label

    def __updater__(self, inpt, label, squash=False):
        inpt, label = self.checker(inpt, label)

        for i, key in enumerate(self.main_dict):
            if squash:
                if "neu" in key or "hap" in key or "sad" in key or "ang" in\
                        key:
                    inpt[i] = torch.mean(inpt[i], dim=0)
                    if self.multilabel:
                        temp_loss = self.get_bce_loss(inpt[i], label[i][0], key)
                    else:
                        temp_loss = self.get_bce_loss(inpt[i], label[i], key)
                    inpt[i] = self.sigmoid(inpt[i])
                elif key == "spkr" and self.config.SPEAKER_IND:
                    lab = int(label[i].cpu().detach())
                    if lab not in self.unique_label:
                        self.unique_label[lab] = \
                            inpt[-1][0].cpu().detach()

                    self.cosine_sim = nn.CosineSimilarity()
                    temp_loss = torch.mean(self.cosine_sim(
                        inpt[-1].cpu().detach(),
                        self.unique_label[lab].view(1, -1)))
                    temp_loss = torch.abs(temp_loss)
                else:
                    predictions = inpt[i].shape[0]
                    if "average" in self.config.acc_type:
                        if label[i].dim() == 1 and self.multilabel:
                            label[i] = label[i].unsqueeze(0)
                        temp_loss = self.criterion(torch.sum(inpt[i], dim=0).reshape(1, -1)/predictions,
                                                   label[i])
                    else:
                        temp_loss = self.criterion(
                            torch.sum(inpt[i], dim=0).reshape(
                                1, -1)/predictions, label[i].squeeze(0))

                current_batch = 1
            else:
                current_batch = inpt[0].shape[0]
                if "neu" in key or "hap" in key or "sad" in key or "ang" in\
                        key:
                    temp_loss = self.get_bce_loss(inpt[i], label[i], key)
                    inpt[i] = self.sigmoid(inpt[i])
                else:
                    # .squeeze() only works on dimensions of size 1
                    temp_loss = self.criterion(inpt[i], label[i].squeeze(1))
            self.main_dict[key]["loss"] = temp_loss
            self.main_dict[key]["reported_loss"] += \
                temp_loss.cpu().detach().numpy() * current_batch

            self.main_dict[key]["counter"] += current_batch
            if key == "spkr" and self.config.SPEAKER_IND and self.data_split != "Train":
                continue
            else:
                m, n = util.calc_num_correct(
                    inpt[i], label[i],
                    self.main_dict[key]["classes"], squash=squash, key=key,
                    acc_type=self.config.acc_type, multilabel=self.multilabel)

                if self.multilabel and key == "main":
                    m, y, pred = m
                    if len(self.prediction_holder) == 0:
                        self.prediction_holder.append(y)
                        self.prediction_holder.append(pred)
                    else:
                        self.prediction_holder[0] = \
                            np.vstack((self.prediction_holder[0], y))
                        self.prediction_holder[1] = \
                            np.vstack((self.prediction_holder[1], pred))

                self.main_dict[key]["matrix"] += m
                self.main_dict[key]["num_correct"] += n

    def __end_epoch__(self, save_model=False, model_to_save=None):
        for key in self.main_dict:
            self.__averager__(key)
            self.__calc_acc__(key, save_model, model_to_save)

    def __reseter__(self):
        self.prediction_holder = []
        for key in self.main_dict:
            self.__reset__(key)

    def __get_losses__(self):
        loss = 0
        for key in self.main_dict:
            temp_loss = self.main_dict[key]["loss"]
            if key == "main":
                loss = loss + (temp_loss * 1.)
            elif key == "spkr":
                loss = loss + (temp_loss * self.config.ALPHA_SPEAKER)
            elif key == "hap":
                loss = loss + (temp_loss * 1.)
            elif key == "sad":
                loss = loss + (temp_loss * 1.)
            elif key == "ang":
                loss = loss + (temp_loss * 1.)
            elif key == "neu":
                loss = loss + (temp_loss * 1.)
        return loss

    def __getter__(self):
        values = {"maxUA": {},
                  "maxWA": {},
                  "maxF1": {},
                  "best_matrix": {},
                  "best_loss_WA": {},
                  "best_loss_F1": {}}
        for key in self.main_dict:
            values["maxUA"][key] = self.main_dict[key]["maxUA"]
            values["maxWA"][key] = self.main_dict[key]["maxWA"]
            values["maxF1"][key] = self.main_dict[key]["maxF1"]
            values["best_loss_WA"][key] = self.main_dict[key]["best_loss"]["WA"]
            values["best_loss_F1"][key] = self.main_dict[key]["best_loss"]["F1"]
            values["best_matrix"][key] = self.main_dict[key]["best_matrix"]
        return values

    def __reset__(self, key):
        if key == "main" and self.multilabel:
            self.main_dict["main"]["matrix"] = np.zeros((len(
                self.config.CLASS_DICT), 2, 2))
        else:
            self.main_dict[key]["matrix"] = np.mat(
                np.zeros((self.main_dict[key]["classes"],
                          self.main_dict[key]["classes"])), dtype=int)
        self.main_dict[key]["num_correct"] = 0
        self.main_dict[key]["loss"] = 0
        self.main_dict[key]["reported_loss"] = 0
        self.main_dict[key]["counter"] = 0
        self.unique_label = {}
        if "pred_values" in self.main_dict[key]:
            del self.main_dict[key]["pred_values"]
            del self.main_dict[key]["labels"]

    def __averager__(self, key):
        self.main_dict[key]["loss"] /= self.main_dict[key]["counter"]
        self.main_dict[key]["reported_loss"] /= self.main_dict[key]["counter"]

    @staticmethod
    def save_emo_model(model, path_to_save, acc_type, emotion=""):
        extension = acc_type + emotion if len(emotion) > 0 else acc_type
        filename_old = path_to_save.split("/")[-1]
        filename_new = filename_old.replace(".pth", f"_{extension}.pth")
        emo_loc = path_to_save.replace(filename_old, filename_new)
        torch.save(model.state_dict(), emo_loc)

    def __calc_acc__(self, key, save_model=False, model_to_save=None):
        if key == "spkr" and self.config.SPEAKER_IND and self.data_split != \
                "Train":
            if self.main_dict[key]["reported_loss"] < self.main_dict[key][
                    "best_sim"]:
                self.main_dict[key]["best_sim"] = self.main_dict[key][
                    "reported_loss"]
        else:
            # Ground Truth Vertical, Prediction Horizontal
            if self.multilabel and key == "main":
                pass
            else:
                total_y = np.zeros((self.main_dict[key]["matrix"].sum()))
                total_pred = np.zeros((self.main_dict[key]["matrix"].sum()))

            if self.multilabel and key == "main":
                pass
            else:
                total_pred = []
                total_y = []
                for i, d in enumerate(self.main_dict[key]["matrix"].getA()):
                    temp_sum = np.sum(d)
                    temp_total = [i] * temp_sum
                    total_y += temp_total
                    for pred_label, num_preds in enumerate(d):
                        temp_pred = [pred_label] * num_preds
                        total_pred += temp_pred

            if self.multilabel and key == "main":
                UA = [i[1, 1] / np.sum(i[1:]) for i in self.main_dict[key][
                     "matrix"]]
                WA = self.main_dict[key]["num_correct"] / self.main_dict[key]["counter"]
                f1 = f1_score(self.prediction_holder[0],
                              self.prediction_holder[1], average="macro")
            else:
                f1 = f1_score(total_y, total_pred, average="macro")
                UA, WA = util.calc_ua_wa(self.main_dict[key]["matrix"],
                                         self.main_dict[key]["num_correct"],
                                         self.main_dict[key]["counter"],
                                         self.main_dict[key]["classes"])
            self.main_dict[key]["UA"] = np.mean(UA)
            self.main_dict[key]["WA"] = WA
            self.main_dict[key]["F1"] = f1
            if np.mean(UA) > self.main_dict[key]["maxUA"]:
                self.main_dict[key]["maxUA"] = np.mean(UA)
                if save_model:
                    if key == "main":
                        self.save_emo_model(model_to_save[0],
                                            model_to_save[1], "ua")
                    else:
                        if key == "neu" and self.config.SAVE_IND_EXPERTS:
                            self.save_emo_model(
                                model_to_save[0].mod_neutral,
                                model_to_save[1], "ua", "N")
                        elif key == "hap" and self.config.SAVE_IND_EXPERTS:
                            self.save_emo_model(
                                model_to_save[0].mod_happy,
                                model_to_save[1], "ua", "H")
                        elif key == "sad" and self.config.SAVE_IND_EXPERTS:
                            self.save_emo_model(
                                model_to_save[0].mod_sad,
                                model_to_save[1], "ua", "S")
                        elif key == "ang" and self.config.SAVE_IND_EXPERTS:
                            self.save_emo_model(
                                model_to_save[0].mod_angry,
                                model_to_save[1], "ua", "A")

            if WA > self.main_dict[key]["maxWA"]:
                self.main_dict[key]["maxWA"] = WA
                self.main_dict[key]["best_matrix"] = self.main_dict[key]["matrix"]
                self.main_dict[key]["best_loss"]["WA"] = self.main_dict[key][
                    "reported_loss"]
                if save_model:
                    if key == "main":
                        torch.save(model_to_save[0].state_dict(),
                                   model_to_save[1].replace(".pth",
                                                            "_wa.pth"))
                    else:
                        if key == "neu" and self.config.SAVE_IND_EXPERTS:
                            self.save_emo_model(
                                model_to_save[0].mod_neutral,
                                model_to_save[1], "wa", "N")
                        elif key == "hap" and self.config.SAVE_IND_EXPERTS:
                            self.save_emo_model(
                                model_to_save[0].mod_happy,
                                model_to_save[1], "wa", "H")
                        elif key == "sad" and self.config.SAVE_IND_EXPERTS:
                            self.save_emo_model(
                                model_to_save[0].mod_sad,
                                model_to_save[1], "wa", "S")
                        elif key == "ang" and self.config.SAVE_IND_EXPERTS:
                            self.save_emo_model(
                                model_to_save[0].mod_angry,
                                model_to_save[1], "wa", "A")

            if f1 > self.main_dict[key]["maxF1"]:
                self.main_dict[key]["maxF1"] = f1
                self.main_dict[key]["best_matrix"] = self.main_dict[key]["matrix"]
                self.main_dict[key]["best_loss"]["F1"] = self.main_dict[key][
                    "reported_loss"]
                if save_model:
                    if key == "main":
                        self.save_emo_model(model_to_save[0],
                                            model_to_save[1], "f1")
                    else:
                        if key == "neu" and self.config.SAVE_IND_EXPERTS:
                            self.save_emo_model(
                                model_to_save[0].mod_neutral,
                                model_to_save[1], "f1", "N")
                        elif key == "hap" and self.config.SAVE_IND_EXPERTS:
                            self.save_emo_model(
                                model_to_save[0].mod_happy,
                                model_to_save[1], "f1", "H")
                        elif key == "sad" and self.config.SAVE_IND_EXPERTS:
                            self.save_emo_model(
                                model_to_save[0].mod_sad,
                                model_to_save[1], "f1", "S")
                        elif key == "ang" and self.config.SAVE_IND_EXPERTS:
                            self.save_emo_model(
                                model_to_save[0].mod_angry,
                                model_to_save[1], "f1", "A")


    def __printer__(self, epoch):
        print(SEPARATOR)
        print(f'\n{self.data_split}: Emotion Results EPOCH:'
              f' {epoch}:\nUnweighted '
              f'Accuracy: {round(self.main_dict["main"]["UA"], 3)}, Weighted '
              f'Accuracy: {round(self.main_dict["main"]["WA"], 3)}, F1-Score:'
              f' {round(self.main_dict["main"]["F1"], 3)}, loss: '
              f'{round(self.main_dict["main"]["reported_loss"], 3)}\n')
        print(f'Best Unweighted Accuracy:'
              f' {round(self.main_dict["main"]["maxUA"], 3)}, Best Weighted '
              f'Accuracy: {round(self.main_dict["main"]["maxWA"], 3)}, '
              f'Best F1-Score: {round(self.main_dict["main"]["maxF1"], 3)}\n')

        print(f'{self.main_dict["main"]["matrix"]}')
        print(SEPARATOR)

        if self.config.MIXTUREOFEXPERTS:
            if self.config.FUSION_LEVEL > -1:
                print(f'\nThe individual classifier UA performance for '
                      f'\"neutral\", \"happy\", \"sad\", \"angry\": '
                      f'{round(self.main_dict["neu"]["UA"], 3)}, '
                      f'{round(self.main_dict["hap"]["UA"], 3)}, '
                      f'{round(self.main_dict["sad"]["UA"], 3)}, '
                      f'{round(self.main_dict["ang"]["UA"], 3)} ')
                print(f'\nThe individual classifier WA performance for '
                      f'\"neutral\", \"happy\", \"sad\", \"angry\": '
                      f'{round(self.main_dict["neu"]["WA"], 3)}, '
                      f'{round(self.main_dict["hap"]["WA"], 3)}, '
                      f'{round(self.main_dict["sad"]["WA"], 3)}, '
                      f'{round(self.main_dict["ang"]["WA"], 3)} ')
                print(SEPARATOR)
                print(f'\nThe best individual classifier UA performance for '
                      f'\"neutral\", \"happy\", \"sad\", \"angry\": '
                      f'{round(self.main_dict["neu"]["maxUA"], 3)}, '
                      f'{round(self.main_dict["hap"]["maxUA"], 3)}, '
                      f'{round(self.main_dict["sad"]["maxUA"], 3)}, '
                      f'{round(self.main_dict["ang"]["maxUA"], 3)} ')
                print(f'\nThe best individual classifier WA performance for '
                      f'\"neutral\", \"happy\", \"sad\", \"angry\": '
                      f'{round(self.main_dict["neu"]["maxWA"], 3)}, '
                      f'{round(self.main_dict["hap"]["maxWA"], 3)}, '
                      f'{round(self.main_dict["sad"]["maxWA"], 3)}, '
                      f'{round(self.main_dict["ang"]["maxWA"], 3)} ')
                print(SEPARATOR)

    def __logger__(self, logger, epoch):
        logger.info(SEPARATOR)
        logger.info(f'\n{self.data_split}: Emotion Results EPOCH: {epoch}:\nUnweighted '
              f'Accuracy: {round(self.main_dict["main"]["UA"], 3)}, Weighted '
              f'Accuracy: {round(self.main_dict["main"]["WA"], 3)}, '
                    f'F1-Score {round(self.main_dict["main"]["F1"], 3)}, loss: '
              f'{round(self.main_dict["main"]["reported_loss"], 3)}\n')
        logger.info(f'Best Unweighted Accuracy:'
                    f'{round(self.main_dict["main"]["maxUA"], 3)}, '
                    f'Best Weighted Accuracy:'
                    f' {round(self.main_dict["main"]["maxWA"], 3)}, F1-Score: '
                    f'{round(self.main_dict["main"]["maxF1"], 3)}\n')
        logger.info(f'{self.main_dict["main"]["matrix"]}')
        logger.info(SEPARATOR)
        if self.config.MIXTUREOFEXPERTS:
            if self.config.FUSION_LEVEL > -1:
                logger.info(f'\nThe individual classifier UA performance for '
                      f'\"neutral\", \"happy\", \"sad\", \"angry\": '
                      f'{round(self.main_dict["neu"]["UA"], 3)}, '
                      f'{round(self.main_dict["hap"]["UA"], 3)}, '
                      f'{round(self.main_dict["sad"]["UA"], 3)}, '
                      f'{round(self.main_dict["ang"]["UA"], 3)} ')
                logger.info(f'\nThe individual classifier WA performance for '
                      f'\"neutral\", \"happy\", \"sad\", \"angry\": '
                      f'{round(self.main_dict["neu"]["WA"], 3)}, '
                      f'{round(self.main_dict["hap"]["WA"], 3)}, '
                      f'{round(self.main_dict["sad"]["WA"], 3)}, '
                      f'{round(self.main_dict["ang"]["WA"], 3)} ')
                logger.info(SEPARATOR)
                logger.info(f'\nThe best individual classifier UA performance for '
                      f'\"neutral\", \"happy\", \"sad\", \"angry\": '
                      f'{round(self.main_dict["neu"]["maxUA"], 3)}, '
                      f'{round(self.main_dict["hap"]["maxUA"], 3)}, '
                      f'{round(self.main_dict["sad"]["maxUA"], 3)}, '
                      f'{round(self.main_dict["ang"]["maxUA"], 3)} ')
                logger.info(f'\nThe best individual classifier WA performance for '
                      f'\"neutral\", \"happy\", \"sad\", \"angry\": '
                      f'{round(self.main_dict["neu"]["maxWA"], 3)}, '
                      f'{round(self.main_dict["hap"]["maxWA"], 3)}, '
                      f'{round(self.main_dict["sad"]["maxWA"], 3)}, '
                      f'{round(self.main_dict["ang"]["maxWA"], 3)} ')
                logger.info(SEPARATOR)
