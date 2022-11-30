import numpy as np
import torch
import torch.nn as nn

from utilities import general_utilities as util

SEPARATOR\
    ='#######################################################################'


def container_dict(classes=4):
    """ Setup the dictionary to use in this container file

    Args:
        classes: int - the number of classes in the dataset

    Returns:
        dict holding relevant keys to be used to hold results during training
    """
    return {"matrix": np.mat(np.zeros((classes, classes)), dtype=int),
            "best_matrix": np.mat(np.zeros((classes, classes)), dtype=int),
            "num_correct": 0,
            "loss": 0,
            "reported_loss": 0,
            "best_loss": 1e3,
            "counter": 0,
            "ua": 0,
            "wa": 0,
            "max_ua": 0,
            "max_wa": 0,
            "classes": classes}


class ResultsContainer:
    def __init__(self, config, class_weights, data_split="Train",
                 use_gpu=False, use_weights=False):
        self.main_dict = {"main": container_dict(classes=len(
            config.CLASS_DICT))}
        self.config = config
        self.use_weights = use_weights
        if data_split == "Train":
            self.class_weights = class_weights[0]
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.ce = nn.CrossEntropyLoss()
        if self.config.MoE:
            if self.config.FUSION_LEVEL > -1:
                for emo in list(self.config.CLASS_DICT):
                    self.main_dict[emo] = container_dict(classes=2)
                if data_split == "Train":
                    self.expert_weights = class_weights[1]

        self.ce_no_weight = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.data_split = data_split
        self.use_gpu = use_gpu

    def get_bce_loss(self, predicted, targets, current_emo):
        current_weights = torch.zeros_like(targets)
        for w in range(current_weights.shape[0]):
            current_weights[w] = self.expert_weights[
                self.config.CLASS_DICT[current_emo]][int(targets[w])]

        bce = nn.BCEWithLogitsLoss(weight=current_weights)

        return bce(predicted, targets)

    def checker(self, inpt, label):
        """
        Sorts the output of the model (and corresponding label) into
        iterable format and re-organises/re-shapes the data for use with
        loss functions

        Args:
            inpt: tuple - the output of the model (if using Mixture of
                Experts, the complete prediciton will be output as well as the
                individual expert outputs)
            label: tuple - the corresponding labels to the output of the model

        Returns:
            inpt: list - the output of the model
            label: list - corresponding labels to the output of the model

        """
        if self.config.MoE:
            if self.config.FUSION_LEVEL > -1:
                inpt, neu, hap, sad, ang = inpt
                inpt = [inpt, neu, hap, sad, ang]
                label = [label[0]]
                if label[0].dim() > 1:
                    for emo in list(self.config.CLASS_DICT):
                        temp_zeros = torch.zeros((inpt[0].shape[0], 1))
                        temp_zeros[torch.where(label[0].cpu() ==
                                               self.config.CLASS_DICT[
                            emo])[0]] = torch.Tensor([1])
                        if self.use_gpu:
                            temp_zeros = temp_zeros.cuda()
                        label.append(temp_zeros)
                else:
                    for i in range(len(list(self.config.CLASS_DICT))):
                        if i != int(label[0]):
                            temp_label = torch.Tensor([0.])
                        else:
                            temp_label = torch.Tensor([1.])
                        if self.use_gpu:
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
        """ Updates the container dictionary w.r.t. losses / metrics

        Args:
            inpt: tuple - the output of the model (if using Mixture of
                Experts, the complete prediciton will be output as well as the
                individual expert outputs)
            label: tuple - the corresponding labels to the outputs of the model
            squash: Bool - set True to average the predicted results,
                this is used in the validation/test phase as we make
                predictions for a complete and single utterance only,
                therefore we average the model output

        Returns: None

        """
        inpt, label = self.checker(inpt=inpt, label=label)

        for i, key in enumerate(self.main_dict):
            if squash:
                inpt[i] = torch.mean(inpt[i], dim=0).view(1, -1)
                if key == "main":
                    temp_loss = self.ce_no_weight(inpt[i], label[i])
                else:
                    temp_loss = self.bce(inpt[i], label[i].unsqueeze(1))
                    inpt[i] = self.sigmoid(inpt[i])
                current_batch = 1
            else:
                current_batch = inpt[0].shape[0]
                if key == "main":
                    temp_loss = self.ce(inpt[i], label[i].squeeze(1))
                else:
                    temp_loss = self.get_bce_loss(predicted=inpt[i],
                                                  targets=label[i],
                                                  current_emo=key)
                    inpt[i] = self.sigmoid(inpt[i])

            self.main_dict[key]["loss"] = temp_loss
            self.main_dict[key]["reported_loss"] += \
                temp_loss.cpu().detach().numpy() * current_batch

            self.main_dict[key]["counter"] += current_batch

            m, n = util.calc_num_correct(
                pred=inpt[i], target=label[i],
                classes=self.main_dict[key]["classes"], squash=squash, key=key)

            self.main_dict[key]["matrix"] += m
            self.main_dict[key]["num_correct"] += n

    def __end_epoch__(self, save_model=False, model_to_save=None):
        """ Average current loss and calculate the accuracy for the epoch

        Args:
            save_model: Bool - set True to save the current model
            model_to_save: tuple - (model to save, path to save location)

        Returns: None

        """
        for key in self.main_dict:
            self.main_dict[key]["reported_loss"] /= \
                self.main_dict[key]["counter"]
            self.__calc_acc__(key, save_model, model_to_save)

    def __resetter__(self):
        """
        Iterates through the keys in the container and resets scores,
        counters and losses

        Returns: None

        """
        for key in self.main_dict:
            self.__reset__(key)

    def __get_losses__(self):
        """ Gets the loss for the current batch

        Returns:
            loss: torch.tensor - the loss for the batch

        """
        loss = 0
        for key in self.main_dict:
            temp_loss = self.main_dict[key]["loss"]
            if key == "main":
                loss = loss + (temp_loss * 1.)
            else:
                loss = loss + (temp_loss * 1.)
        return loss

    def __getter__(self):
        """ Get the best results recorded so far

        Returns:
            dict: holds the best ua/wa/loss/matrix for each of the keys in
                the container
        """
        values = {"max_ua": {},
                  "max_wa": {},
                  "best_matrix": {},
                  "best_loss": {}}
        for key in self.main_dict:
            values["max_ua"][key] = self.main_dict[key]["max_ua"]
            values["max_wa"][key] = self.main_dict[key]["max_wa"]
            values["best_loss"][key] = self.main_dict[key]["best_loss"]
            values["best_matrix"][key] = self.main_dict[key]["best_matrix"]
        return values

    def __reset__(self, key):
        """ Sets several records to default values for starting a new epoch

        Args:
            key: str - dictionary key for container (set to "main" for general
                result holder and set to the respective emotions for Mixture of
                Expert operation)

        Returns: None

        """
        self.main_dict[key]["matrix"] = np.mat(
            np.zeros((self.main_dict[key]["classes"],
                      self.main_dict[key]["classes"])), dtype=int)
        self.main_dict[key]["num_correct"] = 0
        self.main_dict[key]["loss"] = 0
        self.main_dict[key]["reported_loss"] = 0
        self.main_dict[key]["counter"] = 0

    @staticmethod
    def save_emo_model(model, path_to_save, acc_type):
        """ Sorts out the save location and file name to save a model

        Args:
            model: nn.Module - the model to save
            path_to_save: str - the path to save the model
            acc_type: str - used for the extension set to "ua" or "wa"

        Returns: None

        """
        extension = acc_type
        filename_old = path_to_save.split("/")[-1]
        filename_new = filename_old.replace(".pth", f"_{extension}.pth")
        emo_loc = path_to_save.replace(filename_old, filename_new)
        torch.save(model.state_dict(), emo_loc)

    def __calc_acc__(self, key, save_model=False, model_to_save=None):
        """
        Calculates the ua and wa for the current results held in the
        container. If the calculated results are better than the "best"
        results, update and save the current model.

        Args:
            key: str - dictionary key for container (set to "main" for general
                result holder and set to the respective emotions for Mixture of
                Expert operation)
            save_model: Bool - set True to save the current model
            model_to_save: tuple - (model to save, path to save location)

        Returns: None

        """
        # Ground Truth Vertical, Prediction Horizontal
        total_pred = []
        total_y = []
        for i, d in enumerate(self.main_dict[key]["matrix"].getA()):
            temp_sum = np.sum(d)
            temp_total = [i] * temp_sum
            total_y += temp_total
            for pred_label, num_preds in enumerate(d):
                temp_pred = [pred_label] * num_preds
                total_pred += temp_pred

        ua, wa = util.calc_ua_wa(
            matrix=self.main_dict[key]["matrix"],
            num_correct=self.main_dict[key]["num_correct"],
            len_feats_dict=self.main_dict[key]["counter"],
            classes=self.main_dict[key]["classes"])

        self.main_dict[key]["ua"] = np.mean(ua)
        self.main_dict[key]["wa"] = wa
        if np.mean(ua) > self.main_dict[key]["max_ua"]:
            self.main_dict[key]["max_ua"] = np.mean(ua)
            if save_model:
                if key == "main":
                    self.save_emo_model(model=model_to_save[0],
                                        path_to_save=model_to_save[1],
                                        acc_type="ua")

        if wa > self.main_dict[key]["max_wa"]:
            self.main_dict[key]["max_wa"] = wa
            self.main_dict[key]["best_matrix"] = self.main_dict[key]["matrix"]
            self.main_dict[key]["best_loss"] = \
                self.main_dict[key]["reported_loss"]
            if save_model:
                if key == "main":
                    torch.save(model_to_save[0].state_dict(),
                               model_to_save[1].replace(".pth", "_wa.pth"))

    def __printer__(self, epoch):
        """ Print the results for the current epoch and print the best results

        Args:
            epoch: int - the current epoch

        Returns: None

        """
        print(SEPARATOR)
        print(f'\n{self.data_split}: Emotion Results EPOCH:'
              f' {epoch}:\nUnweighted Accuracy:'
              f' {round(self.main_dict["main"]["ua"], 3)}, '
              f'Weighted Accuracy: {round(self.main_dict["main"]["wa"], 3)}, '
              f'loss: {round(self.main_dict["main"]["reported_loss"], 3)}\n')
        print(f'Best Unweighted Accuracy:'
              f' {round(self.main_dict["main"]["max_ua"], 3)}, '
              f'Best Weighted Accuracy:'
              f' {round(self.main_dict["main"]["max_wa"], 3)}\n')

        print(f'{self.main_dict["main"]["matrix"]}')
        print(SEPARATOR)
        if self.config.MoE:
            if self.config.FUSION_LEVEL > -1:
                emos = list(self.config.CLASS_DICT)
                print(f'\nThe individual classifier UA performance for '
                      f'\"{emos[0]}\", \"{emos[1]}\", \"{emos[2]}\", '
                      f'\"{emos[3]}\": '
                      f'{round(self.main_dict[emos[0]]["ua"], 3)}, '
                      f'{round(self.main_dict[emos[1]]["ua"], 3)}, '
                      f'{round(self.main_dict[emos[2]]["ua"], 3)}, '
                      f'{round(self.main_dict[emos[3]]["ua"], 3)} ')
                print(f'\nThe individual classifier WA performance for '
                      f'\"{emos[0]}\", \"{emos[1]}\", \"{emos[2]}\", '
                      f'\"{emos[3]}\": '
                      f'{round(self.main_dict[emos[0]]["wa"], 3)}, '
                      f'{round(self.main_dict[emos[1]]["wa"], 3)}, '
                      f'{round(self.main_dict[emos[2]]["wa"], 3)}, '
                      f'{round(self.main_dict[emos[3]]["wa"], 3)} ')
                print(SEPARATOR)
                print(f'\nThe best individual classifier UA performance for '
                      f'\"{emos[0]}\", \"{emos[1]}\", \"{emos[2]}\", '
                      f'\"{emos[3]}\": '
                      f'{round(self.main_dict[emos[0]]["max_ua"], 3)}, '
                      f'{round(self.main_dict[emos[1]]["max_ua"], 3)}, '
                      f'{round(self.main_dict[emos[2]]["max_ua"], 3)}, '
                      f'{round(self.main_dict[emos[3]]["max_ua"], 3)} ')
                print(f'\nThe best individual classifier WA performance for '
                      f'\"{emos[0]}\", \"{emos[1]}\", \"{emos[2]}\", '
                      f'\"{emos[3]}\": '
                      f'{round(self.main_dict[emos[0]]["max_wa"], 3)}, '
                      f'{round(self.main_dict[emos[1]]["max_wa"], 3)}, '
                      f'{round(self.main_dict[emos[2]]["max_wa"], 3)}, '
                      f'{round(self.main_dict[emos[3]]["max_wa"], 3)} ')
                print(SEPARATOR)

    def __logger__(self, logger, epoch):
        """ Log the results for the current epoch and print the best results

        Args:
            logger: logger - to record the results
            epoch: int - the current epoch

        Returns: None

        """
        logger.info(SEPARATOR)
        logger.info(f'\n{self.data_split}: Emotion Results EPOCH:'
                    f' {epoch}:\nUnweighted Accuracy: '
                    f'{round(self.main_dict["main"]["ua"], 3)}, '
                    f'Weighted Accuracy: '
                    f'{round(self.main_dict["main"]["wa"], 3)}, '
                    f'loss: '
                    f'{round(self.main_dict["main"]["reported_loss"], 3)}\n')
        logger.info(f'Best Unweighted Accuracy:'
                    f' {round(self.main_dict["main"]["max_ua"], 3)}, '
                    f'Best Weighted Accuracy: '
                    f'{round(self.main_dict["main"]["max_wa"], 3)}\n')

        logger.info(f'{self.main_dict["main"]["matrix"]}')
        logger.info(SEPARATOR)
        if self.config.MoE:
            if self.config.FUSION_LEVEL > -1:
                emos = list(self.config.CLASS_DICT)
                logger.info(f'\nThe individual classifier UA performance for '
                            f'\"{emos[0]}\", \"{emos[1]}\", \"{emos[2]}\", '
                            f'\"{emos[3]}\": '
                            f'{round(self.main_dict[emos[0]]["ua"], 3)}, '
                            f'{round(self.main_dict[emos[1]]["ua"], 3)}, '
                            f'{round(self.main_dict[emos[2]]["ua"], 3)}, '
                            f'{round(self.main_dict[emos[3]]["ua"], 3)} ')
                logger.info(f'\nThe individual classifier WA performance for '
                            f'\"{emos[0]}\", \"{emos[1]}\", \"{emos[2]}\", '
                            f'\"{emos[3]}\": '
                            f'{round(self.main_dict[emos[0]]["wa"], 3)}, '
                            f'{round(self.main_dict[emos[1]]["wa"], 3)}, '
                            f'{round(self.main_dict[emos[2]]["wa"], 3)}, '
                            f'{round(self.main_dict[emos[3]]["wa"], 3)} ')
                logger.info(SEPARATOR)
                logger.info(f'\nThe best individual classifier UA performance '
                            f'for \"{emos[0]}\", \"{emos[1]}\", \
                            "{emos[2]}\", \"{emos[3]}\": '
                            f'{round(self.main_dict[emos[0]]["max_ua"], 3)}, '
                            f'{round(self.main_dict[emos[1]]["max_ua"], 3)}, '
                            f'{round(self.main_dict[emos[2]]["max_ua"], 3)}, '
                            f'{round(self.main_dict[emos[3]]["max_ua"], 3)} ')
                logger.info(f'\nThe best individual classifier WA performance '
                            f'for \"{emos[0]}\", \"{emos[1]}\", \
                            "{emos[2]}\", \"{emos[3]}\": '
                            f'{round(self.main_dict[emos[0]]["max_wa"], 3)}, '
                            f'{round(self.main_dict[emos[1]]["max_wa"], 3)}, '
                            f'{round(self.main_dict[emos[2]]["max_wa"], 3)}, '
                            f'{round(self.main_dict[emos[3]]["max_wa"], 3)} ')
                logger.info(SEPARATOR)
