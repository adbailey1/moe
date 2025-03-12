import torch
from torch.utils.data import Dataset
import numpy as np


class DataSet(Dataset):
    def __init__(self, config, data, labels, folder=None, spkr_id=None,
                 multilabel=False):
        self.data = data
        self.labels = labels
        self.folder = folder
        classes = []
        for i in self.labels:
            if i not in classes:
                classes.append(i)
        self.multilabel = multilabel
        if self.multilabel:
            self.label_holder_dict = {}
        self.spkr_id = spkr_id
        self.map = {}
        if spkr_id is not None:
            self.mapper()
        self.class_dict = config.CLASS_DICT
        self.mean, self.std = self.calculate_stats(self.data)

    def calculate_stats(self, x):
        """
        Calculates the mean and the standard deviation of the input

        Input:
            x: Input data array

        Outputs
            mean: The mean of the input
            standard_deviation: The standard deviation of the input
        """
        if x.ndim == 1:
            mean = np.mean(x)
            standard_deviation = np.std(x)
            return mean, standard_deviation
        elif x.ndim == 2:
            axis = 0
        elif x.ndim == 3:
            axis = (0, 2)
        elif x.ndim == 4:
            axis = (0, 1, 3)

        mean = np.mean(x, axis=axis)
        mean = np.reshape(mean, (-1, 1))

        standard_deviation = np.std(x, axis=axis)
        standard_deviation = np.reshape(standard_deviation, (-1, 1))

        return mean, standard_deviation

    def mapper(self):
        """
        Map each speaker to a value in sequential order. This is useful for
        speaker independent setups as speaker 2, 3 may be held out and so
        we want to assign speaker 4, 5 to the values 2, 3 - loss
        calculations and performances are then maintained.
        :return:
        """
        counter = 0
        for value in self.spkr_id:
            if value not in self.map:
                self.map[value] = counter
                counter += 1

    def __getitem__(self, index):
        data = self.data[index]
        if self.folder:
            folder = self.folder[index]
        data = torch.from_numpy(data)
        data = data.float()
        if self.multilabel:
            if index not in self.label_holder_dict:
                label = torch.zeros(len(self.class_dict))
                if "_" in self.labels[index]:
                    for i in self.labels[index].split("_"):
                        label[int(self.class_dict[i])] = 1.
                else:
                    label[int(self.class_dict[self.labels[index]])] = 1.
                self.label_holder_dict[index] = label
            else:
                label = self.label_holder_dict[index]
        else:
            label = self.labels[index]
            if isinstance(label, str):
                label = self.class_dict[label]
                label = label.long()
        if self.spkr_id is not None:
            temp_spkr_id = self.map[self.spkr_id[index]]
            spkr_id = torch.LongTensor([temp_spkr_id])
            if self.folder:
                return data, label, spkr_id, folder
            else:
                return data, label, spkr_id
        else:
            if self.folder:
                return data, label, folder
            else:
                return data, label

    def __len__(self):
        return len(self.data)
