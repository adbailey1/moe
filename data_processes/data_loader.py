import torch
from torch.utils.data import Dataset


class DataSet(Dataset):
    """
    Class to obtain batched data and target
    """
    def __init__(self, data, targets, class_dict, folder=None):
        self.data = data
        self.targets = targets
        self.folder = folder
        self.class_dict = class_dict

    def __getitem__(self, index):
        """
        Used to get data for batches. To be used alongside PyTorch's dataloader
        Args:
            index: int: selects data for batch at this location

        Returns:
            data: numpy.array
            targets: int
            fol (opt): int

        """
        data = self.data[index]

        data = torch.from_numpy(data)
        data = data.float()

        target = torch.tensor([self.targets[index]])

        if self.folder:
            fol = self.folder[index]
            return data, target, fol
        else:
            return data, target

    def __len__(self):
        return len(self.data)
