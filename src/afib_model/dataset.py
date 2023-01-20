import numpy as np
import torch

class Dataset_ori():
    def __init__(self, data_path, label_path):
        # self.root = root
        self.data_path = data_path
        self.label_path = label_path
        self.dataset, self.labelset = self.build_dataset()
        self.length = self.dataset.shape[0]
        # self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[idx, :]
        step = torch.unsqueeze(step, 0)
        target = self.labelset[idx]
        return step, target

    def build_dataset(self):
        '''get dataset of signal'''

        dataset = np.load(self.data_path)
        labelset = np.load(self.label_path)

        dataset = torch.from_numpy(dataset)
        labelset = torch.from_numpy(labelset)

        return dataset, labelset