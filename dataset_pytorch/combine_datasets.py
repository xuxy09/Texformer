from random import sample
from torch.utils.data import Dataset
import random

class Combine(Dataset):
    def __init__(self, dataset_list, main_idx=0, random=True):
        super().__init__()
        self.dataset_list = dataset_list
        self.main_idx = main_idx
        self.random = random

    def __len__(self):
        return len(self.dataset_list[self.main_idx])

    def __getitem__(self, idx):
        sample_list = []
        for ds_idx in range(len(self.dataset_list)):
            dataset = self.dataset_list[ds_idx]

            if ds_idx == self.main_idx:
                sample_list.append(dataset[idx])
            else:
                this_idx = random.randrange(len(dataset)) if self.random else (idx % len(dataset))
                sample_list.append(dataset[this_idx])
        
        return sample_list


    