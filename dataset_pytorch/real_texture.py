import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from dataset_pytorch.data_utils import ToTensor

import tqdm


class RealTextureDataset(Dataset):


    def __getitem__(self, index):
        texture_img_path = self.data[index]
        texture_img = cv2.imread(texture_img_path)
        texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)

        texture_img = cv2.resize(texture_img, dsize=(self.img_size, self.img_size))

        texture_img = self.to_tensor(texture_img)

        return texture_img

    def __len__(self):
        return len(self.data)

    def __init__(self, data_path, img_size=64, normalize=True):
        self.data_path = data_path
        self.img_size = img_size
        self.normalize = normalize
        self.to_tensor = ToTensor(normalize=self.normalize)
        self.data = []
        self.generate_index()

    def generate_index(self):
        print('generating index')
        for root, dirs, files in os.walk(self.data_path):
            for name in tqdm.tqdm(files):
                if name.endswith('.jpg') and 'nongrey' in name:
                    self.data.append(os.path.join(root, name))

        print('finish generating index, found texture image: {}'.format(len(self.data)))


