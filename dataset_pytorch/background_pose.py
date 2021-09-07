import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from dataset_pytorch.data_utils import ToTensor, RandomCrop, RandomFlip, Resize, Resize_pose

import tqdm


class BackgroundDataset(Dataset):

    def __getitem__(self, index):
        texture_img_path = self.data[index]
        texture_img = cv2.imread(texture_img_path)
        texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)
        
        
        texture_img = self.resize(texture_img)
        
        if self.random:
            texture_img = self.random_flip(texture_img)
        
        texture_img = self.to_tensor(texture_img)
        return texture_img

    def __len__(self):
        return len(self.data)

    def __init__(self, data_path_list, img_size=(128, 64), normalize=True, random=True):
        self.data_path_list = data_path_list
        self.img_size = img_size
        self.normalize = normalize
        self.to_tensor = ToTensor(normalize=self.normalize)
        self.data = []
        self.generate_index()
        self.random = random

        self.random_crop = RandomCrop(output_size=self.img_size)
        self.random_flip = RandomFlip(flip_prob=0.5)
        self.resize = Resize_pose(output_size=img_size)

    def generate_index(self):
        print('generating background index')
        for data_path in self.data_path_list:
            for root, dirs, files in os.walk(data_path):
                for name in tqdm.tqdm(files):
                    if name.endswith('.jpg'):
                        self.data.append(os.path.join(root, name))

        print('finish generating background index, found texture image: {}'.format(len(self.data)))

