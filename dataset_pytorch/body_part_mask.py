import os
import torch
import cv2
import numpy as np
import config
import os.path as osp


class TextureMask:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.part = {
            'face': osp.join(config.uv_mask_path, 'face_mask.png'),
            'hand': osp.join(config.uv_mask_path, 'hand_mask.png'),
            'body': osp.join(config.uv_mask_path, 'body_mask.png'),
            'short_up': osp.join(config.uv_mask_path, 'short_up_mask.jpg'),
            'short_trouser': osp.join(config.uv_mask_path, 'short_trouser_mask.jpg'),
            'left_arm': osp.join(config.uv_mask_path, 'left_arm.png'),
            'left_leg': osp.join(config.uv_mask_path, 'left_leg.png'),
            'right_arm': osp.join(config.uv_mask_path, 'right_arm.png'),
            'right_leg': osp.join(config.uv_mask_path, 'right_leg.png'),
        }

    def get_mask(self, part):
        mask_path = self.part[part]
        mask = cv2.imread(mask_path)

        mask = cv2.resize(mask, self.size)
        mask = mask / 255.
        mask = mask.transpose((2, 0, 1))
        mask = np.expand_dims(mask, 0)
        mask = torch.from_numpy(mask).float()
        return mask

    def get_numpy_mask(self, part):
        mask_path = self.part[part]
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, self.size)
        mask = mask / 255.
        return mask


if __name__ == '__main__':
    masker = TextureMask(size=64)
    mask = masker.get_mask("face")
    cv2.imshow('mask', mask)
    cv2.waitKey()
