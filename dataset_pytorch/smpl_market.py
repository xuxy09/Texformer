from torch.utils.data import Dataset
import os.path as osp
import pickle
import torch
import imageio
import numpy as np


class SMPLMarket(Dataset):
    """
    Enhanced Market-1501 dataset

    """
    def __init__(self, data_dir, train_flag=True, random_pick=True):
        super().__init__()
        self.data_dir = data_dir
        self.random_pick = random_pick

        paths_pkl_path = osp.join(data_dir, 'train_test_img_paths_pid.pkl')   
        with open(paths_pkl_path, 'rb') as f:
            all_paths = pickle.load(f)
        
        if train_flag:
            self.img_paths_dict = all_paths['out_dict_train']
        else:
            self.img_paths_dict = all_paths['out_dict_test']
        
        self.pids = list(self.img_paths_dict.keys())

        # smpl dir
        self.smpl_dir = osp.join(data_dir, 'SMPL_RSC', 'pkl')

        # smpl part_seg dir
        self.smpl_part_seg_dir = osp.join(data_dir, 'SMPL_RSC', 'parts')
        self.smpl_part_seg_mapping = {3:1, 6:2, 1:3, 2:3, 7:4, 8:4, 4:5, 5:5, 9:6, 10:6, 11:7, 12:7}
        
        # part_seg dir
        self.part_seg_dir = osp.join(data_dir, 'part_seg_EANet')

    def __len__(self):
        return len(self.pids)

    def preprocess_img(self, img):
        # input: HxWxC, uint8(0~255)
        img = (img / 255.) * 2 -1
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        return img

    def preprocess_seg(self, seg):
        seg_float = (seg / 7.) * 2 -1
        seg_float = torch.from_numpy(seg_float).float().unsqueeze(0)
        return seg_float

    def preprocess_smpl_seg(self, smpl_seg):
        smpl_seg_long = np.zeros(smpl_seg.shape, dtype=int)
        for k in self.smpl_part_seg_mapping.keys():
            smpl_seg_long[smpl_seg==k] = self.smpl_part_seg_mapping[k]
        smpl_seg_long = torch.from_numpy(smpl_seg_long).long().unsqueeze(0)
        return smpl_seg_long

    def get_coord(self, shape):
        y = np.linspace(-1.0, 1.0, num=shape[0])
        x = np.linspace(-1.0, 1.0, num=shape[1])
        coord_y, coord_x = np.meshgrid(y, x, indexing='ij')
        coord = np.concatenate((coord_y[None], coord_x[None]), axis=0)
        return torch.from_numpy(coord).float()

    def get_data(self, img_path, suffix=''):
        img_name = img_path.split('/')[-1]

        # * read image
        img = imageio.imread(osp.join(self.data_dir, img_path))
        img = self.preprocess_img(img)

        coord = self.get_coord(img.shape[-2:])

        # * smpl
        pkl_path = osp.join(self.smpl_dir, img_name[:-4]+'.pkl')

        with open(pkl_path, 'rb') as f:
            smpl_list = pickle.load(f)
        verts = torch.from_numpy(smpl_list[0])
        cam_t = torch.from_numpy(smpl_list[1])

        # * smpl part seg
        smpl_seg_path = osp.join(self.smpl_part_seg_dir, img_name[:-4]+'.png')
        smpl_seg = imageio.imread(smpl_seg_path)
        smpl_seg = self.preprocess_smpl_seg(smpl_seg)


        # * part_seg
        seg_path = osp.join(self.part_seg_dir, img_path.split('.')[0]+'.png')
        try:
            seg = imageio.imread(seg_path)
        except:
            seg = imageio.imread(seg_path+'.png')
        
        seg_long = torch.from_numpy(seg).long().unsqueeze(0)
        seg_float = self.preprocess_seg(seg)

        sample = {'img'+suffix: img, 
                  'verts'+suffix: verts,
                  'cam_t'+suffix: cam_t,
                  'seg'+suffix: seg_float,
                  'seg_long'+suffix: seg_long,
                  'smpl_seg'+suffix: smpl_seg,
                  'coord'+suffix: coord, 
                  'img_name'+suffix: img_name,
                  }

        return sample
    
    def __getitem__(self, idx):
        pid = self.pids[idx]
        pid_all_paths = self.img_paths_dict[pid]
        if self.random_pick:
            img_path1, img_path2 = np.random.choice(a=pid_all_paths, size=2, replace=False)
        else:
            img_path1, img_path2 = pid_all_paths[:2]

        sample = self.get_data(img_path1, '')
        sample2 = self.get_data(img_path2, '2')
        
        sample.update(sample2)

        return sample

