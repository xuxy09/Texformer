import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from easydict import EasyDict

import config
from NMR.neural_render_test import NrTextureRenderer
from loss.part_style_loss import PartStyleLoss
from loss.PCB_PerLoss import ReIDLoss
from loss.pytorch_ssim import SSIM

from dataset_pytorch.smpl_market import SMPLMarket
from dataset_pytorch.background_pose import BackgroundDataset
from dataset_pytorch.real_texture import RealTextureDataset
from dataset_pytorch.combine_datasets import Combine
from dataset_pytorch.body_part_mask import TextureMask
from transformers.texformer import Texformer


class Trainer:
    def __init__(self, opts):
        self.opts = opts
        self.device = opts.device
        self.src_size = (128, 64)    # input human image size
        self.uv_size = 128    # default
        
        # * data
        self.background_dataset = BackgroundDataset([config.PRW_img_path, config.CUHK_SYSU_path], img_size=self.src_size)
        self.surreal_dataset = RealTextureDataset(data_path=config.surreal_texture_path, img_size=self.uv_size)
        self.train_dataset = SMPLMarket(config.market1501_dir, train_flag=True, random_pick=True)
        
        self.combined_dataset = Combine([self.train_dataset, self.background_dataset, self.surreal_dataset], random=True)
        self.combined_dataloader = DataLoader(self.combined_dataset, batch_size=opts.batch_size, 
                                              shuffle=True, num_workers=opts.num_workers, drop_last=True)

        self.test_dataset = SMPLMarket(config.market1501_dir, train_flag=False, random_pick=False)
        
        self.combined_dataset_test = Combine([self.test_dataset, self.background_dataset, self.surreal_dataset], random=False)
        self.combined_dataloader_test = DataLoader(self.combined_dataset_test, batch_size=opts.batch_size, 
                                                   shuffle=False, num_workers=opts.num_workers, drop_last=True)

        self.tgt = torch.from_numpy(np.load(config.uv_encoding_path)).permute(2, 0, 1)[None]
        self.tgt = (self.tgt * 2 -1).float().to(self.device)

        # mask for face & hand
        texture_mask = TextureMask(size=self.uv_size)  # load mask with uv_size x uv_size
        self.face_mask = texture_mask.get_mask('face').to(self.device)
        self.hand_mask = texture_mask.get_mask('hand').to(self.device)
        self.mask = self.face_mask + self.hand_mask

        # * model
        self.model = Texformer(opts)
        self.model.to(self.device)
        
        self.renderer = NrTextureRenderer(render_res=self.src_size[0], device=self.device)     

        self.reid_loss = ReIDLoss(config.reid_weight_path, device=self.device, normalize_feat=opts.reid_norm_feat, permute=opts.permute)
        self.face_loss = torch.nn.MSELoss()
        self.ssim_fn = SSIM(window_size=11, size_average=True)
        self.part_style_loss = PartStyleLoss(7, False, None)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=opts.lr)

        # * others
        self.summary_writer = SummaryWriter(opts.summary_dir)
        self.loss_name_list = ['loss_reid', 'loss_part_style', 'loss_face_structure', 'loss_reid2', 'loss_part_style2']
        self.show_img_dict = {'concat': ['img', 'seg', 'rendered_img', 'img2', 'seg2', 'rendered_img2'],
                              'uvmap': ['uvmap', 'combine_mask']}
        self.var = EasyDict()    # to store intermediate variables

    def train(self):
        self.model.train()
        self.step_count = 0

        for epoch_idx in tqdm(range(self.opts.num_epochs)):
            for batch in tqdm(self.combined_dataloader, desc=f'Epoch{epoch_idx}'):
                # run model
                self.forward_step(batch)
                loss = self.step_output['loss_final']      

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.step_count += 1

                # summary
                if self.step_count % self.opts.summary_steps == 0:
                    self.train_summaries() 
            
            if (epoch_idx+1) % self.opts.save_epochs == 0:
                torch.save(self.model.state_dict(), '{}/ep{:03d}_step{:06d}.pt'.format(self.opts.checkpoint_dir, epoch_idx+1, self.step_count))
    
    def show_img(self, idx=0, to_numpy=False):
        img_dict = {}

        for k in self.show_img_dict:
            img_name_list = self.show_img_dict[k]
            img_list = []
            
            for name in img_name_list:
                tmp = self.step_output[name]
                if isinstance(tmp, torch.Tensor):
                    tmp = tmp[idx].detach().cpu()
                    if tmp.shape[0] == 1:
                        tmp = tmp.expand(3, -1, -1)
                    if tmp.min() < 0:
                        tmp = (tmp + 1) / 2.
                    tmp = tmp.clamp(0, 1)
                    img_list.append(tmp)
            
            cat_img = torch.cat(img_list, dim=2)
            if to_numpy:
                cat_img = (cat_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_dict[k] = cat_img

        return img_dict

    def train_summaries(self):
        loss_names = self.loss_name_list

        img_dict = self.show_img(idx=0, to_numpy=False)

        # add image
        for k in img_dict:
            self.summary_writer.add_image(k, img_dict[k], self.step_count)

        # add scalar
        for loss_name in loss_names:
            val = self.step_output[loss_name]
            self.summary_writer.add_scalar(loss_name, val, self.step_count)

    def set_input_data(self, batch):
        sample = batch[0]
        self.var.background_image_batch = batch[1].to(self.device)
        self.var.real_texture_batch = batch[2].to(self.device)

        self.var.img = sample['img'].to(self.device)
        self.var.verts = sample['verts'].to(self.device)
        self.var.cam_t = sample['cam_t'].to(self.device)
        self.var.seg = sample['seg'].to(self.device)
        self.var.seg_long = sample['seg_long'].to(self.device)
        self.var.smpl_seg = sample['smpl_seg'].to(self.device)
        self.var.smpl_seg_float = (self.var.smpl_seg.float() / 7.) * 2 -1
        self.var.coord = sample['coord'].to(self.device)
        self.var.img_name = sample['img_name']

        self.var.img2 = sample['img2'].to(self.device)
        self.var.verts2 = sample['verts2'].to(self.device)
        self.var.cam_t2 = sample['cam_t2'].to(self.device)
        self.var.seg2 = sample['seg2'].to(self.device)
        self.var.seg_long2 = sample['seg_long2'].to(self.device)
        self.var.smpl_seg2 = sample['smpl_seg2'].to(self.device)
        self.var.coord2 = sample['coord2'].to(self.device)

    def generate_uvmap(self, img, seg, coord):
        src = torch.cat([img, seg], dim=1)  # Key

        tgt = self.tgt.expand(src.shape[0], -1, -1, -1)  # Query

        if not self.opts.mask_fusion:
            value = coord if self.opts.out_type == 'flow' else img
        else:
            value = torch.cat([coord, img], dim=1)
        out = self.model(tgt, src, value)
        
        # generate uvmap
        combine_mask = 0
        if not self.opts.mask_fusion:
            if self.opts.out_type == 'flow':
                uvmap = torch.nn.functional.grid_sample(img, out.permute(0, 2, 3, 1))
            elif self.opts.out_type == 'rgb':
                uvmap = out
        else:
            combine_mask = out[2]
            uvmap_flow = torch.nn.functional.grid_sample(img, out[0].permute(0, 2, 3, 1))
            uvmap_rgb = out[1]
            uvmap = uvmap_flow * combine_mask + uvmap_rgb * (1-combine_mask)

        return uvmap, combine_mask
    
    def render_img(self, verts, cam_t, uvmap, background_image_batch):
        rendered_img, depth, mask = self.renderer.render(verts, cam_t, uvmap, crop_width=self.src_size[0]-self.src_size[1])
        mask = mask.unsqueeze(1)
        generated_img_batch = rendered_img * mask + background_image_batch * (1 - mask)
        generated_img_batch = generated_img_batch.contiguous()   
        return rendered_img, generated_img_batch
    
    def compute_face_structure_loss(self, uvmap, real_texture_batch):
        if self.opts.face_structure_loss_weight != 0:
            uvmap_face_hand = uvmap * self.mask
            real_face_hand_batch = real_texture_batch * self.mask
            loss_face_structure = 1-self.ssim_fn(uvmap_face_hand, real_face_hand_batch, only_structure=1)
        else:
            loss_face_structure = 0
        return loss_face_structure
    
    def compute_part_style_loss(self, features, seg_long, smpl_seg):
        if self.opts.part_style_loss_weight != 0 and features is not None:
            layer_idx = 0
            loss_part_style = self.part_style_loss(features[0][layer_idx], features[1][layer_idx], smpl_seg, seg_long)   # only the features of layer1 is used!
        else:
            loss_part_style = 0
        return loss_part_style
    
    def compute_reid_loss(self, generated_img_batch, img):
        if self.opts.reid_loss_weight != 0:
            loss_reid, features = self.reid_loss(generated_img_batch, img)
        else:
            loss_reid = 0
            features = None
        return loss_reid, features

    def compute_img_loss(self, generated_img_batch, img, seg_long, smpl_seg, coeff_part_style=1):
        loss_reid, features = self.compute_reid_loss(generated_img_batch, img)
        loss_part_style = self.compute_part_style_loss(features, seg_long, smpl_seg)

        loss = self.opts.reid_loss_weight * loss_reid + \
               coeff_part_style * self.opts.part_style_loss_weight * loss_part_style 
        return loss, loss_reid, loss_part_style

    def forward_step(self, batch):
        # get input data
        self.set_input_data(batch)

        # run model --> generate uvmap
        seg = self.var.seg 
        uvmap, combine_mask = self.generate_uvmap(self.var.img, seg, self.var.coord)

        # render image --> compute image loss, pose from 1, texture from 1
        rendered_img1, generated_img_batch1 = self.render_img(self.var.verts, self.var.cam_t, uvmap, self.var.background_image_batch)
        loss1, loss_reid1, loss_part_style1 = self.compute_img_loss(generated_img_batch1, self.var.img, self.var.seg_long, self.var.smpl_seg)
        # render image --> compute image loss, pose from 2, texture from 1
        rendered_img2, generated_img_batch2 = self.render_img(self.var.verts2, self.var.cam_t2, uvmap, self.var.background_image_batch)
        loss2, loss_reid2, loss_part_style2 = self.compute_img_loss(generated_img_batch2, self.var.img2, self.var.seg_long2, self.var.smpl_seg2)

        # loss face&hand structure 
        loss_face_structure = self.compute_face_structure_loss(uvmap, self.var.real_texture_batch)

        # total loss
        loss = loss1 + \
               self.opts.mv_loss_weight * loss2 + \
               self.opts.face_structure_loss_weight * loss_face_structure 

        # output for backprop and logging 
        self.step_output = {'uvmap': uvmap, 'rendered_img': rendered_img1, 'rendered_img2': rendered_img2, 'combine_mask': combine_mask,
                            'loss': loss1, 'loss_reid': loss_reid1, 'loss_part_style': loss_part_style1,
                            'loss2': loss2, 'loss_reid2': loss_reid2, 'loss_part_style2': loss_part_style2,
                            'loss_final': loss, 'loss_face_structure': loss_face_structure,}
        self.step_output.update(self.var)


if __name__ == '__main__':
    from options import TrainOptions
    opts = TrainOptions().parse_args()
    trainer = Trainer(opts)

    trainer.train()
