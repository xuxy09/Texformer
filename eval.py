""" Evaluation on the SMPLMarket dataset

"""
import torch
import numpy as np
from tqdm import tqdm
import config
from transformers.texformer import Texformer
from NMR.neural_render_test import NrTextureRenderer
from dataset_pytorch.smpl_market_eval import SMPLMarket
from dataset_pytorch.background_pose import BackgroundDataset
from loss.PCB_PerLoss import ReIDLoss
from loss.pytorch_ssim import SSIM
import lpips
from reid_resnet.main import ReIDModel


class Tester:
    def __init__(self, opts):
        self.device = 'cuda'
        self.checkpoint_path = opts.checkpoint_path
        self.opts = opts 
        
        self.model = Texformer(opts)
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.model.eval()
        self.model.to(self.device)

        self.tgt = torch.from_numpy(np.load(config.uv_encoding_path)).permute(2, 0, 1)[None]
        self.tgt = (self.tgt * 2 -1).float().to(self.device)

        self.renderer_numerical = NrTextureRenderer(render_res=128, device=self.device, factor=1)

        # * test dataset
        self.background_dataset = BackgroundDataset([config.PRW_img_path, config.CUHK_SYSU_path], img_size=(128, 64), random=False)
        self.test_dataset = SMPLMarket(config.market1501_dir)

        # * test metrics
        self.ssim = SSIM(window_size=11, size_average=True)
        self.lpips = lpips.LPIPS(pretrained=True, net='alex')
        self.lpips.to(self.device)
        self.reid = ReIDLoss(config.reid_weight_path, normalize_feat=True, permute=0)
        self.reid_resnet = ReIDModel()
    
    def forward_step(self, sample):
        img = sample['img'].to(self.device)[None]
        verts = sample['verts'].to(self.device)[None]
        cam_t = sample['cam_t'].to(self.device)[None]
        seg = sample['seg'].to(self.device)[None]
        seg_long = sample['seg_long'].to(self.device)[None]
        smpl_seg = sample['smpl_seg'].to(self.device)[None]
        smpl_seg_float = (smpl_seg.float() / 7.) * 2. -1
        coord = sample['coord'].to(self.device)[None]
        
        # ---------- foward ------------
        src = torch.cat([img, seg], dim=1)
        tgt = self.tgt.expand(src.shape[0], -1, -1, -1)
        value = torch.cat([coord, img], dim=1)
        out = self.model(tgt, src, value)
        
        # generate uvmap
        combine_mask = out[2]
        tex_flow = out[0]
        uvmap_flow = torch.nn.functional.grid_sample(img, out[0].permute(0, 2, 3, 1))
        uvmap_rgb = out[1]
        uvmap = uvmap_flow * combine_mask + uvmap_rgb * (1-combine_mask)

        return verts, cam_t, uvmap, combine_mask, tex_flow, uvmap_rgb, uvmap_flow

    @torch.no_grad()
    def generate_numerical_results(self):
        self.ssim_list = []
        self.lpips_list = []
        self.cossim_list = []
        self.cossimR_list = []

        for i, sample in tqdm(enumerate(self.test_dataset), total=len(self.test_dataset)):
            # if i % 100 == 0:
            #     print(i, len(self.test_dataset))
            
            img_name = sample['img_name']
            verts, cam_t, uvmap, combine_mask, tex_flow, uvmap_rgb, uvmap_flow = self.forward_step(sample)
            rendered_img, depth, mask = self.renderer_numerical.render(verts, cam_t, uvmap, crop_width=64)
            
            rendered_img = rendered_img.clamp(-1, 1)
            uvmap = uvmap.clamp(-1, 1)
            result = ((rendered_img[0].cpu().permute(1, 2, 0).numpy()+1)*0.5*255).astype(np.uint8)
            mask = (mask[0].cpu().numpy()*255).astype(np.uint8)
            gt = ((sample['img'].permute(1, 2, 0).numpy() + 1) * 0.5 * 255).astype(np.uint8)
            uvmap = ((uvmap[0].cpu().permute(1, 2, 0).numpy()+1)*0.5*255).astype(np.uint8)
            background = self.background_dataset[i % len(self.background_dataset)]

            self.eval_metrics(result, mask, gt, background)

        print('+'*6 + ' Summary ' + '+'*6)
        print('CosSim: {:.4f}'.format(np.mean(self.cossim_list)))
        print('CosSim-R: {:.4f}'.format(np.mean(self.cossimR_list)))
        print('SSIM: {:.4f}'.format(np.mean(self.ssim_list)))
        print('LPIPS: {:.4f}'.format(np.mean(self.lpips_list)))

    def eval_metrics(self, result, mask, gt, background):
        result = (result / 255.0) * 2 - 1
        mask = mask / 255.0
        gt = gt / 255.0 * 2 -1

        result = torch.from_numpy(result).permute(2, 0, 1)[None].float().to(self.device)
        mask = torch.from_numpy(mask)[None, None].float().to(self.device)
        gt = torch.from_numpy(gt).permute(2, 0, 1)[None].float().to(self.device)
        background = background.to(self.device)
        
        self.ssim_list.append(self.compute_ssim(result*mask, gt*mask))
        self.lpips_list.append(self.compute_LPIPS(result*mask, gt*mask))
        self.cossim_list.append(self.compute_reid(result*mask+background*(1-mask), gt))
        self.cossimR_list.append(self.compute_reid_resnet(result*mask+background*(1-mask), gt))

    def compute_ssim(self, img, gt, mask=None):
        v = self.ssim(img, gt, mask)
        v = v.cpu().item()
        return v

    def compute_LPIPS(self, img, gt):
        v = self.lpips(img, gt)
        v = v.cpu().item()
        return v

    def compute_reid(self, img, gt):
        v, _ = self.reid.forward_cosine(img, gt)
        v = v.cpu().item()
        return v
    
    def compute_reid_resnet(self, result, gt):
        result_feat_dict = self.reid_resnet.run_reid_model(result)
        gt_feat_dict = self.reid_resnet.run_reid_model(gt)

        similarity = 0
        for module_name in result_feat_dict:
            result_feat = result_feat_dict[module_name]
            gt_feat = gt_feat_dict[module_name]
            similarity += torch.cosine_similarity(result_feat, gt_feat, dim=1)

        similarity = (similarity / len(result_feat_dict)).mean().item()
        return similarity

    def number_of_params(self):
        total = 0
        for m in self.model.parameters():
            total += m.numel()
        print('Params: {:.1f}M'.format(total/1e6))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./pretrained/texformer_ep500.pt', help='path to your checkpoint')
    parser.add_argument('--src_ch', type=int, default=4, help='Key map')
    parser.add_argument('--tgt_ch', type=int, default=3, help='Query map')
    parser.add_argument('--feat_dim', type=int, default=128, help='feature dimension')
    parser.add_argument('--nhead', type=int, default=8, help='number of heads')
    parser.add_argument('--mask_fusion', type=int, default=1, help='use mask fusion for output')
    parser.add_argument('--out_ch', type=int, default=3, help='not useful when mask_fusion is True')
    
    options = parser.parse_args()
    tester = Tester(options)
    tester.generate_numerical_results()
    tester.number_of_params()