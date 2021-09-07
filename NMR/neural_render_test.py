import pickle

import torch.nn.functional as F
import torch
import numpy as np
import neural_renderer as nr

from scipy.spatial.transform import Rotation as SciR
import config


class NrTextureRenderer:
    def __init__(self, render_res=224, device='cuda', factor=1, anti_aliasing=False):
        self.device = device
        with open(config.ra_body_path, 'rb') as f:
            this_dict = pickle.load(f)

        # load faces, vt, uv map
        self.faces = this_dict['faces'].to(device)   # 13776x3
        self.faces_uv = this_dict['faces_uv'].to(device)   # 13776x3

        verts_uv = this_dict['verts_uv']   # 7576x2
        verts_uv[:, 1] = 1-verts_uv[:, 1]

        self.verts_uv_t = verts_uv[None, None].to(self.device) * 2 - 1

        self.focal_length = 5000
        self.render_res = render_res
        # We use Neural 3D mesh renderer for rendering masks and part segmentations
        self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.render_res,
                                           image_size=self.render_res,
                                           light_intensity_ambient=1,
                                           light_intensity_directional=0,
                                           anti_aliasing=anti_aliasing)
        self.st = 2   # resolution of texture image

        # for part segmentation
        textures = np.load(config.VERTEX_TEXTURE_FILE)
        self.textures = torch.from_numpy(textures).float().to(self.device)
        self.cube_parts = torch.from_numpy(np.load(config.cube_parts_path)).float().to(self.device)

        # for extra control
        self.factor = factor

    def set_light_param(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.neural_renderer, k, v)

    def triangle_to_cube(self, tex_tensor):
        # input: Bx3x13776x3
        if self.st == 1:
            batch_size = tex_tensor.shape[0]
            tex_tensor = tex_tensor.mean(dim=-1)  # Bx3x13776
            tex_tensor = tex_tensor.permute(0, 2, 1).view(batch_size, -1, 1, 1, 1, 3)  # Bx13776x1x1x1x3
        else:
            tmp = torch.linspace(0, 1, self.st).to(tex_tensor.device)
            x, y, z = torch.meshgrid(tmp, tmp, tmp)
            x = x[None, None, None]
            y = y[None, None, None]
            z = z[None, None, None]
            # B x 3 x 13776 x st x st x st
            tex_tensor = x * tex_tensor[:, :, :, 0:1, None, None] + y * tex_tensor[:, :, :, 1:2, None, None] + z * tex_tensor[:, :, :, 2:, None, None]
            # B x 13776 x st x st x st x 3
            tex_tensor = tex_tensor.permute(0, 2, 3, 4, 5, 1)
        return tex_tensor

    def get_tex_tensor(self, uv_map_t):
        batch_size = uv_map_t.shape[0]
        verts_uv_t = self.verts_uv_t.expand(batch_size, -1, -1, -1)
        sampled_uv = F.grid_sample(uv_map_t, verts_uv_t)   # Bx3x1x7576

        # generate texture tensor
        tex_tensor = sampled_uv.squeeze(2)[:, :, self.faces_uv.flatten()]   # Bx3x(13776*3)
        tex_tensor = tex_tensor.view(batch_size, 3, -1, 3)   # Bx3x13776x3
        tex_tensor = self.triangle_to_cube(tex_tensor)

        return tex_tensor

    def W_crop(self, img, width=64):
        # output_width = width
        W = img.shape[-1]
        start = (W - width) // 2
        end = start + width
        return img[..., start: end]

    def render(self, verts, cam_t, uv_map_t, crop_width=None, euler=None, tex_available=False):
        if tex_available:
            tex_tensor = uv_map_t
        else:
            tex_tensor = self.get_tex_tensor(uv_map_t)

        batch_size = verts.shape[0]
        K = torch.eye(3, device=self.device)
        K[0, 0] = self.focal_length * self.render_res/224 * self.factor
        K[1, 1] = self.focal_length * self.render_res/224 * self.factor
        K[2, 2] = 1
        K[0, 2] = self.render_res / 2.
        K[1, 2] = self.render_res / 2.
        K = K[None, :, :].expand(batch_size, -1, -1)
        if euler is None:
            R = torch.eye(3, device=self.device)[None, :, :].expand(batch_size, -1, -1)
        else:
            R = SciR.from_euler('zyx', euler, degrees=True).as_dcm()
            R = torch.from_numpy(R).type(torch.float32).to(self.device)[None, :, :].expand(batch_size, -1, -1)

        faces = self.faces[None, :, :].expand(batch_size, -1, -1)
        parts, depth, mask = self.neural_renderer(verts, faces, textures=tex_tensor, K=K, R=R, t=cam_t.unsqueeze(1))

        if crop_width is not None:
            parts = self.W_crop(parts, crop_width)
            depth = self.W_crop(depth, crop_width)
            mask = self.W_crop(mask, crop_width)

        return parts, depth, mask

    def get_parts(self, parts, mask):
        """Process renderer part image to get body part indices."""
        bn,c,h,w = parts.shape
        mask = mask.view(-1,1)
        parts_index = torch.floor(100*parts.permute(0,2,3,1).contiguous().view(-1,3)).long()
        parts = self.cube_parts[parts_index[:,0], parts_index[:,1], parts_index[:,2], None]   
        parts *= mask
        parts = parts.view(bn,h,w).long()
        return parts

    def render_part(self, verts, cam_t, crop_width=None, euler=None):
        batch_size = verts.shape[0]
        K = torch.eye(3, device=self.device)
        K[0, 0] = self.focal_length * self.render_res/224
        K[1, 1] = self.focal_length * self.render_res/224
        K[2, 2] = 1
        K[0, 2] = self.render_res / 2.
        K[1, 2] = self.render_res / 2.
        K = K[None, :, :].expand(batch_size, -1, -1)
        if euler is None:
            R = torch.eye(3, device=self.device)[None, :, :].expand(batch_size, -1, -1)
        else:
            R = SciR.from_euler('zyx', euler, degrees=True).as_dcm()
            R = torch.from_numpy(R).type(torch.float32).to(self.device)[None, :, :].expand(batch_size, -1, -1)

        faces = self.faces[None, :, :].expand(batch_size, -1, -1)
        parts, depth, mask = self.neural_renderer(verts, faces, textures=self.textures, K=K, R=R, t=cam_t.unsqueeze(1))
        parts_discrete = self.get_parts(parts, mask)

        if crop_width is not None:
            parts_discrete = self.W_crop(parts_discrete, crop_width)
        
        return parts_discrete



