import torch
import config
from RSC_net.hmr import hmr
from RSC_net.smpl import SMPL
import RSC_net.constants as constants


# Test the RA model, input: img, output: theta, beta, vertices
class RaRunner:
    def __init__(self, device='cuda'):
        # Load model
        self.device = device
        checkpoint_name = config.RSC_checkpoint
        self.model = hmr(config.SMPL_MEAN_PARAMS).to(self.device)
        checkpoint = torch.load(checkpoint_name)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()

        self.smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1, create_transl=False).to(self.device)

        self.IMG_NORM_MEAN = constants.IMG_NORM_MEAN
        self.IMG_NORM_STD = constants.IMG_NORM_STD

    def pre_process_batch(self, batch):
        """
        normalize the batch
        :param batch: torch Tensor with shape bx3x224x224, 0~1
        :return: normalized tensor
        """
        return (batch - torch.tensor(self.IMG_NORM_MEAN).view(1, -1, 1, 1)) / torch.tensor(self.IMG_NORM_STD).view(1, -1, 1, 1)

    def get_3D_batch(self, input_batch, scale=0, pre_process=True):
        if pre_process:
            input_batch = self.pre_process_batch(input_batch)
            input_batch = input_batch.to(self.device)
        
        pred_rotmat, pred_betas, pred_camera, _ = self.model(input_batch, scale=scale)

        pred_output = self.smpl(betas=pred_betas,  
                                body_pose=pred_rotmat[:, 1:],
                                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                                pose2rot=False)
        pred_vertices = pred_output.vertices

        pred_cam_t = torch.stack([pred_camera[:, 1],
                                  pred_camera[:, 2],
                                  2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:, 0] + 1e-9)],
                                  dim=-1)

        return pred_vertices, pred_cam_t, pred_rotmat, pred_betas

    @staticmethod
    def get_scale(res):
        range_points = [224, 128, 64, 40]  # 24
        scale = len(range_points)
        for i, p in enumerate(range_points):
            if res >= p:
                scale = i
                break
        return scale


