import os
import json
import argparse
import numpy as np
import torch
import random

class TrainOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--exp_name', type=str, default='texformer', help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--num_workers', type=int, default=4, help='Number of processes used for data loading')

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')

        net = self.parser.add_argument_group('network structures')
        net.add_argument('--feat_dim', type=int, default=128, help='base channel number of features')
        net.add_argument('--out_type', type=str, default='rgb', help='output flow or rgb?')   # out_ch = value_ch
        net.add_argument('--nhead', type=int, default=8, help='output flow or rgb?')
        net.add_argument('--mask_fusion', type=int, default=1, help='whether use mask fusion')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=500, help='Total number of training epochs')
        train.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
        train.add_argument('--batch_size', type=int, default=16, help='Batch size')
        train.add_argument('--summary_steps', type=int, default=50, help='Summary saving frequency')
        train.add_argument('--save_epochs', type=int, default=100, help='Checkpoint saving frequency')
        train.add_argument("--device", default='cuda', help="Training device, has to be cuda")
        train.add_argument('--reid_loss_weight', type=float, default=5000, help='weight of reid loss')
        train.add_argument('--part_style_loss_weight', type=float, default=0.4, help='weight of part style loss')
        train.add_argument('--face_structure_loss_weight', type=float, default=0.01, help='weight of face-structure loss')
        train.add_argument('--mv_loss_weight', type=float, default=1, help='weight of multi-view loss')
        train.add_argument('--reid_norm_feat', type=int, default=1, help='normalize feature or not? default do normalization, ie consine distance')
        train.add_argument('--permute', type=int, default=0, help='should use permute=0')
        train.add_argument('--seed', type=int, default=1234, help='Random seed')

    def parse_args(self, manual_args=None):
        """Parse input arguments."""
        self.args = self.parser.parse_args(manual_args)
        self.set_seed()
        self.set_input_output_channels()
        self.make_log_dirs()    
        self.post_processing()     
        self.save_dump()
        return self.args

    def post_processing(self):
        if self.args.reid_norm_feat == 0:
            self.args.reid_loss_weight = 0.1
    
    def make_log_dirs(self):
        self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.exp_name)

        # mkdir for log, log/tensorboard, log/checkpoints
        self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
        os.makedirs(self.args.log_dir, exist_ok=True)
        
        self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
    
    def set_input_output_channels(self):
        self.args.src_ch = 4
        self.args.tgt_ch = 3

        out_ch_dict = {'flow': 2, 'rgb': 3}
        self.args.out_ch = out_ch_dict[self.args.out_type]

    def set_seed(self):
        seed = self.args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.deterministic = True

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        

