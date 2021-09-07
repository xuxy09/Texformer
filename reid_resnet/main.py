import torch
import torch.nn.functional as F
import config
from reid_resnet.resnet import resnet50 
from reid_resnet.torchtools import load_pretrained_weights

class ReIDModel:
    def __init__(self):
        self.model = resnet50(num_classes=751,
                              loss='softmax',
                              pretrained=True,
                              use_gpu=True)

        load_pretrained_weights(self.model, config.reid_resnet50_market_weight_path)
        self.device = 'cuda'
        self.model.to(self.device)
        self.model.eval()
        
        self.feature_list = ['layer1', 'layer2', 'layer3', 'layer4']
        self.size = (256, 128)
        self.normalize_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.normalize_mean = self.normalize_mean.view(1, 3, 1, 1).to(self.device)

        self.normalize_std = torch.Tensor([0.229, 0.224, 0.225])
        self.normalize_std = self.normalize_std.view(1, 3, 1, 1).to(self.device)
    
    @torch.no_grad()
    def run_reid_model(self, x):
        x = self.preprocess(x)
        feature_dict = {}
        for module_name, module in self.model.named_children():
            x = module(x)
            if module_name == 'global_avgpool':
                x = torch.flatten(x, 1)
            if module_name in self.feature_list:
                feature_dict[module_name] = torch.flatten(x, 1)   # un-normalized
            if module_name == self.feature_list[-1]:
                break
        return feature_dict

    def preprocess(self, data):
        """
        the input image is normalized in [-1, 1] and in bgr format, should be changed to the format accecpted by model
        :param data:
        :return:
        """
        data_unnorm = data / 2.0 + 0.5
        
        data_rgb_unnorm = data_unnorm
        data_rgb_unnorm = F.upsample(data_rgb_unnorm, size=self.size, mode='bilinear')
        data_rgb = (data_rgb_unnorm - self.normalize_mean) / self.normalize_std
        return data_rgb
