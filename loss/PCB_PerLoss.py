import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import os
from loss.resnet_market1501 import resnet50
import sys

# ReID Loss
class ReIDLoss(nn.Module):
    def __init__(self, model_path, num_classes=1501, size=(384, 128), device='cuda', is_trainable=False, w = [1,1,1,1], normalize_feat=True, permute=0):
        super(ReIDLoss, self).__init__()
        self.size = size
        model_structure = resnet50(num_features=256, dropout=0.5, num_classes=num_classes, cut_at_pooling=False,
                                   FCN=True)

        checkpoint = torch.load(model_path, map_location='cpu')
        
        model_dict = model_structure.state_dict()
        checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model_structure.load_state_dict(model_dict)
        self.model = model_structure
        self.model.eval()
        
        self.w = w
        
        self.normalize_feat = normalize_feat
        
        self.model.to(device)
            
        self.is_trainable = is_trainable
        for param in self.model.parameters():
            param.requires_grad = self.is_trainable
        
        self.MSELoss = nn.MSELoss()

        self.normalize_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.normalize_mean = self.normalize_mean.expand(384, 128, 3).permute(2, 0, 1) 

        self.normalize_std = torch.Tensor([0.229, 0.224, 0.225])
        self.normalize_std = self.normalize_std.expand(384, 128, 3).permute(2, 0, 1) 

        self.normalize_std = self.normalize_std.to(device)
        self.normalize_mean = self.normalize_mean.to(device)

        self.permute = permute

    def extract_feature(self, inputs):
        for n, m in self.model.base.named_children():

            inputs = m.forward(inputs)

            if n == 'layer1':
                o1_ = inputs
            elif n == 'layer2':
                o2_ = inputs
            elif n == 'layer3':
                o3_ = inputs
            elif n == 'layer4':
                o4_ = inputs
                break
        
        o1 = o1_.view(o1_.size(0),-1)
        if self.normalize_feat:
            o1 = o1 / o1.norm(2, 1, keepdim=True).expand_as(o1)
        
        o2 = o2_.view(o2_.size(0),-1)
        if self.normalize_feat:
            o2 = o2 / o2.norm(2, 1, keepdim=True).expand_as(o2)

        o3 = o3_.view(o3_.size(0),-1)
        if self.normalize_feat:
            o3 = o3 / o3.norm(2, 1, keepdim=True).expand_as(o3)
        
        o4 = o4_.view(o4_.size(0),-1)
        if self.normalize_feat:
            o4 = o4 / o4.norm(2, 1, keepdim=True).expand_as(o4)
        
        feature_tri = torch.cat((o1,o2,o3,o4), dim = 1)
        
        return (o1, o2, o3, o4), feature_tri, (o1_, o2_, o3_, o4_)   # o1 is normalized, o1_ is not (ie original)

    def preprocess(self, data):
        """
        the input image is normalized in [-1, 1] and in bgr format, should be changed to the format accecpted by model
        :param data:
        :return:
        """
        data_unnorm = data / 2.0 + 0.5
        
        if self.permute == 1:
            permute = [2, 1, 0]
            data_rgb_unnorm = data_unnorm[:, permute]
        elif self.permute == 0:
            data_rgb_unnorm = data_unnorm
        
        data_rgb_unnorm = F.upsample(data_rgb_unnorm, size=self.size, mode='bilinear')
        data_rgb = (data_rgb_unnorm - self.normalize_mean) / self.normalize_std
        return data_rgb

    def forward(self, data, label):
        
        assert label.requires_grad is False
        data = self.preprocess(data)
        label = self.preprocess(label)

        feature_tri_data, f_data, orig_feature_data = self.extract_feature(data)
        feature_tri_label, f_label, orig_feature_label = self.extract_feature(label)
        
        perceptual_loss = self.w[0] * self.MSELoss(feature_tri_data[0],feature_tri_label[0]) + \
                            self.w[1] * self.MSELoss(feature_tri_data[1],feature_tri_label[1]) + \
                            self.w[2] * self.MSELoss(feature_tri_data[2],feature_tri_label[2]) + \
                            self.w[3] * self.MSELoss(feature_tri_data[3],feature_tri_label[3])

        return perceptual_loss, (orig_feature_data, orig_feature_label)

    def forward_cosine(self, data, label):
        """
        cosine similarity, the higher the better
        """
        
        assert label.requires_grad is False
        data = self.preprocess(data)
        label = self.preprocess(label)

        feature_tri_data, f_data, orig_feature_data = self.extract_feature(data)
        feature_tri_label, f_label, orig_feature_label = self.extract_feature(label)
        
        perceptual_loss = (((feature_tri_data[0] * feature_tri_label[0]).sum(dim=1) + \
                           (feature_tri_data[1] * feature_tri_label[1]).sum(dim=1) + \
                           (feature_tri_data[2] * feature_tri_label[2]).sum(dim=1) + \
                           (feature_tri_data[3] * feature_tri_label[3]).sum(dim=1)) / 4).mean()
        
        return perceptual_loss, (orig_feature_data, orig_feature_label)


    
    
   