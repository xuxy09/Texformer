import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips.pretrained_networks import vgg16

class PartStyleLoss(nn.Module):
    def __init__(self, idx_max, extract_features=False, device=None):
        super().__init__()
        self.idx_max = idx_max
        self.loss_fn = nn.MSELoss()
        if extract_features:
            self.feature_extractor = vgg16(requires_grad=False, pretrained=True)
            self.feature_extractor.to(device)
            self.feature_extractor.eval()
            self.feature_extractor_mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            self.feature_extractor_std = torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def extract_features(self, img, to_01=True):
        if to_01:
            img = (img + 1) / 2   # the input is assumed to be [-1, 1]
        img = (img - self.feature_extractor_mean) / self.feature_extractor_std
        features = self.feature_extractor(img)
        return features

    def gram_matrix_batch(self, feature):
        # feature: BxCx(HW)
        mat = torch.bmm(feature, feature.transpose(1, 2))  # BxCxC
        return mat 

    def gram_matrix(self, feature):
        # feature: CxL
        mat = torch.mm(feature, feature.t())  # CxC
        return mat.div(feature.shape[-1])

    def forward(self, feature1, feature2, mask1, mask2):
        B, C, H1, W1 = feature1.shape
        _, _, H2, W2 = feature2.shape
        
        if mask1.shape[2:] != feature1.shape[2:]:
            mask1 = F.interpolate(mask1.float(), (H1, W1), mode='nearest').long()
        if mask2.shape[2:] != feature2.shape[2:]:
            mask2 = F.interpolate(mask2.float(), (H2, W2), mode='nearest').long()

        loss_list = []
        
        for idx in range(1, self.idx_max+1):
            mask1_part_batch = (mask1 == idx)
            mask2_part_batch = (mask2 == idx)

            # parallel implementation, about 10x faster
            L1 = mask1_part_batch.sum(dim=(2, 3)).squeeze(1)   # B-vector
            L2 = mask2_part_batch.sum(dim=(2, 3)).squeeze(1)

            valid_sample = (L1 >= 0.004*H1*W1) & (L2 >= 0.004*H2*W2)

            if valid_sample.sum() == 0:
                loss_list.append(torch.tensor(0.).to(feature1.device))
            else:                
                f1 = feature1 * mask1_part_batch  # B x C x H x W
                f2 = feature2 * mask2_part_batch

                L1[~valid_sample] = 1.   # avoid dividing-by-zero
                L2[~valid_sample] = 1.
                
                
                gram1 = self.gram_matrix_batch(f1.view(B, C, -1)) 
                gram1 = gram1 * valid_sample.view(-1, 1, 1) / L1.view(-1, 1, 1)   # Note: add-and-multiply are faster than slicing
               

                gram2 = self.gram_matrix_batch(f2.view(B, C, -1)) 
                gram2 = gram2 * valid_sample.view(-1, 1, 1) / L2.view(-1, 1, 1)
    

                loss = self.loss_fn(gram1, gram2) 
                loss_list.append(loss)
        
        return sum(loss_list)


            
        
    

