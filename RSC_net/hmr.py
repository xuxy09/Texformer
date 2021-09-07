import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import numpy as np
import math
from collections import OrderedDict
from .utils.geometry import rot6d_to_rotmat


class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, alpha):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out = alpha * out + residual
        out = self.relu(out)

        return out


class HMRLayer(nn.Module):
    def __init__(self, block, in_planes, planes, blocks, stride=1):
        super(HMRLayer, self).__init__()
        self.hmr_layer = nn.ModuleList()
        self.in_planes = in_planes
        # downsample
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        self.hmr_layer.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            self.hmr_layer.append(block(self.in_planes, planes))

        # initial alphas to [0, 0, ..., 0, 1]
        # alphas =
        # add alphas parameters
        self.alphas = nn.ParameterDict({
            '0': nn.Parameter(torch.from_numpy(np.ones(blocks, dtype=np.float32)).float().view(-1, 1, 1, 1, 1)),
            '1': nn.Parameter(torch.from_numpy(np.ones(blocks, dtype=np.float32)).float().view(-1, 1, 1, 1, 1)),
            '2': nn.Parameter(torch.from_numpy(np.ones(blocks, dtype=np.float32)).float().view(-1, 1, 1, 1, 1)),
            '3': nn.Parameter(torch.from_numpy(np.ones(blocks, dtype=np.float32)).float().view(-1, 1, 1, 1, 1)),
            '4': nn.Parameter(torch.from_numpy(np.ones(blocks, dtype=np.float32)).float().view(-1, 1, 1, 1, 1))
        })
        self.alphas.requires_grad = True

    def init_alphas(self, scale, device):
        """
        scale = [1, 2, 3, 4], need to be larger than 0
        """
        self.alphas[str(scale)].data = self.alphas[str(scale - 1)].detach().clone()




    def forward(self, x, scale):
        out = x
        alphas = self.alphas[str(scale)]

        for i, res_block in enumerate(self.hmr_layer):
            alpha = alphas[i]
            out = res_block(out, alpha)

        return out



class HMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params):
        self.inplanes = 64
        super(HMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = HMRLayer(block, self.inplanes, 64, layers[0])
        self.layer2 = HMRLayer(block, self.layer1.in_planes, 128, layers[1], stride=2)
        self.layer3 = HMRLayer(block, self.layer2.in_planes, 256, layers[2], stride=2)
        self.layer4 = HMRLayer(block, self.layer3.in_planes, 512, layers[3], stride=2)
        self.layer4_mlp = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256))
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def init_alphas(self, scale, device):
        self.layer1.init_alphas(scale, device)
        self.layer2.init_alphas(scale, device)
        self.layer3.init_alphas(scale, device)
        self.layer4.init_alphas(scale, device)

    def forward(self, x, scale, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x, scale)
        x2 = self.layer2(x1, scale)
        x3 = self.layer3(x2, scale)
        x4 = self.layer4(x3, scale)
        feat_layer4 = x4.view(batch_size, x4.size(1), -1).mean(dim=-1)
        feat_layer4 = self.layer4_mlp(feat_layer4)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        feat_list = [feat_layer4]

        return pred_rotmat, pred_shape, pred_cam, feat_list

def hmr(smpl_mean_params, pretrained=True, **kwargs):
    """ Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    if pretrained:
        resnet_imagenet = resnet.resnet50(pretrained=True)
        state_dict = resnet_imagenet.state_dict()
        renamed_state_dict = OrderedDict()
        # change the names in the state_dict to match the new layer
        for key, value in state_dict.items():
            if 'layer' in key:
                names = key.split('.')
                names[1:1] = ['hmr_layer']
                new_key = '.'.join(n for n in names)
                renamed_state_dict[new_key] = value
            else:
                renamed_state_dict[key] = value
        model.load_state_dict(renamed_state_dict,strict=False)
        # state_dict = model.state_dict()
    return model


if __name__ == '__main__':
    import config
    a = torch.ones((1, 3, 224, 224))
    model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True)

    for i in range(3):
        model.init_alphas(scale=i, device=None)


    out = model(a, scale=1)


