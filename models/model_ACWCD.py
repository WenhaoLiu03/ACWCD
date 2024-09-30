import torch
import torch.nn as nn
import torch.nn.functional as F
from . import mix_transformer
from .seg_head import SimpleSeg
import numpy as np


class ACWCD(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None, ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride
        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes-1, kernel_size=1,
                                    bias=False)

        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(self.in_channels[3], self.num_classes, kernel_size=1)

        self.decoder = SimpleSeg(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)

        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.diff_c4 = conv_diff_d(in_channels=2 * c4_in_channels, out_channels=c4_in_channels)
        self.diff_at = conv_diff_d(in_channels=32, out_channels=16)

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def forward(self, x1, x2, cam_only=False, seg_detach=True,):
        _x1, _attns1 = self.encoder(x1)
        _x2, _attns2 = self.encoder(x2)

        _, _, _, _c4_1 = _x1
        _, _, _, _c4_2 = _x2

        ### integration ###

        _c4 = torch.absolute(_c4_1 - _c4_2)

        seg = self.decoder(_c4)

        attn_cat1 = torch.cat(_attns1[-2:], dim=1)  # .detach()
        attn_cat2 = torch.cat(_attns2[-2:], dim=1)  # .detach()

        _attns = torch.absolute(attn_cat1 - attn_cat2)

        attn_cat = _attns + _attns.permute(0, 1, 3, 2)
        change_attn = self.attn_proj(attn_cat)
        change_attn = torch.sigmoid(change_attn)[:, 0, ...]

        if cam_only:
            cam_s4 = F.conv2d(_c4, self.classifier.weight).detach()
            return cam_s4, change_attn

        cls = self.pooling(_c4, (1, 1))
        cls = self.classifier(cls)
        cls = cls.view(-1, self.num_classes-1)

        return cls, seg, change_attn

if __name__ == "__main__":
    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    acwcd = ACWCD('mit_b1', num_classes=2, embedding_dim=256, pretrained=True)
    acwcd._param_groups()
    dummy_input = torch.rand(2, 3, 256, 256)
    acwcd(dummy_input)

def conv_diff_d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )

