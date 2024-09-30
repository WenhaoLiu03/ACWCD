import numpy as np
import sys
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class SimpleSeg(nn.Module):
    def __init__(self, feature_strides=None, in_channels=128, embedding_dim=256, num_classes=2, **kwargs):
        super(SimpleSeg, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c4_in_channels = self.in_channels[-1]

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)

        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, c4):
        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=(64, 64), mode='bilinear',align_corners=False)
        feature = self.linear_fuse(_c4)
        x = self.dropout(feature)
        x = self.linear_pred(x)
        x_squeezed = torch.squeeze(x, 1)

        return x_squeezed

