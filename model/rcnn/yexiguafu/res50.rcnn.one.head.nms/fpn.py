import math

import megengine.functional as F
import megengine.module as M

class FPN(M.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """
    def __init__(self, bottom_up):
        super(FPN, self).__init__()
        in_channels = [256, 512, 1024, 2048]
        fpn_dim = 256
        use_bias =True

        # lateral_convs = list()
        # output_convs = list()
        lateral_convs, output_convs = [], []
        for idx, in_channels in enumerate(in_channels):
            lateral_conv = M.Conv2d(
                in_channels, fpn_dim, kernel_size=1, bias=use_bias)
            output_conv = M.Conv2d(
                fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1, bias=use_bias)
            M.init.msra_normal_(lateral_conv.weight, mode="fan_in")
            M.init.msra_normal_(output_conv.weight, mode="fan_in")
            if use_bias:
                M.init.fill_(lateral_conv.bias, 0)
                M.init.fill_(output_conv.bias, 0)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.bottom_up = bottom_up

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        bottom_up_features = bottom_up_features[::-1]
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            bottom_up_features[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):  
            fh, fw = features.shape[2:]
            top_down_features = F.nn.interpolate(
                prev_features, size = (fh, fw), mode="BILINEAR")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            results.append(output_conv(prev_features))
        # p6
        last_p6 = F.max_pool2d(results[0], kernel_size=1, stride=2, padding=0)
        results.insert(0, last_p6)
        return results
