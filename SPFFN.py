import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        channel_attention = self.channel_att(x)
        x = x * channel_attention

        # Spatial Attention
        spatial_attention = self.spatial_att(x)
        x = x * spatial_attention

        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SPFFN(nn.Module):
    def __init__(self, num_classes=5):
        super(SPFFN, self).__init__()

        # self.se_attention = DES-SE(5)

        self.features = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv(5, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                CBAM(64),
                DepthwiseSeparableConv(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                CBAM(64),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                DepthwiseSeparableConv(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                CBAM(128),
                DepthwiseSeparableConv(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                CBAM(128),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                DepthwiseSeparableConv(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                CBAM(256),
                DepthwiseSeparableConv(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                CBAM(256),
                DepthwiseSeparableConv(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                CBAM(256),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                DepthwiseSeparableConv(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                CBAM(512),
                DepthwiseSeparableConv(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                CBAM(512),
                DepthwiseSeparableConv(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                CBAM(512),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            nn.Sequential(
                DepthwiseSeparableConv(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                CBAM(512),
                DepthwiseSeparableConv(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                CBAM(512),
                DepthwiseSeparableConv(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                CBAM(512),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        ])

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(512, 256, kernel_size=1)
        ])

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.classifier = nn.Sequential(
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.se_attention(x)

        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x)

        fpn_out = []
        for i in range(len(features)):
            lateral_conv = self.lateral_convs[i](features[i])
            fpn_out.append(lateral_conv)

        for i in range(len(fpn_out) - 1, 0, -1):
            fpn_out[i - 1] += self.upsample(fpn_out[i])

        if fpn_out:
            x = fpn_out[0]
            x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        else:
            raise ValueError("FPN output is empty. Check the feature extraction step.")

        return x



model = SPFFN(num_classes=7)
