import torch
import torch.nn as nn
import timm

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = nn.ReLU()(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet50, self).__init__()

        if pretrained:
            print("Loading Pretrained Weights for ResNet50...")
            self.model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        else:
            self.in_channels = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(64, 3)
            self.layer2 = self._make_layer(128, 4, stride=2)
            self.layer3 = self._make_layer(256, 6, stride=2)
            self.layer4 = self._make_layer(512, 3, stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        if hasattr(self, "model"):  # pretrained model
            return self.model(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

import torch
import torch.nn as nn
import timm

import torch
import torch.nn as nn
import timm

class ViT_S16(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ViT_S16, self).__init__()

        if pretrained:
            print("Loading Pretrained Weights for ViT-S/16...")
            self.model = timm.create_model(
                "vit_small_patch16_224", pretrained=True, img_size=32
            )

            # 224x224 → 32x32
            self.model.patch_embed.proj = nn.Conv2d(3, self.model.embed_dim, kernel_size=4, stride=4)

            pos_embed = self.model.pos_embed 
            class_token = pos_embed[:, :1, :]
            grid_embed = pos_embed[:, 1:, :]

            num_tokens = grid_embed.shape[1]
            new_num_tokens = (32 // 4) ** 2

            new_grid_embed = nn.functional.interpolate(
                grid_embed.reshape(1, int(num_tokens**0.5), int(num_tokens**0.5), -1).permute(0, 3, 1, 2),
                size=(int(new_num_tokens**0.5), int(new_num_tokens**0.5)),
                mode="bilinear",
                align_corners=False
            ).flatten(2).permute(0, 2, 1)

            self.model.pos_embed = nn.Parameter(torch.cat([class_token, new_grid_embed], dim=1))

        else:
            self.model = timm.create_model("vit_small_patch16_224", pretrained=False, img_size=32)

        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
