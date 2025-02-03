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