import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# -------------------------------------------------------------
# Model 1: Custom From-Scratch CNN / U-Net (Pure Deep Learning)
# -------------------------------------------------------------
class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> GELU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)

class BaselineCustomUNet(nn.Module):
    """
    Model 1: Custom Scratch U-Net with no pretrained weights.
    Encoder: 4 stages (32 -> 64 -> 128 -> 256)
    Bottleneck: 512
    Decoder: 4 stages with ConvTranspose2d + skip connections
    Head: 1x1 conv -> 4 channels (seg_logits)
    Optional classification head: GlobalAvgPool -> Linear -> 4 classes
    """
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.bot = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64, 32)
        
        self.seg_out = nn.Conv2d(32, num_classes, kernel_size=1)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Kaiming normal init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        bot = self.bot(x4)
        
        cls_logits = self.cls_head(bot)
        
        d1 = self.conv1(torch.cat([self.up1(bot), x4], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d1), x3], dim=1))
        d3 = self.conv3(torch.cat([self.up3(d2), x2], dim=1))
        d4 = self.conv4(torch.cat([self.up4(d3), x1], dim=1))
        seg_logits = self.seg_out(d4)
        
        return cls_logits, seg_logits

# -------------------------------------------------------------
# Model 2: Standard Transfer Learning Model
# -------------------------------------------------------------
class StandardTransferUNet(nn.Module):
    """
    Model 2: Standard ImageNet Pretrained Backbone (ResNet18 / ResNet34) with U-Net decoder.
    Single segmentation head output.
    """
    def __init__(self, backbone_name='resnet18', num_classes=4, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )
        ch = self.encoder.feature_info.channels()  # [64, 64, 128, 256, 512]
        
        self.up4 = nn.ConvTranspose2d(ch[4], ch[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(ch[3] + ch[3], ch[3])
        
        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(ch[2] + ch[2], ch[2])
        
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(ch[1] + ch[1], ch[1])
        
        self.up1 = nn.ConvTranspose2d(ch[1], 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32 + ch[0], 32)
        
        self.up0 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec0 = DoubleConv(16, 16)
        
        self.seg_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Repeat 1 ch to 3 ch if grayscale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self.encoder(x)  # C1 (stride 2), C2 (4), C3 (8), C4 (16), C5 (32)
        
        d4 = self.dec4(torch.cat([self.up4(feats[4]), feats[3]], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), feats[2]], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), feats[1]], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), feats[0]], dim=1))
        d0 = self.dec0(self.up0(d1))
        seg_logits = self.seg_out(d0)
        return seg_logits

# -------------------------------------------------------------
# Model 3: Dual-Head Multi-Task FPN (Grayscale & Stride-2 Optimized)
# -------------------------------------------------------------
class DualHeadMTFPN(nn.Module):
    """
    Model 3: Production-Grade Multi-Task Feature Pyramid Network.
    - Native 1-channel Grayscale Backbone (in_chans=1)
    - Stride-2 High-Resolution FPN: multi-scale lateral connections across all pyramid stages
      smoothed and fused at first extracted stage resolution (stride 2 or stride 4),
      preserving fine boundary details for thin vertical scratches (defect_b) and pits (defect_a).
    - Auxiliary Classification Head: AdaptiveAvgPool2d from deepest layer C5 predicting defect probabilities.
    - Output: (cls_logits [B, 4], seg_logits [B, 4, 128, 800])
    """
    def __init__(self, backbone_name='tf_efficientnetv2_s', num_classes=4, pretrained=True, fpn_dim=128):
        super().__init__()
        try:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                in_chans=1
            )
        except Exception:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True
            )
            self._adapt_first_conv_to_grayscale()

        self.feature_channels = self.encoder.feature_info.channels()
        c5_ch = self.feature_channels[-1]
        
        # 1. Image-level Defect Classification Head (from C5)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(c5_ch, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )
        
        # 2. FPN Lateral & Smoothing Layers
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, fpn_dim, kernel_size=1) for ch in self.feature_channels
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.GELU()
            ) for _ in self.feature_channels
        ])
        
        # 3. Final Multi-Scale Segmentation Head
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_dim * len(self.feature_channels), fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        )

    def _adapt_first_conv_to_grayscale(self):
        conv1 = None
        parent_name = ''
        attr_name = ''
        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                conv1 = module
                if '.' in name:
                    parent_name, attr_name = name.rsplit('.', 1)
                else:
                    attr_name = name
                break
                
        if conv1 is not None:
            w_3ch = conv1.weight.data  # (out_ch, 3, k, k)
            w_1ch = w_3ch.sum(dim=1, keepdim=True)  # (out_ch, 1, k, k)
            
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=(conv1.bias is not None)
            )
            new_conv.weight.data.copy_(w_1ch)
            if conv1.bias is not None:
                new_conv.bias.data.copy_(conv1.bias.data)
                
            if parent_name:
                parent = dict(self.encoder.named_modules())[parent_name]
            else:
                parent = self.encoder
            setattr(parent, attr_name, new_conv)

    def forward(self, x):
        # x shape: (B, 1, 128, 800)
        feats = self.encoder(x)
        
        # Auxiliary Classification Prediction
        cls_logits = self.cls_head(feats[-1])  # (B, 4)
        
        # Top-down FPN pathway
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, feats)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode='nearest'
            )
            
        # Target size: highest resolution stage (e.g. stride 2: 64 x 400)
        target_size = feats[0].shape[-2:]
        smoothed = [
            F.interpolate(sm(lat), size=target_size, mode='bilinear', align_corners=False)
            for sm, lat in zip(self.smooth_convs, laterals)
        ]
        
        # Concatenate multi-scale representations
        fused = torch.cat(smoothed, dim=1)
        seg_feats = self.seg_head(fused)
        
        # Bilinear upsample to full input resolution (128, 800)
        seg_logits = F.interpolate(seg_feats, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return cls_logits, seg_logits
