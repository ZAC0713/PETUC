import torch
import torch.nn as nn
from Anatomical_Guided_Transformer import AnatomicalGuidedTransformer
from pytorch_wavelets import DWTForward


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),)

    def forward(self, x):
        yL, yH = self.wt(x)
        # Concatenate the low-frequency subband with the three high-frequency subbands (LH, HL, HH)
        x = torch.cat([yL] + [yH[0][:, :, i, ::] for i in range(3)], dim=1)
        return self.conv_bn_relu(x)


def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, -1, H, W)


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(EUCB, self).__init__()
        self.in_channels = in_channels
        # Depthwise conv after upsampling to capture local spatial context per channel
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.up_dwc(x)
        # Channel shuffle to enhance cross-channel information flow before pointwise conv
        x = channel_shuffle(x, self.in_channels)
        return self.pwc(x)


class CBAMLayer(nn.Module):
    def __init__(self, spatial_kernel=7, k_size=3):
        super(CBAMLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.channel_sigmoid = nn.Sigmoid()
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention: squeeze spatial dims and learn inter-channel dependencies
        y = self.avg_pool(x)
        y = self.channel_conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x = x * self.channel_sigmoid(y).expand_as(x)

        # Spatial attention: aggregate channel info via max and mean, then learn spatial weights
        spatial_in = torch.cat([torch.max(x, dim=1, keepdim=True)[0],torch.mean(x, dim=1, keepdim=True)], dim=1)
        x = x * self.spatial_sigmoid(self.spatial_conv(spatial_in))
        return x


class CBAMUNet_NPJ(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super(CBAMUNet_NPJ, self).__init__()

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=1)

        # Encoder
        self.encoder1 = self._conv_block(base_channels, 64)
        self.pool1    = Down_wt(64, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.pool2    = Down_wt(128, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.pool3    = Down_wt(256, 256)
        self.encoder4 = self._conv_block(256, 512)

        # Bottleneck
        self.bottleneck = AnatomicalGuidedTransformer(512, n_heads=8, d_head=256, context_dim=512)

        # Decoder
        self.decoder4 = self._conv_block(512, 512)
        self.upconv3  = EUCB(512, 256)
        self.decoder3 = self._conv_block(512, 256)
        self.upconv2  = EUCB(256, 128)
        self.decoder2 = self._conv_block(256, 128)
        self.upconv1  = EUCB(128, 64)
        self.decoder1 = self._conv_block(128, base_channels)

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(True),
            CBAMLayer(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(True),
            CBAMLayer(),
        )

    def forward(self, x, cond):
        res = x
        x = self.in_conv(x)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        dec = self.decoder4(self.bottleneck(enc4, cond))

        # Skip connections from encoder are concatenated with upsampled decoder features
        dec = self.decoder3(torch.cat([self.upconv3(dec), enc3], dim=1))
        dec = self.decoder2(torch.cat([self.upconv2(dec), enc2], dim=1))
        dec = self.decoder1(torch.cat([self.upconv1(dec), enc1], dim=1))

        # Global residual connection to preserve the original input information
        return res + self.final_conv(dec)
