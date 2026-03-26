import torch
import torch.nn as nn
import torch.nn.functional as F
from Anatomical_Guided_Transformer import AnatomicalGuidedTransformer
from pytorch_wavelets import DWTForward


def top_K_softmax(tensor, k):
    """
    Perform softmax on the input tensor along the channel dimension and retain the top k largest channel values.

    Args:
        tensor (torch.Tensor): Input tensor, with shape (B, C, H, W)
        k (int): The maximum number of values to retain (1 <= k <= C)

    Returns:
        torch.Tensor: The processed tensor, with the same shape as the input."""
    assert 1 <= k <= tensor.size(1), "k must be between 1 and the number of channels."

    # Apply softmax along the channel dimension
    sm_x = F.softmax(tensor, dim=1)

    # Select top-k channels and generate a binary mask via scatter, then apply it
    _, idx_2d = torch.topk(sm_x, k=k, dim=1)
    mask = torch.zeros_like(sm_x).scatter_(1, idx_2d, 1)

    return sm_x * mask


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


class Gate_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gate_block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # Output sparse gating weights with top-k softmax (k=4) over expert dimension
        return top_K_softmax(self.layers(x), 4)


class Self_Attention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Self_Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = 1.0 / (out_channels ** 0.5)

    def forward(self, feature, feature_map):
        query = self.query_conv(feature)
        key = self.key_conv(feature)
        value = self.value_conv(feature)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)

        # Residual addition of attended values onto the feature map
        return feature_map + attended_values


class Scale_Expert_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Scale_Expert_block, self).__init__()
        self.attn = Self_Attention(in_channels, in_channels)

        self.expert = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5,stride=1, padding=2),
            nn.SiLU(True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.expert(self.attn(x, x))

        return out




class Pixel_MOE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pixel_MOE, self).__init__()
        self.experts = nn.ModuleList([Scale_Expert_block(in_channels, out_channels) for _ in range(7)])
        # Each expert operates at a different spatial scale; tuple entries use absolute target size
        self.scales = [0.25, (42 ,42) ,0.5, 1, 2, 3, 4]
        self.restore_scales = [4, (128 ,128), 2, 1, 0.5, 1/3, 0.25]

    def forward(self, x, gate_weight):
        outputs = []
        for i, expert in enumerate(self.experts):
            scale = self.scales[i]
            if isinstance(scale, tuple):
                xi = F.interpolate(x, size=scale, mode='bicubic', align_corners=False)
            else:
                xi = F.interpolate(x, scale_factor=scale, mode='bicubic', align_corners=False) if scale != 1 else x

            xi = expert(xi)

            # Restore expert output back to the original spatial resolution
            restore = self.restore_scales[i]
            if isinstance(restore, tuple):
                xi = F.interpolate(xi, size=restore, mode='bicubic', align_corners=False)
            elif restore != 1:
                xi = F.interpolate(xi, scale_factor=restore, mode='bicubic', align_corners=False)

            outputs.append(xi * gate_weight[:, i:i + 1, :, :])

        return sum(outputs)


class CBAMUNet_MICCAI(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super(CBAMUNet_MICCAI, self).__init__()

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

        # Output
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.gate       = Gate_block(1, 7)
        self.MOE        = Pixel_MOE(1, 1)

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

        dec = self.final_conv(dec)
        # Gate weights derived from the input residual guide the MOE fusion, then add global residual
        return res + self.MOE(dec, self.gate(res))
