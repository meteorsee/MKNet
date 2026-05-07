import torch
from torch import nn
from ultralytics.nn.modules.conv import DWConv
from ultralytics.nn.modules.conv import *
import torch.nn.functional as F


# ######  Mobilenetv3
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class conv_bn_hswish(nn.Module):
    def __init__(self, c1, c2, stride):
        super(conv_bn_hswish, self).__init__()
        self.conv = nn.Conv2d(c1, c2, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # self.act = h_swish()
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SimAM(nn.Module):
    """Parameter-free attention from 'SimAM' (CVPR'21)."""
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=(2, 3), keepdim=True)
        var  = ((x - mean) ** 2).mean(dim=(2, 3), keepdim=True)
        # energy (scaled); a common, stable form
        e = (x - mean) ** 2 / (4 * (var + self.e_lambda)) + 0.5
        attn = torch.sigmoid(e)
        return x * attn

# Original SimAM from "https://github.com/ZjjConan/SimAM/blob/master/networks/attentions/simam_module.py"
class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

# ==============================================================================

class SpaceToDepthStem_DW(nn.Module):
    """
        S2D -> Ultralytics DWConv (DW 3x3 + PW 1x1 inside)
        This approach is directly using Depthwise Conv from Ultralytics library,
        which combines Depthwise and Pointwise conv in one module.
    """
    def __init__(self, c1=3, c2=16, down=2, act_layer=nn.Hardswish):
        super().__init__()
        assert down == 2
        in_ch = c1 * down * down
        self.unshuffle = nn.PixelUnshuffle(down)
        self.dw = DWConv(in_ch, c2, 3, 1)             # does DW(3x3) + PW(1x1) internally
        self.bn = nn.BatchNorm2d(c2)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.dw(x)
        x = self.act(self.bn(x))
        return x

# ==============================================================================

class CSPCAM_MBV3(nn.Module):
    """
    MobileNetV3-style CSP-CAM block:
      - 5 branches (skip + top + 3 dilated branches)
      - uses depthwise 3x3 for spatial convs
      - keeps channel dimension (c -> c)
    Args in YAML: [c, use_se, use_hs]
      c:      number of channels
      use_se: 0/1 -> whether to apply SimAM after fusion
      use_hs: 0/1 -> Hardswish (1) or ReLU (0)
    """
    def __init__(self, c: int, use_se: int = 0, use_hs: int = 1):
        super().__init__()
        Act = nn.Hardswish if use_hs else nn.ReLU

        def pw_act():
            # pointwise conv + BN + act
            return nn.Sequential(
                nn.Conv2d(c, c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c),
                Act(inplace=True),
            )

        def dw_act(dilation: int):
            # depthwise 3x3 with given dilation
            pad = dilation
            return nn.Sequential(
                nn.Conv2d(c, c, 3, 1, pad, dilation=dilation,
                          groups=c, bias=False),
                nn.BatchNorm2d(c),
                Act(inplace=True),
            )

        # --- top branch: 1x1 -> 1x1
        self.top = nn.Sequential(
            pw_act(),
            pw_act(),
        )

        # --- middle branch 1: dilation = 1
        self.b1 = nn.Sequential(
            pw_act(),
            dw_act(dilation=1),
            pw_act(),
        )

        # --- middle branch 2: dilation = 3
        self.b2 = nn.Sequential(
            pw_act(),
            dw_act(dilation=3),
            pw_act(),
        )

        # --- middle branch 3: dilation = 5
        self.b3 = nn.Sequential(
            pw_act(),
            dw_act(dilation=5),
            pw_act(),
        )

        # optional SimAM (or SE)
        self.attn = SimAM() if use_se else nn.Sequential()

        # final 3x3 DW (+BN+act), no outer skip
        self.final = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            Act(inplace=True),
        )

    def forward(self, x):
        # 5 branches
        top = self.top(x)
        b1  = self.b1(x)
        b2  = self.b2(x)
        b3  = self.b3(x)
        skip = x

        fusion = skip + top + b1 + b2 + b3
        fusion = self.attn(fusion)
        out = self.final(fusion)
        return out

# ==============================================================================

class AKMixDW(nn.Module):
    def __init__(self, c, k_small=3, k_big=5, stride=1, act=nn.Hardswish):
        super().__init__()
        p1, p2 = k_small//2, k_big//2
        self.dw3 = nn.Conv2d(c, c, k_small, stride, p1, groups=c, bias=False)
        self.dw5 = nn.Conv2d(c, c, k_big,   stride, p2, groups=c, bias=False)
        self.bn3, self.bn5 = nn.BatchNorm2d(c), nn.BatchNorm2d(c)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(nn.Conv2d(c, c, 1, bias=True), nn.Sigmoid())  # α ∈ [0,1]^(B×C×1×1)
        self.act = act(inplace=True)
    def forward(self, x):
        y3 = self.bn3(self.dw3(x))
        y5 = self.bn5(self.dw5(x))
        alpha = self.gate(self.gap(x))
        y = (1 - alpha) * y3 + alpha * y5
        return self.act(y)

#====================================================
# Modified MobiletNetv3 with sigle AKMixDW & with Partial Conv

def channel_shuffle(x, groups: int):
    # x: [B, C, H, W]
    b, c, h, w = x.size()
    assert c % groups == 0, "channels must be divisible by groups"
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)

class AKMixDW_PartialSCFuse(nn.Module):
    """
    Branch A: AKMixDW on all C channels (adaptive DW)
    Branch B: Standard 3x3 on only r*C channels (true rich conv, but cheap)
    Concat -> 1x1 fuse back to C
    """
    def __init__(self, c, stride=1, k_small=3, k_big=5,
                 ratio=0.25, shuffle=True, act=nn.Hardswish):
        super().__init__()
        self.shuffle = shuffle

        c_sc = max(1, int(round(c * ratio)))
        # keep divisible/compatible (optional): make it even
        if c_sc % 2 == 1 and c_sc < c:
            c_sc += 1
        c_sc = min(c_sc, c)

        self.c_sc = c_sc

        self.branch_a = AKMixDW(c, k_small, k_big, stride=stride, act=act)

        # Standard conv ONLY on subset channels
        self.branch_b = nn.Sequential(
            nn.Conv2d(c_sc, c_sc, 3, stride, 1, bias=False),
            nn.BatchNorm2d(c_sc),
            act(inplace=True),
        )

        # Fuse: (C + c_sc) -> C
        self.fuse = nn.Sequential(
            nn.Conv2d(c + c_sc, c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c),
            act(inplace=True),
        )

    def forward(self, x):
        ya = self.branch_a(x)

        # split off the small subset for rich conv
        x_sc = x[:, :self.c_sc, :, :]
        yb = self.branch_b(x_sc)

        y = torch.cat([ya, yb], dim=1)  # [B, C + c_sc, H', W']
        if self.shuffle:
            # shuffle across 2 groups is still useful here
            y = channel_shuffle(y, groups=2)
        return self.fuse(y)

#====================================================
### Original MobileNetV3 Inverted Residual (for ablation)
class MobileNetV3_InvertedResidual(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(MobileNetV3_InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # h_swish() if use_hs else nn.ReLU(inplace=True),
                nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # h_swish() if use_hs else nn.ReLU(inplace=True),
                nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # h_swish() if use_hs else nn.ReLU(inplace=True),
                nn.Hardswish(inplace=True) if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y

#====================================================
##### Inverted Residual with single AKMixDW (for ablation)

#class MobileNetV3_InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs,
#                  use_akm: bool = True, k_small: int = 3, k_big: int = 5):
#         super().__init__()
#         assert stride in [1, 2]
#         Act = nn.Hardswish if use_hs else nn.ReLU
#         self.identity = (stride == 1 and inp == oup)

#         def dw_stage(c, k, s):
#             if use_akm and stride == 2 and c >= 24:                 # optional skip on tiny channels
#                 return AKMixDW(c, k_small, k_big, s, act=Act)       # already BN+act inside
#             else:
#                 return nn.Sequential(
#                     nn.Conv2d(c, c, k, s, (k - 1)//2, groups=c, bias=False),
#                     nn.BatchNorm2d(c),
#                     Act(inplace=True),
#                 )
#
#         if inp == hidden_dim:
#             # no-expand variant
#             self.conv = nn.Sequential(
#                 dw_stage(hidden_dim, kernel_size, stride),          # DW or AKMixDW
#                 (SELayer(hidden_dim) if use_se else nn.Identity()), # attention
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),    # PW-linear
#                 nn.BatchNorm2d(oup),
#             )
#         else:
#             # expand → DW/AKMix → (SE) → project (linear)
#             self.conv = nn.Sequential(
#                 nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),    # PW expand
#                 nn.BatchNorm2d(hidden_dim),
#                 Act(inplace=True),
#                 dw_stage(hidden_dim, kernel_size, stride),          # DW or AKMixDW
#                 (SELayer(hidden_dim) if use_se else nn.Identity()), # attention
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),    # PW-linear
#                 nn.BatchNorm2d(oup),
#             )
#
#         # zero-init last BN gamma for identity blocks (stability)
#         if self.identity and isinstance(self.conv[-1], nn.BatchNorm2d):
#             nn.init.zeros_(self.conv[-1].weight)
#
#     def forward(self, x):
#         y = self.conv(x)
#         return x + y if self.identity else y

#====================================================

##### Inverted Residual with single AKMixDW + Partial Conv (for ablation)
#class MobileNetV3_InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs,
#                  use_akm: bool = True,
#                  k_small: int = 3, k_big: int = 5,
#                  ps_ratio: float = 0.25, ps_shuffle: bool = True):
#         super().__init__()
#         assert stride in [1, 2]
#         Act = nn.Hardswish if use_hs else nn.ReLU
#         self.identity = (stride == 1 and inp == oup)

#         def dw_stage(c, k, s):
#             # ✅ Use the local `s`, not the outer `stride`
#             if use_akm and s == 2 and c >= 24:
#                 # ✅ Replace old AKMixDW with the new fused module
#                 return AKMixDW_PartialSCFuse(
#                     c=c,
#                     stride=s,
#                     k_small=k_small,
#                     k_big=k_big,
#                     ratio=ps_ratio,
#                     shuffle=ps_shuffle,
#                     act=Act
#                 )
#             else:
#                 return nn.Sequential(
#                     nn.Conv2d(c, c, k, s, (k - 1)//2, groups=c, bias=False),
#                     nn.BatchNorm2d(c),
#                     Act(inplace=True),
#                 )

#         if inp == hidden_dim:
             # no-expand variant
#             self.conv = nn.Sequential(
#                 dw_stage(hidden_dim, kernel_size, stride),
#                 (SELayer(hidden_dim) if use_se else nn.Identity()),
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#         else:
             # expand → DW/AKMix → (SE) → project (linear)
#             self.conv = nn.Sequential(
#                 nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 Act(inplace=True),
#                 dw_stage(hidden_dim, kernel_size, stride),
#                 (SELayer(hidden_dim) if use_se else nn.Identity()),
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )

#         if self.identity and isinstance(self.conv[-1], nn.BatchNorm2d):
#             nn.init.zeros_(self.conv[-1].weight)

#     def forward(self, x):
#         y = self.conv(x)
#         return x + y if self.identity else y

