import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numpy as np
import cv2
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath
from torchvision.ops import DeformConv2d
import pdb

##########################################################################
# Basic modules
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)


class UshapeN(nn.Module):
    # def __int__(self, in_nfeats, out_nfeats, nfeats=16):
    def __init__(self, in_nfeats, out_nfeats, nfeats=16):
        super(UshapeN, self).__init__()

        self.conv1 = conv(in_nfeats, nfeats, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv(nfeats, nfeats, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv(nfeats, nfeats, 3)
        # self.up1 = F.upsample(scale_factor=2)
        self.conv4 = conv(nfeats, nfeats, 3)
        # self.up2= F.upsample(scale_factor=2)
        self.conv5 = conv(nfeats, nfeats, 3)
        self.conv6 = conv(nfeats, out_nfeats, 1)

    def forward(self, x):
        # pdb.set_trace()
        x1 = nn.ReLU(inplace=True)(self.conv1(x))
        x2 = nn.ReLU(inplace=True)(self.conv2(self.pool1(x1)))
        x3 = nn.ReLU(inplace=True)(self.conv3(self.pool2(x2)))
        x4 = nn.ReLU(inplace=True)(self.conv4(F.upsample(x3, scale_factor=2)))
        x5 = nn.ReLU(inplace=True)(self.conv5(F.upsample(x4, scale_factor=2)))
        x6 = self.conv6(x5)
        return x6


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, 64, kernel_size, bias=bias))
            else:
                m.append(conv(64, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

    
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class Deform_Conv_V1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                    padding=1, dilation=1, groups=1, offset_group=1):
        super(Deform_Conv_V1, self).__init__()
        offset_channels = 2 * kernel_size * kernel_size
        self.conv_offset = nn.Conv2d(
            in_channels,
            offset_channels * offset_group,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            )
        # pdb.set_trace()
        # self.conv_offset = UshapeN(in_channels, offset_channels, 16)


        self.DCN_V1 = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False
            )

    def forward(self, x):
        offset = self.conv_offset(x)
        # pdb.set_trace()
        y = self.DCN_V1(x, offset)
        y = nn.ReLU(inplace=True)(y)
        return y


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class UpSample2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample2, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.input_resolution = input_resolution
        self.n_levels = 3
        chunk_dim = conv_dim // self.n_levels

        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        # self.conv_block = nn.Sequential(
        #         nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
        #         nn.ReLU(True),
        #         nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
        #         )

        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        self.fusion = nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)

        # self.conv_block1 = Deform_Conv_V1(self.conv_dim, self.conv_dim)
        # self.conv_block2 = Deform_Conv_V1(self.conv_dim, self.conv_dim)

        # self.conv1_3 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim, 1, 1, 0, bias=True)

    def forward(self, x):
        # pdb.set_trace()
        # x1 = self.conv1_3(x1)
        h, w = x.size()[-2:]
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        # conv_x = self.conv_block(conv_x)+ conv_x
        # conv_x2 = self.conv_block2(conv_x1)

        xc = conv_x.chunk(self.n_levels, dim=1)
        conv_out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            conv_out.append(s)

        conv_x = torch.cat(conv_out, dim=1)
        conv_x = self.fusion(conv_x)
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        v_x = Rearrange('b c h w -> b h w c')(conv_x)
        trans_x = self.trans_block(trans_x, v_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res

        return x


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x, v):
        x = x + self.drop_path(self.msa(self.ln1(x), self.ln1(v)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.embedding_layer = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.kv_embedding = nn.Linear(self.input_dim, 2*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def forward(self, x, kv):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        # pdb.set_trace()
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        kv = rearrange(kv, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        kv = rearrange(kv, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        x = self.embedding_layer(x)
        kv = self.kv_embedding(kv)
        k, v = rearrange(kv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(2, dim=0)
        # pdb.set_trace()
        q = rearrange(x, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]


class Denoiser(nn.Module):
    def __init__(self, features):
        super(Denoiser, self).__init__()
        layers = []
        self.head = conv(features*2, features, 3)
        for i in range(5):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.out = CALayer(features)

    def forward(self, x, y):
        if y is not None:
            y = torch.cat((x, y), dim=1)
            y = self.head(y)
        else:
            y = x
        x = self.out(self.layers(y)) + x
        return x


class Fusion(nn.Module):
    def __init__(self, features, num):
        super(Fusion, self).__init__()
        # layers = []
        # self.head = conv(features*num, features, 3)
        self.body = CALayer(features*num, reduction=1)
        self.tail = conv(features * num, 3, 3)

    def forward(self, x1, x2, x3):
        # pdb.set_trace()
        x = torch.cat((x1, x2, x3), dim=1)
        # x = torch.cat(*x, dim=1)
        x = self.tail(self.body(x))
        return x


class DPWM(nn.Module):
    def __init__(self, n_feats):
        super(DPWM, self).__init__()
        drop_path_rate = 0.0
        config = [2, 2, 2, 2, 2, 2, 2]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        self.head = nn.Sequential(ConvTransBlock(n_feats // 2, n_feats // 2, 6, 8, dpr[0]),
                                  ConvTransBlock(n_feats // 2, n_feats // 2, 6, 8, dpr[0]),
                                  ConvTransBlock(n_feats // 2, n_feats // 2, 6, 8, dpr[0]),)
        # self.out = ConvTransBlock(n_feats // 2, n_feats // 2, 6, 8, dpr[0])

        # self.fusion = conv(n_feats*2, n_feats, 3, bias=False)
        modules_body = []
        modules_body.append(conv(n_feats*2, n_feats, 3, bias=False))
        modules_body.append(nn.PReLU())
        modules_body.append(conv(n_feats, n_feats, 3, bias=False))
        self.fusion = nn.Sequential(*modules_body)
        # self.out = Den_Wei(n_feats)

    def forward(self, x, x_feats):

        if x_feats is not None:
            x_fusion = torch.cat((x, x_feats), dim=1)
            x = self.fusion(x_fusion)
        else:
            x_feats = x
        x = self.head(x)
        # x = self.out(x)
        return x


##########################################################################
## DGUNet_plus  
class MRUNet(nn.Module):
    def __init__(self, n_feat=48, kernel_size=3, reduction=4, bias=False, depth=5):
        super(MRUNet, self).__init__()
        # Extract Shallow Features
        act = nn.PReLU()

        self.shallow_feat_1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),)
        self.shallow_feat_2 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),)
        self.shallow_feat_3 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),)
        self.shallow_feat_4 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),)
        
        # Gradient Descent Module (GDM)
        self.phi_1 = ResBlock(default_conv,3,3)
        self.phit_1 = ResBlock(default_conv,3,3)
        self.phi_2 = ResBlock(default_conv,3,3)
        self.phit_2 = ResBlock(default_conv,3,3)
        self.phi_3 = ResBlock(default_conv,3,3)
        self.phit_3 = ResBlock(default_conv,3,3)
        self.phi_4 = ResBlock(default_conv,3,3)
        self.phit_4 = ResBlock(default_conv,3,3)

        # scale 2
        self.phi_12 = ResBlock(default_conv, 3, 3)
        self.phit_12 = ResBlock(default_conv, 3, 3)
        self.phi_22 = ResBlock(default_conv, 3, 3)
        self.phit_22 = ResBlock(default_conv, 3, 3)
        self.phi_32 = ResBlock(default_conv, 3, 3)
        self.phit_32 = ResBlock(default_conv, 3, 3)
        self.phi_42 = ResBlock(default_conv, 3, 3)
        self.phit_42 = ResBlock(default_conv, 3, 3)

        self.phi_13 = ResBlock(default_conv, 3, 3)
        self.phit_13 = ResBlock(default_conv, 3, 3)
        self.phi_23 = ResBlock(default_conv, 3, 3)
        self.phit_23 = ResBlock(default_conv, 3, 3)
        self.phi_33 = ResBlock(default_conv, 3, 3)
        self.phit_33 = ResBlock(default_conv, 3, 3)
        self.phi_43 = ResBlock(default_conv, 3, 3)
        self.phit_43 = ResBlock(default_conv, 3, 3)

        #  Super-Parameter
        self.r_1 = nn.Parameter(torch.Tensor([0.5]))
        self.r_2 = nn.Parameter(torch.Tensor([0.5]))
        self.r_3 = nn.Parameter(torch.Tensor([0.5]))
        self.r_4 = nn.Parameter(torch.Tensor([0.5]))

        self.r_12 = nn.Parameter(torch.Tensor([0.5]))
        self.r_22 = nn.Parameter(torch.Tensor([0.5]))
        self.r_32 = nn.Parameter(torch.Tensor([0.5]))
        self.r_42 = nn.Parameter(torch.Tensor([0.5]))

        self.r_13 = nn.Parameter(torch.Tensor([0.5]))
        self.r_23 = nn.Parameter(torch.Tensor([0.5]))
        self.r_33 = nn.Parameter(torch.Tensor([0.5]))
        self.r_43 = nn.Parameter(torch.Tensor([0.5]))

        self.stage_1 = DPWM(n_feat)
        self.stage_2 = DPWM(n_feat)
        self.stage_3 = DPWM(n_feat)
        self.stage_4 = DPWM(n_feat)

        self.up1 = UpSample(n_feat, n_feat)
        self.up2 = UpSample(n_feat, n_feat)
        self.up3 = UpSample(n_feat, n_feat)
        self.up4 = UpSample(n_feat, n_feat)

        self.up12 = UpSample2(n_feat, n_feat)
        self.up22 = UpSample2(n_feat, n_feat)
        self.up32 = UpSample2(n_feat, n_feat)
        self.up42 = UpSample2(n_feat, n_feat)

        self.fusion1 = Fusion(n_feat, 3)
        self.fusion2 = Fusion(n_feat, 3)
        self.fusion3 = Fusion(n_feat, 3)
        self.fusion4 = Fusion(n_feat, 3)

    def forward(self, img):
        lum = img
        lum_down_2 = F.interpolate(img, scale_factor=0.5)
        lum_down_3 = F.interpolate(img, scale_factor=0.25)

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## GDM
        lum_1 = self.phi_1(lum) - lum
        x1_lum = lum - self.r_1*self.phit_1(lum_1)
        # PMM
        x1_lum = self.shallow_feat_1(x1_lum)
        x1_features_lum = self.stage_1(x1_lum, None)

        lum_12 = self.phi_12(lum_down_2) - lum_down_2
        x1_lum2 = lum_down_2 - self.r_12 * self.phit_12(lum_12)
        # PMM
        x1_lum2 = self.shallow_feat_1(x1_lum2)
        x1_features_lum2 = self.stage_1(x1_lum2, None)

        lum_13 = self.phi_13(lum_down_3) - lum_down_3
        x1_lum3 = lum_down_3 - self.r_13 * self.phit_13(lum_13)
        # PMM
        x1_lum3 = self.shallow_feat_1(x1_lum3)
        x1_features_lum3 = self.stage_1(x1_lum3, None)

        stage1_lum2_t = self.up1(x1_features_lum2)
        stage1_lum3_t = self.up12(x1_features_lum3)
        stage1_lum = self.fusion1(x1_features_lum, stage1_lum2_t, stage1_lum3_t)
        stage1_lum2 = F.interpolate(stage1_lum, scale_factor=0.5)
        stage1_lum3 = F.interpolate(stage1_lum, scale_factor=0.25)

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## GDM

        lum_2 = self.phi_2(stage1_lum) - lum
        x2_lum = stage1_lum - self.r_2 * self.phit_2(lum_2)
        # PMM
        x2_lum = self.shallow_feat_2(x2_lum)
        x2_features_lum = self.stage_2(x2_lum, x1_features_lum)

        lum_22 = self.phi_22(stage1_lum2) - lum_down_2
        x2_lum2 = stage1_lum2 - self.r_22 * self.phit_22(lum_22)
        # PMM
        x2_lum2 = self.shallow_feat_2(x2_lum2)
        x2_features_lum2 = self.stage_2(x2_lum2, x1_features_lum2)

        lum_23 = self.phi_23(stage1_lum3) - lum_down_3
        x2_lum3 = stage1_lum3 - self.r_23 * self.phit_23(lum_23)
        # PMM
        x2_lum3 = self.shallow_feat_2(x2_lum3)
        x2_features_lum3 = self.stage_2(x2_lum3, x1_features_lum3)

        stage2_lum2_t = self.up2(x2_features_lum2)
        stage2_lum3_t = self.up22(x2_features_lum3)
        stage2_lum = self.fusion2(x2_features_lum, stage2_lum2_t, stage2_lum3_t)
        stage2_lum2 = F.interpolate(stage2_lum, scale_factor=0.5)
        stage2_lum3 = F.interpolate(stage2_lum, scale_factor=0.25)

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## GDM
        lum_3 = self.phi_3(stage2_lum) - lum
        x3_lum = stage2_lum - self.r_3 * self.phit_3(lum_3)
        # PMM
        x3_lum = self.shallow_feat_3(x3_lum)
        x3_features_lum = self.stage_3(x3_lum, x2_features_lum)

        lum_32 = self.phi_32(stage2_lum2) - lum_down_2
        x3_lum2 = stage2_lum2 - self.r_32 * self.phit_32(lum_32)
        # PMM
        x3_lum2 = self.shallow_feat_3(x3_lum2)
        x3_features_lum2 = self.stage_3(x3_lum2, x2_features_lum2)

        lum_33 = self.phi_33(stage2_lum3) - lum_down_3
        x3_lum3 = stage2_lum3 - self.r_33 * self.phit_33(lum_33)
        # PMM
        x3_lum3 = self.shallow_feat_3(x3_lum3)
        x3_features_lum3 = self.stage_3(x3_lum3, x2_features_lum3)

        stage3_lum2_t = self.up3(x3_features_lum2)
        stage3_lum3_t = self.up32(x3_features_lum3)
        stage3_lum = self.fusion3(x3_features_lum, stage3_lum2_t, stage3_lum3_t)
        stage3_lum2 = F.interpolate(stage3_lum, scale_factor=0.5)
        stage3_lum3 = F.interpolate(stage3_lum, scale_factor=0.25)

        ##-------------------------------------------
        ##-------------- Stage 4---------------------
        ##-------------------------------------------
        ## GDM
        lum_4 = self.phi_4(stage3_lum) - lum
        x4_lum = stage3_lum - self.r_4 * self.phit_4(lum_4)
        # PMM
        x4_lum = self.shallow_feat_4(x4_lum)
        x4_features_lum = self.stage_4(x4_lum, x3_features_lum)

        lum_42 = self.phi_42(stage3_lum2) - lum_down_2
        x4_lum2 = stage3_lum2 - self.r_42 * self.phit_42(lum_42)
        # PMM
        x4_lum2 = self.shallow_feat_4(x4_lum2)
        x4_features_lum2 = self.stage_4(x4_lum2, x3_features_lum2)

        lum_43 = self.phi_43(stage3_lum3) - lum_down_3
        x4_lum3 = stage3_lum3 - self.r_43 * self.phit_43(lum_43)
        # PMM
        x4_lum3 = self.shallow_feat_4(x4_lum3)
        x4_features_lum3 = self.stage_4(x4_lum3, x3_features_lum3)

        stage4_lum2_t = self.up4(x4_features_lum2)
        stage4_lum3_t = self.up42(x4_features_lum3)
        stage4_lum = self.fusion4(x4_features_lum, stage4_lum2_t, stage4_lum3_t)

        return [stage4_lum, stage3_lum, stage2_lum, stage1_lum]
