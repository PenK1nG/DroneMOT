from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

# from lib.models.networks.DCNv2_latest.dcn_v2 import DCN
# from lib.models.networks.DCNv2.dcn_v2_amp import DCN as DCN
from lib.models.networks.DCNv2.dcn_v2 import DCN

from lib.models.networks.correlation_package.correlation import Correlation


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        # levels：[1, 1, 1, 2, 2, 1],
        # channels：[16, 32, 64, 128, 256, 512],
        # block = BasicBlock
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        # 1088*608*3 -> 1088*608*16
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        # 1088*608*16 -> 1088*608*16
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        # 1088*608*16 -> 544*304*32
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        # if x.dtype == torch.float16:
        # x = x.float()
        # print(x.dtype)
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        # first_level=2, channels[2:]=[64, 128, 256, 512], scales=[1, 2, 4, 8]
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x

import torch.nn.functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch import nn,Tensor
import copy
from typing import Optional, Any
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout

class DroneAttention(nn.Module):
    def __init__(self, d_model: int = 512, 
                 nhead: int = 8, 
                 num_encoder_layers: int = 6, 
                 num_decoder_layers: int = 6, 
                 dim_feedforward: int = 512, 
                 dropout: float = 0.1, 
                 activation: str = "relu"):
        super(DroneAttention, self).__init__()
        Spatial_layer = SpatialAttentionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        Spatial_norm = nn.LayerNorm(d_model)
        self.Spatial = SpatialAttention(Spatial_layer, num_encoder_layers, Spatial_norm)

        Temporal_layer = TemporalAttentionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        Temporal_norm = nn.LayerNorm(d_model)
        self.Temporal = TemporalAttention(Temporal_layer, num_decoder_layers, Temporal_norm)

        self.d_model = d_model
        self.nhead = nhead
        
        # self.downsample = nn.AvgPool2d(kernel_size=8, stride=8)
    
    # def forward(self, src: Tensor, buffer: Tensor):
    def forward(self, src: Tensor, presrc: Tensor, prehm):
        src = src.permute(0,2,1)
        presrc = presrc.permute(0, 2, 1)
        # prehm = self.downsample(prehm)
        
        pre_memory = self.Spatial(presrc, presrc)
        memory = self.Spatial(src, src)
        # k,v = memory, q = pre_memory
        output = self.Temporal(memory, pre_memory, prehm)

        return output

class SpatialAttention(nn.Module):
# class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(SpatialAttention, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor,srcc: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TemporalAttention(nn.Module):
# class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TemporalAttention, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor,srcc: Tensor, prehm, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, srcc, prehm, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class SpatialAttentionLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=384, dropout=0.1, activation="relu"):
        super(SpatialAttentionLayer, self).__init__()
        self.self_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn3 = MultiheadAttention(d_model, nhead, dropout=dropout)
        channel=dim_feedforward//2
        # self.modulation=TIF(channel)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
    
        
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(SpatialAttentionLayer, self).__setstate__(state)

    def forward(self, src: Tensor,srcc: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        # b,c,s=src.permute(0,2,1).size()
        
        
        src1 = self.self_attn1(srcc, src, src, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        # print(1)
        srcs1 = src + self.dropout1(src1)
        srcs1 = self.norm1(srcs1)
        
        src2 = self.self_attn2(srcs1, srcs1, srcs1, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        srcs2 = srcs1 + self.dropout2(src2)
        srcs2 = self.norm2(srcs2)
        
        # src=self.modulation(srcs2.view(b,c,int(s**0.5),int(s**0.5))\
        #                      ,srcs1.contiguous().view(b,c,int(s**0.5),int(s**0.5))).view(b,c,-1).permute(2, 0, 1)
        
        src3 = self.self_attn3(srcs2, srcs2, srcs2, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        srcs1 = src + self.dropout3(src3)
        srcs1 = self.norm3(srcs1)

        return srcs1

class TemporalAttentionLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=384, dropout=0.1, activation="relu"):
        super(TemporalAttentionLayer, self).__init__()
        self.self_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn3 = MultiheadAttention(d_model, nhead, dropout=dropout)
        channel=dim_feedforward//2
        # self.modulation=TIF(channel)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
    
        
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1)
        self.linear = nn.Linear(64 + 64, 64)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.alpha = nn.Parameter(torch.ones(1, 10, 1, 1))
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TemporalAttentionLayer, self).__setstate__(state)

    def forward(self, src: Tensor,srcc: Tensor, prehm, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        # b,c,s=src.permute(0,2,1).size()
        
        # print(prehm.size())
        b, _, w, h = prehm.size()
        refined_heatmap = (prehm * self.alpha).sum(dim=1, keepdim=True)
        # print(refined_heatmap.size())
        _, c, _, _ = srcc.view(b,-1,w,h).size()
        refined_heatmap_expanded = refined_heatmap.repeat(1, c, 1, 1)
        # print(refined_heatmap_expanded.size())
        prehm = refined_heatmap_expanded * srcc.view(b,-1,w,h)
        x = self.conv1(prehm)
        weights = self.gap(x).squeeze(-1).squeeze(-1)

        
        
        
        src1 = self.self_attn1(srcc, src, src, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        # print(1)
        srcs1 = src + self.dropout1(src1)
        srcs1 = self.norm1(srcs1)
        
        src2 = self.self_attn2(srcs1, srcs1, srcs1, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        srcs2 = srcs1 + self.dropout2(src2)
        srcs2 = self.norm2(srcs2)

        srcs21 = torch.cat([srcs1, srcs2], dim=-1)
        srcs21 = self.linear(srcs21)
        weights = weights.view(-1, 1, 64)
        srcs2 = srcs21 + srcs21 * weights
        
        # src=self.modulation(srcs2.view(b,c,int(s**0.5),int(s**0.5))\
        #                      ,srcs1.contiguous().view(b,c,int(s**0.5),int(s**0.5))).view(b,c,-1).permute(2, 0, 1)
        
        src3 = self.self_attn3(srcs2, srcs2, srcs2, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        srcs1 = src + self.dropout3(src3)
        srcs1 = self.norm3(srcs1)

        return srcs1


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        mask = torch.zeros((x.size(0), x.size(2), x.size(3)), dtype=torch.bool, device=x.device)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        # dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos



class DLATransformer(nn.Module):
    def __init__(self,
                 base_name,
                 heads,
                 pretrained,
                 down_ratio,
                 final_kernel,
                 last_level,
                 head_conv,
                 out_channel=0):
        super(DLATransformer, self).__init__()
        assert down_ratio in [2, 4, 8, 16] # 4
        self.first_level = int(np.log2(down_ratio)) # 2
        self.last_level = last_level # 5
        self.base = globals()[base_name](pretrained=pretrained) # 构建dla34网络
        channels = self.base.channels # [16, 32, 64, 128, 256, 512]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))] # scales = [1 2 4 8]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales) # 构建dla_up网络
        if out_channel == 0:
            out_channel = channels[self.first_level] # 64

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        self.Attention = DroneAttention(d_model = channels[self.first_level],
                                        dim_feedforward = channels[self.first_level],
                                        num_encoder_layers = 4,
                                        num_decoder_layers = 6)
        self.postion_encoding = PositionEmbeddingSine(num_pos_feats=(channels[self.first_level]//2),
                                                      normalize=True)
        self.heads = heads
        
        self.conv2d = nn.Conv2d(channels[self.first_level], channels[self.first_level], kernel_size=1, stride=1, padding=0)
        
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              # chanels[self.first_level]=64, head_conv=256, classes=self.heads[head]   
              fc = nn.Sequential(
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
            
    def forward(self, pre_x, x, pre_hm, hm):
        pre_x = self.base(pre_x)
        x = self.base(x)
        x = self.dla_up(x)
        pre_x = self.dla_up(pre_x)
        y = []
        pre_y = []
        for i in range(self.last_level - self.first_level): # 0, 1, 2
            y.append(x[i].clone())
            pre_y.append(pre_x[i].clone())
        self.ida_up(y, 0, len(y))
        self.ida_up(pre_y, 0, len(pre_y))
        
        # x_last_pos = self.postion_encoding(x_last)
        y_last_pos = self.postion_encoding(y[-1])
        pre_y_last_pos = self.postion_encoding(pre_y[-1])
        y_last = y[-1] + y_last_pos
        pre_y_last = pre_y[-1] + pre_y_last_pos
        b, c, w, h=y_last.size()
        # hm = torch.cat([x_last, hm], dim=1)
        
        x_transformer = self.Attention(y_last.view(b,c,-1), pre_y_last.view(b,c,-1), pre_hm)
        pre_x_transformer = self.Attention(pre_y_last.view(b,c,-1), y_last.view(b,c,-1), hm)
        x_transformer = self.conv2d(x_transformer.view(b,c,w,h))
        pre_x_transformer = self.conv2d(pre_x_transformer.view(b,c,w,h))

        z = {}
        pre_z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x_transformer)
            pre_z[head] = self.__getattr__(head)(pre_x_transformer)
        return [pre_z, z]
        
def get_mot_net(num_layers, heads, head_conv=256, down_ratio=4):
    model = DLATransformer('dla{}'.format(num_layers), heads,
                 pretrained=True,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv)
    return model
