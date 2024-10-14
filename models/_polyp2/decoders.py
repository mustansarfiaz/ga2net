import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import os
from math import sqrt
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from .deform_conv import DeformableConv2d as dcn2d

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (dcn2d, nn.GELU, nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)

class LayerNorm_a(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    def initialize(self):
        weight_init(self)
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
    def initialize(self):
        weight_init(self)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias,mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
    
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x,mask=None):
        b,c,h,w = x.shape
        q=self.qkv1conv(self.qkv_0(x))
        k=self.qkv2conv(self.qkv_1(x))
        v=self.qkv3conv(self.qkv_2(x))
        if mask is not None:
            q=q*mask
            k=k*mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)

class MSA_head(nn.Module):
    def __init__(self, mode='dilation',dim=128, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias,mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x,mask=None):
        x = x + self.attn(self.norm1(x),mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)

class mla_layer(nn.Module):
    """
    layer attention module v7: groupwise operation of v2
    when groups = channels, channelwise (Q(K)' is then pointwise(channelwise) multiplication)
    
    Args:
        input_dim: input channel c (output channel is the same)
        k_size: channel dimension of Q, K
        input : [b, c, h, w]
        output: [b, c, h, w]
        
        Wq, Wk: conv1d
        Wv: conv2d
        Q: [b, 1, c]
        K: [b, 1, c]
        V: [b, c, h, w]
    """
    def __init__(self, input_dim, groups=None, dim_pergroup=None, k_size=None):
        super(mla_layer, self).__init__()
        self.input_dim = input_dim
        
        if (groups == None) and (dim_pergroup == None):
            raise ValueError("arguments groups and dim_pergroup cannot be None at the same time !")
        elif dim_pergroup != None:
            groups = int(input_dim / dim_pergroup)
        else:
            groups = groups
        self.groups = groups
        
        if k_size == None:
            t = int(abs((log(input_dim, 2) + 1) / 2.))
            k_size = t if t % 2 else t+1
        self.k_size = k_size
    
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Wq = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wk = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False) 
        self._norm_fact = 1 / sqrt(input_dim / groups)
        # nn.ReLU(inplace=True)
        # self.depthwise1 = nn.Conv2d(channel, channel, kernel_size=1, groups=channel, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # [b, c, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2) # [b, 1, c]
        
        Q = self.Wq(y) # Q: [b, 1, c] 
        K = self.Wk(y) # K: [b, 1, c]
        V = self.Wv(x) # V: [b, c, h, w]
        # Q = Q.chunk(self.groups, dim = -1) # a tuple of g * [b, 1, c/g]
        # K = K.chunk(self.groups, dim = -1)
        # V = V.chunk(self.groups, dim = 1) # a tuple of g * [b, c/g, h, w]
        Q = Q.view(b, self.groups, 1, int(c/self.groups)) # [b, g, 1, c/g]
        K = K.view(b, self.groups, 1, int(c/self.groups)) # [b, g, 1, c/g]
        V = V.view(b, self.groups, int(c/self.groups), h, w) # [b, g, c/g, h, w]
        # Q.is_contiguous()
        
        atten = torch.einsum('... i d, ... j d -> ... i j', Q, K) * self._norm_fact
        # atten.size() # [b, g, 1, 1]
    
        atten = self.sigmoid(atten.view(b, self.groups, 1, 1, 1)) # [b, g, 1, 1, 1]
        output = V * atten.expand_as(V) # [b, g, c/g, h, w]
        output = output.view(b, c, h, w)
        
        return output    

class mrla_module(nn.Module):
    dim_pergroup = 32
    
    def __init__(self, input_dim):
        super(mrla_module, self).__init__()
        self.mla = mla_layer(input_dim=input_dim, dim_pergroup=self.dim_pergroup)
        self.lambda_t = nn.Parameter(torch.randn(input_dim, 1, 1))  # nn.Parameter(torch.zeros(1, 1, embed_dim))
        
    def forward(self, xt, ot_1):
        atten_t = self.mla(xt)
        out = atten_t + self.lambda_t.expand_as(ot_1) * ot_1 # o_t = atten(x_t) + lambda_t * o_{t-1}
        return out
    
class MSA_module(nn.Module):
    def __init__(self, dim=128):
        super(MSA_module, self).__init__()
        kernel_size = 5
        self.proj = nn.Conv2d(2*dim, dim, kernel_size=3, padding=1)

        self.dilconv = nn.Sequential(
            LayerNorm_a(dim*2, eps=1e-6, data_format="channels_first"),            
            nn.Conv2d(dim*2, dim, kernel_size=3,stride=1,dilation=2,padding=2 ,bias=True),
            nn.GELU()
            )
        
        self.dwconv = nn.Sequential(
            LayerNorm_a(dim, eps=1e-6, data_format="channels_first"),            
            nn.Conv2d(dim, dim, kernel_size=kernel_size,stride=1,padding=kernel_size//2 ,bias=False,groups=dim),
            nn.GELU(),
            )
        
        self.dwconv_f = nn.Sequential(
            LayerNorm_a(dim, eps=1e-6, data_format="channels_first"),            
            nn.Conv2d(dim, dim, kernel_size=kernel_size,stride=1,padding=kernel_size//2 ,bias=False,groups=dim),
            nn.GELU(),
            )
        
        self.conv = nn.Sequential(
            LayerNorm_a(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, dim, kernel_size=1,stride=1,padding=0,bias=False,groups=dim),
        )
        self.mlp = ConvMLP(dim)
        self.act = nn.GELU()
        self.sid_enh = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
    
    def forward(self, x, side_x, mask):
        N,C,H,W = x.shape

        xd = self.dilconv(torch.cat((x, side_x),1))
        mask_d = mask.detach()
        mask_d = torch.sigmoid(mask_d)
        xf = mask_d*x
        xf = self.dwconv_f(xf)
        x1 = self.dwconv(side_x)
   
        x = torch.cat((xf, x1),1)
        x = xd + self.proj(x)

        x = x + self.mlp(x)
        return x
    
    def initialize(self):
        weight_init(self)

class ConvMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm_a(dim, eps=1e-6, data_format="channels_first")
        
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()
        
    def initialize(self):
        weight_init(self)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x
    
class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()

        self.conv1 = nn.Conv2d(channels*3, channels+3, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels*3)

        self.conv2 = nn.Conv2d(channels*2, channels*2, kernel_size=5, stride=1, padding=2, bias=True, groups = channels)
        self.bn2 = nn.BatchNorm2d(channels*2)

        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(channels*2)
        self.focal_level = 3
        self.act = nn.GELU()
        self.h = nn.Conv2d(3*channels, channels, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, input1, input2, input3):
        B, C, H, W = input3.shape

        if input1.size()[2:] != input3.size()[2:]:
            input1 = F.interpolate(input1, size=input3.size()[2:], mode='bilinear')
        if input2.size()[2:] != input3.size()[2:]:
            input2 = F.interpolate(input2, size=input3.size()[2:], mode='bilinear')

        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.act(self.conv1(self.bn1(fuse)))
        
        q, gates = torch.split(fuse, (C, 3), 1)

        # context aggreation
        ctx_all = 0 
        input1 = input1*(gates[:,0,:,:].unsqueeze(1))
        input2 = input2*(gates[:,1,:,:].unsqueeze(1))
        input3 = input3*(gates[:,2,:,:].unsqueeze(1))
        ctx_all = torch.cat([input1,input2,input3],dim=1)

        modulator = self.h(ctx_all)
        x_out = torch.cat([q,modulator],dim=1)
        
        fuse = self.act(self.conv2(self.bn2(x_out)))
        fuse = self.conv3(self.bn3(fuse))
        return fuse

    def initialize(self):
        weight_init(self)

class DFA(nn.Module):
    """ Enhance the feature diversity.
    """
    def __init__(self, x, y):
        super(DFA, self).__init__()
        
        self.oriConv = nn.Conv2d(x, y, kernel_size=3, stride=1, padding=1)
        self.atrConv = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=3, dilation=2, padding=2, stride=1), nn.BatchNorm2d(y), nn.PReLU()
        )           
        self.conv2d = nn.Conv2d(y*2, y, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(y)
        self.initialize()

    def forward(self, f):
        p1 = self.oriConv(f)
        
        p3 = self.atrConv(f)
        p  = torch.cat((p1,  p3), 1)
        p  = F.gelu(self.bn2d(self.conv2d(p)))

        return p

    def initialize(self):
        #pass
        weight_init(self)
        
class Attention_2_branches(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn = None
    def forward(self, x, x2):
        B, N, C = x2.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn2 = (q @ k2.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)        
        self.attn = attn
        attn = self.attn_drop(attn)
        attn2 = self.attn_drop(attn2)
        

        x = ( attn @ v ) 
        x2 = ( attn2 @ v2 )
        

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x2 = x2.transpose(1, 2).reshape(B, N, C)
        x2 = self.proj(x2)
        x2 = self.proj_drop(x2)

        return x, x2
    
class PSCA(nn.Module):
    """ Progressive Spectral Channel Attention (PSCA) 
    """

    def __init__(self, d_model, ratio=4):
        super().__init__()
        d_ff = d_model*ratio
        self.w_1 = nn.Conv2d(d_model, d_ff, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.w_2 = nn.Conv2d(d_ff, d_model, 1, bias=False)
        self.w_3 = nn.Conv2d(d_model, d_model, 1, bias=False)
        self.pos = nn.Conv2d(d_ff, d_ff, 3, padding=1, groups=d_ff)
        self.act = nn.GELU()

    def initialize(self):
        weight_init(self)
        #nn.init.zeros_(self.w_3.weight)

    def forward(self, x):
        res = x
        x = self.w_3(x) * x + x
        x = self.w_1(x)
        x = x + self.act(self.pos(x))
        x = self.w_2(x)
        return x

################# Dynamic kernel for each stage ##################
class AG_new(nn.Module):
    def __init__(self, F_g, F_l,kernel_size=7):
        super(AG_new,self).__init__()
        self.dwconv = nn.Sequential(
            LayerNorm_a(F_g, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(F_g, F_g, kernel_size=1,stride=1,padding=0,bias=True),
            nn.GELU(),
            nn.Conv2d(F_g, F_g, kernel_size=kernel_size,stride=1,padding=kernel_size//2 ,bias=True,groups=F_g),
            )
        
        # self.conv = nn.Sequential(
        #     nn.Conv2d(F_l, F_l, kernel_size=1,stride=1,padding=0,bias=True,groups=F_g),
        # )
        self.proj = nn.Conv2d(2*F_l, F_l, kernel_size=5,stride=1,padding=2,bias=True,groups=F_l)

        #self.g_enhance = nn.Conv2d(F_l, F_l, 3, padding=1, groups=F_l)
        
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        
    def initialize(self):
        weight_init(self)
    
        #self.relu = nn.ReLU(inplace=True)

    def forward(self,x,g):
        #print(g.shape)
        #print(x.shape)
        #print('*********************8')
        g1 = self.dwconv(g)
        
        x1 = x #self.conv(x)  
        cga = torch.cat([g1,x1],dim=1)
       
        out = self.sigmoid(self.proj(cga))*g
        #out = self.psa(out)+out
        return out

class Decoder(nn.Module):
    def __init__(self, channels, dims, n_class=1):
        super(Decoder, self).__init__()
        self.n_classes = n_class

        self.side_conv1 = nn.Conv2d(dims[0], channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2 = nn.Conv2d(dims[1], channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3 = nn.Conv2d(dims[2], channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4 = nn.Conv2d(dims[3], channels, kernel_size=3, stride=1, padding=1)

        self.conv_block = Conv_Block(channels)
        
        self.fuse1 = nn.Sequential(nn.BatchNorm2d(channels*2),
                                   nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.GELU(),
                                   dcn2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True))
        self.fuse2 = nn.Sequential(nn.BatchNorm2d(channels*2),
                                   nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.GELU(),
                                   dcn2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True))
        self.fuse3 = nn.Sequential(nn.BatchNorm2d(channels*2),
                                   nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.GELU(),
                                   dcn2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True))
       
        self.MSA5 = MSA_module(dim = channels)
        self.MSA4 = MSA_module(dim = channels)
        self.MSA3 = MSA_module(dim = channels)
        self.MSA2 = MSA_module(dim = channels)

        self.predtrans1  = nn.Conv2d(channels, n_class, kernel_size=3, padding=1)
        self.predtrans2  = nn.Conv2d(channels, n_class, kernel_size=3, padding=1)
        self.predtrans3  = nn.Conv2d(channels, n_class, kernel_size=3, padding=1)
        self.predtrans4  = nn.Conv2d(channels, n_class, kernel_size=3, padding=1)
        self.predtrans5  = nn.Conv2d(channels, n_class, kernel_size=3, padding=1)

        #self.pred1  = nn.Conv2d(n_class, 1, kernel_size=3, padding=1)
        self.pred2  = nn.Conv2d(n_class, 1, kernel_size=3, padding=1)
        self.pred3  = nn.Conv2d(n_class, 1, kernel_size=3, padding=1)
        self.pred4  = nn.Conv2d(n_class, 1, kernel_size=3, padding=1)
        self.pred5  = nn.Conv2d(n_class, 1, kernel_size=3, padding=1)
        
        self.initialize()
        self.last_activation = nn.Sigmoid() # if using BCELoss
        


    def forward(self, E4, E3, E2, E1,shape):
        E4, E3, E2, E1 = self.side_conv1(E4), self.side_conv2(E3), self.side_conv3(E2), self.side_conv4(E1)
        
        '''
        if E4.size()[2:] != E3.size()[2:]:
            E4 = F.interpolate(E4, size=E3.size()[2:], mode='bilinear')
        if E2.size()[2:] != E3.size()[2:]:
            E2 = F.interpolate(E2, size=E3.size()[2:], mode='bilinear')
        '''

        E5 = self.conv_block(E4, E3, E2)
        
        E4 = torch.cat((E4, F.interpolate(E5, size=E4.size()[2:], mode='bilinear')), 1)
        E3 = torch.cat((E3, F.interpolate(E5, size=E3.size()[2:], mode='bilinear')), 1)
        E2 = torch.cat((E2, E5), 1)

        E4 = F.relu(self.fuse1(E4), inplace=True)
        E3 = F.relu(self.fuse2(E3), inplace=True)
        E2 = F.relu(self.fuse3(E2), inplace=True)
        
        #E4 = self.AG4(E5, E4)
        #E3 = self.AG3(E5, E3)
        #E2 = self.AG2(E5, E2)
        #E1 = self.AG1(E5, E1)
        
        P5 = self.predtrans5(E5)

        E5 = F.interpolate(E5, size=E4.size()[2:], mode='bilinear')
        P5 = F.interpolate(P5, size=E4.size()[2:], mode='bilinear')
        D4 = self.MSA5(E5, E4, self.pred5(P5))
        D4 = F.interpolate(D4, size=E3.size()[2:], mode='bilinear')
        P4  = self.predtrans4(D4)
        
        D3 = self.MSA4(D4, E3, self.pred4(P4))
        D3 = F.interpolate(D3,   size=E2.size()[2:], mode='bilinear')
        P3  = self.predtrans3(D3)  
        
        D2 = self.MSA3(D3, E2, self.pred3(P3))
        D2 = F.interpolate(D2, size=E1.size()[2:], mode='bilinear')
        P2  = self.predtrans2(D2)
        
        D1 = self.MSA2(D2, E1, self.pred2(P2))
        P1  =self.predtrans1(D1)

        P1 = F.interpolate(P1, size=shape, mode='bilinear')
        P2 = F.interpolate(P2, size=shape, mode='bilinear')
        P3 = F.interpolate(P3, size=shape, mode='bilinear')
        P4 = F.interpolate(P4, size=shape, mode='bilinear')
        P5 = F.interpolate(P5, size=shape, mode='bilinear')
        
        if self.n_classes ==1:         
            P1 = self.last_activation(P1)
            P2 = self.last_activation(P2)
            P3 = self.last_activation(P3)
            P4 = self.last_activation(P4)
            P5 = self.last_activation(P5)
                 
                 
        
        return P5, P4, P3, P2, P1

    def initialize(self):
        weight_init(self)

 
