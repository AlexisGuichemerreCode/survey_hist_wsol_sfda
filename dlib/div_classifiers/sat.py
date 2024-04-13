import torch
import torch.nn as nn
from functools import partial
# from utils.cfgs import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
# from utils.accuracy import *
# from utils.func import *
from skimage import measure
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from dlib.configure import constants
from dlib.utils.shared import count_params

import sys
from os.path import dirname, abspath, join
root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)
IMG_NET_W_FD = join(root_dir, constants.FOLDER_PRETRAINED_IMAGENET)

__all__ = [
    'deit_sat_tiny_patch16_224', 'deit_sat_small_patch16_224', 'deit_sat_base_patch16_224',
]

def get_kernel(kernlen=3, nsig=6):    
    interval = (2*nsig+1.)/kernlen  
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)                                 
    kern1d = np.diff(st.norm.cdf(x))    
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))   
    kernel = kernel_raw/kernel_raw.sum()          
    return kernel

import torch
import torch.nn as nn


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
}

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, vis=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim   
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.loc_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.loc_embed = nn.Parameter(torch.zeros(1,  1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, vis=vis)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.loc_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.loc_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'loc_embed', 'loc_token'} 

class SAT(VisionTransformer):
    def __init__(self, encoder_name: str, encoder_weights: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1) 
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)

        kernel = get_kernel(kernlen=3,nsig=6)  
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)    
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        
        self.encoder_name: str = encoder_name
        self.task: str = constants.STD_CL
        self.classification_head = None
        
        self.name = "u-{}".format(self.encoder_name)
        self.encoder_weights = encoder_weights
        self.cams = None

        self.method = constants.METHOD_SAT
        self.arch = constants.SATCLASSIFIER
        
        if self.encoder_weights == constants.IMAGENET:
            self._init_load_pretrained_w()
        
    def _init_load_pretrained_w(self):
        if self.encoder_name == constants.DEIT_SAT_TINY_P16_224:
            # checkpoint = torch.hub.load_state_dict_from_url(
            #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            #     check_hash=True, progress=True, model_dir=IMG_NET_W_FD
            # )['model']
            # model_dict = self.state_dict()

            # for k in ['head.weight', 'head.bias', 'head_dist.weight',
            #           'head_dist.bias']:
            #     if k in checkpoint and checkpoint[k].shape != model_dict[
            #         k].shape:
            #         print(f"Removing key {k} from pretrained checkpoint")
            #         del checkpoint[k]

            # pretrained_dict = {k: v for k, v in checkpoint.items() if
            #                    k in model_dict}
            # model_dict.update(pretrained_dict)
            # self.load_state_dict(model_dict)
            checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            check_hash=True, progress=True, model_dir=IMG_NET_W_FD
            )
            if 'model' in checkpoint.keys():
                checkpoint = checkpoint['model']
            model_dict = self.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint : 
                    del checkpoint[k]
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} 
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

        if self.encoder_name == constants.DEIT_SAT_SMALL_P16_224:
            # checkpoint = torch.hub.load_state_dict_from_url(
            #     url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            #     check_hash=True, progress=True, model_dir=IMG_NET_W_FD
            # )['model']
            # model_dict = self.state_dict()
            # for k in ['head.weight', 'head.bias', 'head_dist.weight',
            #           'head_dist.bias']:
            #     if k in checkpoint and checkpoint[k].shape != model_dict[
            #         k].shape:
            #         print(f"Removing key {k} from pretrained checkpoint")
            #         del checkpoint[k]
            # pretrained_dict = {k: v for k, v in checkpoint.items() if
            #                    k in model_dict}
            # model_dict.update(pretrained_dict)
            # self.load_state_dict(model_dict)
            checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            check_hash=True, progress=True, model_dir=IMG_NET_W_FD
            )
            if 'model' in checkpoint.keys():
                checkpoint = checkpoint['model']
            model_dict = self.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint : 
                    del checkpoint[k]
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} 
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

        if self.encoder_name == constants.DEIT_SAT_BASE_P16_224:
            # checkpoint = torch.hub.load_state_dict_from_url(
            #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            #     check_hash=True, progress=True, model_dir=IMG_NET_W_FD
            # )['model']
            # model_dict = self.state_dict()
            # for k in ['head.weight', 'head.bias', 'head_dist.weight',
            #           'head_dist.bias']:
            #     if k in checkpoint and checkpoint[k].shape != model_dict[
            #         k].shape:
            #         print(f"Removing key {k} from pretrained checkpoint")
            #         del checkpoint[k]
            # pretrained_dict = {k: v for k, v in checkpoint.items() if
            #                    k in model_dict}
            # model_dict.update(pretrained_dict)
            # self.load_state_dict(model_dict)
            checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
             check_hash=True, progress=True, model_dir=IMG_NET_W_FD
            )
            if 'model' in checkpoint.keys():
                checkpoint = checkpoint['model']
            model_dict = self.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint : 
                    del checkpoint[k]
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} 
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
             
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        loc_tokens = self.loc_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x, loc_tokens), dim=1)
        pos_embed = torch.cat([self.pos_embed, self.loc_embed], 1)
        x = x + pos_embed
        x = self.pos_drop(x) 
        mask_all = []
        for cur_depth, blk in enumerate(self.blocks):
            x,  mask = blk(x, cur_depth) 
            mask_all.append(mask)

        x = self.norm(x)
        return x[:, 0], x[:, -1], x[:, 1:-1],  mask_all

    def forward(self, x, labels=None, phase='train'):
        batch = x.size(0)
        x_cls, x_loc, x_patch,  mask_all = self.forward_features(x)  
        n, p, c = x_patch.shape

        mask_all = torch.stack(mask_all)
        mask_all = mask_all[-3:,:,:,:,1:-1] 
        mask_all = torch.mean(mask_all, dim=2) 
        mask_all = torch.mean(mask_all, dim=0) 
        mask_all = mask_all.reshape(batch,1,14,14) 

        x_patch = torch.reshape(x_patch , [n, int(p**0.5), int(p**0.5), c])  
        x_patch = x_patch.permute([0, 3, 1, 2])   
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2) 
        mask_avg = mask_all.clone() 
        mask_avg = F.conv2d(mask_avg, self.weight, padding=1) 
        
        self.logits_dict = {'logits': x_logits}

        # if phase == 'train':
        self.losses_dict = {'ba_loss': mask_all.view(batch,-1).mean(-1), 'norm_loss':((1-mask_avg)*mask_avg).view(batch,-1).mean(-1)}
            # return x_logits, mask_all.view(batch,-1).mean(-1), ((1-mask_avg)*mask_avg).view(batch,-1).mean(-1)
        # else: 
            
        if labels is not None:
            n, c, h, w = x_patch.shape 
            mask_all = mask_all.reshape([n, h, w])
            self.cams = mask_all.detach()
            # return x_logits, mask_all
            
        return x_logits
    
    def flush(self):
        self.encoder_last_features = None
        self.cams = None

    def get_info_nbr_params(self) -> str:
        totaln = count_params(self)
        cl_head_n = 0
        if self.classification_head:
            cl_head_n = count_params(self.classification_head)

        info = self.__str__() + ' \n NBR-PARAMS: \n'

        info += '\tEncoder [{}]: {}. \n'.format(self.name, totaln - cl_head_n)
        if self.classification_head:
            info += '\tClassification head [{}]: {}. \n'.format(
                self.classification_head.name, cl_head_n)
        info += '\tTotal: {}. \n'.format(totaln)

        return info


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., vis=False):
        super().__init__()
        self.vis = vis
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cur_depth=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = nn.Sigmoid()(attn[:,:,-1,:].unsqueeze(2).mean(1).unsqueeze(1))

        attn = attn.softmax(dim=-1)  
        if cur_depth >= 9 :    
            attn = attn * mask
         
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, mask


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, vis=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, vis=vis)
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, cur_depth=None):
        o,  mask = self.attn(self.norm1(x), cur_depth=cur_depth)
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x,  mask

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
 

# @register_model
def deit_sat_tiny_patch16_224(encoder_weights=constants.IMAGENET,
                              pretrained=False, **kwargs):
    model = SAT(
        encoder_weights=encoder_weights,
        encoder_name=constants.DEIT_SAT_TINY_P16_224,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint : 
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

# @register_model
def deit_sat_small_patch16_224(encoder_weights=constants.IMAGENET,
                               pretrained=False, **kwargs):
    model = SAT(
        encoder_weights=encoder_weights,
        encoder_name=constants.DEIT_SAT_SMALL_P16_224,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint : 
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


# @register_model
def deit_sat_base_patch16_224(encoder_weights=constants.IMAGENET,
                              pretrained=False, **kwargs):
    model = SAT(
        encoder_weights=encoder_weights,
        encoder_name=constants.DEIT_SAT_BASE_P16_224,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint : 
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict} 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}
    frozen_param = 0
    train_param = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue   
        train_param += param.nelement()
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    total_param = frozen_param + train_param
    print("frozen_param = {:.2f}M ({:.2f}%), train_param = {:.2f}M ({:.2f}%), total param = {:.2f}M,".format(
        frozen_param/1e6, frozen_param/total_param*100, train_param/1e6, train_param/total_param*100, total_param/1e6))
    return list(parameter_group_vars.values())

def get_sat_params_for_opt(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    weight_decay = args.weight_decay
    # if weight_decay and filter_bias_and_bn:
    if filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    return parameters

def test_SAT():
    import datetime as dt

    import torch.nn.functional as F

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    txt = {0: 'torch', 1: 'wsol'}

    pretrained = False
    num_classes = 200

    for encoder in [constants.DEIT_SAT_BASE_P16_224,
                    constants.DEIT_SAT_SMALL_P16_224,
                    constants.DEIT_SAT_TINY_P16_224]:

        if encoder == constants.DEIT_SAT_TINY_P16_224:
            model = deit_sat_tiny_patch16_224(pretrained=pretrained,
                                                num_classes=num_classes)

        if encoder == constants.DEIT_SAT_SMALL_P16_224:
            model = deit_sat_small_patch16_224(pretrained=pretrained,
                                                 num_classes=num_classes)

        if encoder == constants.DEIT_SAT_BASE_P16_224:
            model = deit_sat_base_patch16_224(pretrained=pretrained,
                                                num_classes=num_classes)

        model.to(device)
        print(model.get_info_nbr_params())
        bsize = 1
        h, w = 224, 224
        x = torch.rand(bsize, 3, 224, 224).to(device)
        labels = torch.zeros((bsize,), dtype=torch.long)
        model(x)
        # print(f'logits shape : {logits.shape} x : {x.shape} '
        #       f'classes : {num_classes}')

        t0 = dt.datetime.now()
        model(x, labels=labels)
        cams = model.cams
        print(cams.shape, x.shape)
        # if cams.shape != (1, h, w):
        #     tx = dt.datetime.now()
        #     full_cam = F.interpolate(
        #         input=cams.unsqueeze(0),
        #         size=[h, w],
        #         mode='bilinear',
        #         align_corners=True)
        # print(x.shape, cams.shape)
        print('time: {}'.format(dt.datetime.now() - t0))


if __name__ == "__main__":
    test_SAT()
