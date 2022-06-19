import torch 
from torch import nn
from einops import rearrange, repeat 
from einops.layers.torch import Rearrange 
from collections import OrderedDict 
from typing import Tuple, Union 
import torch.nn.functional as F 
import numpy as np 


## **************************
# some helpers function
## **************************

def pair(t): 
    return t if isinstance(t, tuple) else (t,t)

class PreNorm(nn.Module): 
    '''
    PreNorm class try to normalize the input before feeding to FN model.
    '''
    def __init__(self, dim, fn): 
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn= fn 

    def forward(self, x, **kwargs): 
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module): 
    def __init__(self, dim, hidden_dim, dropout=0.): 
        super().__init__()

        self.net= nn.Sequential(
            nn.Linear(dim, hidden_dim), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, dim), 
            nn.Dropout(dropout), 
        )

    def forward(self, x): 
        return self.net(x)

class Attention(nn.Module): 
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.): 
        super().__init__()
        inner_dim= dim_head * heads
        project_out= not(heads==1 and dim_head == dim)

        self.heads= heads 
        #if this scale implements in m_transfer technique --> change scale d
        self.scale= dim_head** -0.5

        self.attend= nn.Softmax(dim=-1)
        self.dropout= nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        self.to_out= nn.Sequential(nn.Linear(inner_dim, dim), 
                    nn.Dropout(dropout)
                    ) if project_out else nn.Indentity() 

    def forward(self, x): 
        qkv=  self.to_qkv(x).chunk(3, dim=-1).chunk(3, dim=-1)
        q, k, v= map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h= self.heads), qkv)
        
        dots= torch.matmul(q, k.transpose(-1, -2)) * self.scale 
        attn= self.attend(dots)
        attn= self.dropout(attn)

        out= torch.matmul(attn, v)
        out= rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
 
class transformer_encoder(nn.Module): 
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()

        self.layers= nn.ModuleList([])
        for _ in range(depth): 
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads= heads, dim_head=dim_head, dropout=dropout)), 
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x): 
        for attn, ff in self.layers: 
            x= attn(x)+ x
            x= ff(x)+ x
        return x

## ************************************************************
# Plugin all blocks together --> Standard VIT model
## ************************************************************

class ViT(nn.Module): 

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                pool='cls', channels=3,dim_head=64, dropout=0., emb_dropout=0.): 
        super().__init__() 
        image_height, image_width= pair(image_size)
        patch_height, patch_width= pair(patch_size)

        assert image_height % patch_height ==0 and image_width%patch_width==0, 'Image dimensions must be divisible by the patch size.'
        num_patches= (image_height// patch_height) * (image_width // patch_width)
        assert pool in {'cls', 'mean'}, 'Pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding= nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token= nn.Parameter(torch.randn(1, 1, dim))
        self.dropout= nn.Dropout(emb_dropout)
        
        self.transf_encoder= transformer_encoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool= pool 
        self.to_latent= nn.Identity()
        self.mlp_head= nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, num_classes)
        )

    def forward(self, img): 
        x = self.to_patch_embedding(img)
        b, n, _= x.shape
        
        cls_tokens= repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x+= self.pos_embedding[:, :(n+1)]
        x= self.dropout(x)

        x=self.transf_encoder(x)
        x=x.mean(dim=1) if self.pool =='mean' else x[:, 0]
        x= self.to_latent(x)
        return self.mlp_head(x)

## ***********************************************************
# Implement the standard CLIP VITs model + Pretraining Weights
#1 Pretrained Weight From OpenAI + From Open Source Community
## ***********************************************************

## Helpers functions 
# class LayerNorm(nn.LayerNorm): 
#     """Subclass torch's LayerNorm to handle fp16.. during training!"""

#     def forward(self, x: torch.Tensor): 
#         orig_type = x.dtype 
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type) 

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class QuickGELU(nn.Module): 
    def forward(self, x: torch.Tensor): 
        return x*torch.sigmoid(1.702*x)

class ResidualAttentionBlock(nn.Module): 
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None): 
        super().__init__() 
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1= LayerNorm(d_model)
        self.mlp= nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model*4)), 
            ("gelu", QuickGELU()), 
            ("c_proj", nn.Linear(d_model*4, d_model)), 
        ]))
        self.ln_2= LayerNorm(d_model)
        self.attn_mask= attn_mask 

    def attention(self, x: torch.Tensor): 
        self.attn_mask= self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask= self.attn_mask)[0]

    def forward(self, x: torch.Tensor): 
        x= x+ self.attention(self.ln_1(x))
        x= x+ self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module): 
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor=None): 
        super().__init__()
        self.width= width 
        self.layers= layers
        self.resblocks= nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
    
    def forward(self, x: torch.Tensor): 
        return self.resblocks(x)

class VisionTransformer(nn.Module): 
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int): 
        super().__init__()

        self.input_resolution= input_resolution 
        self.output_dim= output_dim 
        self.conv1= nn.Conv2d(in_channels=3, out_channels= width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale= width **-0.5 
        self.class_embedding= nn.Parameter(scale*torch.randn(width))
        self.positional_embedding = nn.Parameter(scale*torch.randn((input_resolution // patch_size)**2+1, width))
        self.ln_pre= LayerNorm(width)

        self.transformer= Transformer(width, layers, heads)
        self.ln_post= LayerNorm(width)
        self.proj= nn.Parameter(scale*torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor): 
        x= self.conv1(x) # shape =[*, width, grid, grid]
        x= x.reshape(x.shape[0], x.shape[1], -1) #shape= [*, width, grid**2]
        x= x.permute(0, 2, 1)# shape = [*, grid**2, width]
        # shape = [*, grid ** 2 + 1, width]
        x= torch.cat([self.class_embedding.to(x.dtype)+ torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype. device=x.device), x], dim=1)
        x= x+ self.postition_embedding.to(x.dtype)
        x= self.ln_pre(x) 

        x=x.permute(1, 0, 2) #NLD -> LND
        x= self.transformer(x)
        x= x.permute(1, 0, 2) # LND -> NLD 

        x=self.ln_post(x[:, 0, :])

        if self.proj is not None: 
            x= x@self.proj 
        return x 
