import torch 
from torch import nn
from einops import rearrange, repeat 
from einops.layers.torch import Rearrange 
from collections import OrderedDict 
from typing import Tuple, Union , Callable, Optional
import torch.nn.functional as F 
import numpy as np 
from torch.utils.checkpoint import checkpoint 

from .timm_model import TimmModel 
from .utils import freeze_batch_norm_2d

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

def convert_weights_to_fp16(model: nn.Module): 
    """
    Convert applicable model parameters to fp16
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)): 
        l.weight.data= l.weight.data.half()
        if l.bias is not None: 
            l.bias.data= l.bias.data.half()
    
    if isinstance(l, nn.MultiheadAttention): 
        for attr in [*[f"{s}_proj_weight" for s in ["in",  "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]: 
            tensor = getattr(l, attr)
            if tensor is not None: 
                tensor.data= tensor.data.half() 
    
    for name in ["text_projection", "proj"]: 
        if hasattr(l, name): 
            attr= getattr(l, name)
            if attr is not None: 
                attr.data= attr.data.half() 
        
class QuickGELU(nn.Module): 
    def forward(self, x: torch.Tensor): 
        return x*torch.sigmoid(1.702*x)

class ResidualAttentionBlock(nn.Module): 
    def __init__(self, d_model: int, n_head: int,act_layer: Callable=nn.GELU): 
        super().__init__() 
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1= LayerNorm(d_model)
        self.mlp= nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model*4)), 
            ("gelu", QuickGELU()), 
            ("c_proj", nn.Linear(d_model*4, d_model)), 
        ]))
        self.ln_2= LayerNorm(d_model)
        #self.attn_mask= attn_mask 

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None): 
        self.attn_mask= self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask= attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None): 
        x= x+ self.attention(self.ln_1(x), attn_mask= attn_mask)
        x= x+ self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module): 
    def __init__(self, width: int, layers: int, heads: int,mlp_ratio: float=4.0,
         act_layer: Callable=nn.GELU, attn_mask: torch.Tensor=None): #attn_mask: torch.Tensor=None
        super().__init__()
        self.width= width 
        self.layers= layers
        self.grad_checkpointing=False 
        #self.resblocks= nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.resblocks= nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer) for _ in range(layers)
        ])
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]= None): 
        for r in self.resblocks: 
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x=checkpoint(r, x, attn_mask)
            else: 
                x= r(x, attn_mask=attn_mask)
        return x 

class VisionTransformer(nn.Module): 
    def __init__(self, image_size: int, patch_size: int, width: int, layers: int, heads: int, mlp_ratio: float, 
                    output_dim: int, act_layer: Callable=nn.GELU): 
        super().__init__()

        self.image_size= image_size 
        self.output_dim= output_dim 
        self.conv1= nn.Conv2d(in_channels=3, out_channels= width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale= width **-0.5 
        self.class_embedding= nn.Parameter(scale*torch.randn(width))
        self.positional_embedding = nn.Parameter(scale*torch.randn((image_size // patch_size)**2+1, width))
        self.ln_pre= LayerNorm(width)

        self.transformer= Transformer(width, layers, heads)
        self.ln_post= LayerNorm(width)
        self.proj= nn.Parameter(scale*torch.randn(width, output_dim))
    
    
    ## Editting this function to support partial layer fine-tune
    def lock(self, unlocked_groups=0, freeze_bn_stats=False): 
        assert unlocked_groups==0, 'Partial locking not currently supported for this model'
        for param in self.parameters(): 
            param.requires_grad=False 

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True): 
        self.transformer.grad_checkpointing= enable


    def forward(self, x: torch.Tensor): 
        x= self.conv1(x) # shape =[*, width, grid, grid]
        x= x.reshape(x.shape[0], x.shape[1], -1) #shape= [*, width, grid**2]
        x= x.permute(0, 2, 1)# shape = [*, grid**2, width]
        # shape = [*, grid ** 2 + 1, width]
        x= torch.cat([self.class_embedding.to(x.dtype)+ torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x= x+ self.postition_embedding.to(x.dtype)
        x= self.ln_pre(x) 

        x=x.permute(1, 0, 2) #NLD -> LND
        x= self.transformer(x)
        x= x.permute(1, 0, 2) # LND -> NLD 

        x=self.ln_post(x[:, 0, :])

        if self.proj is not None: 
            x= x@self.proj 
        return x 

@dataclass 
class ViTCfg: 
    layers: Union[Tuple[int, int, int, int], int]= 12 
    width: int=786 
    head_width: int =64
    mlp_ratio: float=4.0 
    patch_size: int =16 
    image_size: Union[Tuple[int, int], int]= 224 
    timm_model_name: str=None # a valid model name overrides layers, width, patch_size 
    timm_model_pretrained: bool= False # use (imagenet) pretrained weights for named model 
    timm_pool: str= 'avg' # feature pooling for timm model ('abs attn', 'rot_attn', 'avg', '')
    tim_proj: str= 'linear' # linear pojection for timm model output ('linear', 'mlp', '')

class VIT_Pretrained(nn.Module): 
    def __init__(
        self, embed_dim: int,
        vision_cfg: ViTCfg, 
        quick_gelu: bool = False, 
        ): 
        super().__init__() 
        if isinstance(vision_cfg, dict): 
            vision_cfg= ViTCfg(**vision_cfg)

        ## QuickGELU vs native nn.GELU --> is both faster and more memory efficient 
        ## timm model is alway using GELU 
        act_layer=QuickGELU if quick_gelu else nn.GELU 

        ## ViT configure through Timm 
        if vision_cfg.timm_model_name: 
            self.visual= TimmModel(
            vision_cfg.timm_model_name, 
            pretrained= vision_cfg.timm_model_pretrained, 
            pool= vision_cfg.timm_pool, 
            proj= vision_cfg.timm_proj, 
            embed_dim= embed_dim, 
            image_size= vision_cfg.image_size, 
            )

        vision_heads= vision_cfg.width  // vision_cfg.head_width
        
        self.visual= VisionTransformer(
            image_size= vision_cfg.image_size, 
            patch_size= vision_cfg.patch_size, 
            width= vision_cfg.width, 
            layers= vision_cfg.layers, 
            heads= vision_heads, 
            mlp_ratio= vision_cfg.mlp_ratio, 
            output_dim= embed_dim, 
            act_layer= act_layer, 
        )

        self.init_parameters() 

        def init_parameters(self): 
            if hasattr(self.visual, "init_parameters"): 
                self.visual.init_parameters() 

        def lock_image_tower(self, unlocked_groups=10, freeze_bn_stats=False): 
            # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
            self.visual.lock(unlocked_groups= unlocked_groups, freeze_bn_stats= freeze_bn_stats)

        @torch.jit.ignore
        def set_grad_checkpointing(self, enable=True): 
            self.visual.set_grad_checkpointing(enable)
            self.transformer.grad_checkpointing= enable 
        
        def encode_image(self, image):
            return self.visual(image)

        def forward(self, image): 
            image_features=self.encode_image(image)
            image_features= F.normalize(image_features, dim=-1)
            return image_features


def build_model_from_state_dict(state_dict: dict): 
    vit="visual.proj" in state_dict 

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") 
                            and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size= round((state_dict["visual.positional_embedding"].shape[0]-1)**0.5)
        image_size= vision_patch_size *grid_size 

    else: # ViT using Conv1 for patches 
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers= tuple(counts)
        vision_width= state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width= round((state_dict["visual.attnpool.positional_embedding"].shape[0]-1)**0.2)
        vision_patch_size= None 

        assert output_width**2 +1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size= output_width*32 

    embed_dim= state_dict["text_projection"].shape[1]
    context_length= state_dict["positional_embedding"].shape[0]
    vocab_size= state_dict["token_embedding.weight"].shape[0]
    transformer_width= state_dict["ln_final.weight"].shape[0]
    transformer_head= transformer_width //64 
    transformer_layers= len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    vision_conf= ViTCfg(layers= vision_layers, 
                width= vision_width, 
                patch_size= vision_patch_size, 
                image_size= image_size, 
                )
    ## Attention for OpenAI pretrained model weight using quick_gelu else: set it to False 
    model=model.lock_image_tower(unlocked_groups=10, freeze_bn_stats= False)
    model = VIT_Pretrained(embed_dim,vision_cfg=vision_conf, quick_gelu=True)

    #state diction remove some layers input
    state_dict.pop("input_resolution", None )

    convert_weights_to_fp16(model)
    model.load_state_dict(state_dict)
    return model.eval() 


## Function to testing the model implementation correct or not 
def trace_model(model, batch_size=256, device=torch.device('cpu')): 
    model.eval() 
    image_size= model.visual.image_size 
    example_images= torch.ones((batch_size,3, image_size, image_size), device= device)
    model= torch.jet.trace_module(
        model, 
        inputs=dict(
            forward=(example_images),
             
            encode_image=(example_images)
        ))
    model.visual.image_size=image_size
    return model 

    

