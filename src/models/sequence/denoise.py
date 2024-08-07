import copy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy
from src.models.sequence.convNext import NConvNeXth
from collections import namedtuple
try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None
from flash_attn.utils.distributed import  all_gather_raw
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]

def expand_simplex(xt, alphas, prior_pseudocount):
    prior_weights = (prior_pseudocount / (alphas + prior_pseudocount - 1))[:, None, None]
    return torch.cat([xt * (1 - prior_weights), xt * prior_weights], -1), prior_weights

def sample_cond_prob_path(args, seq, alphabet_size):
    B, L = seq.shape
    seq_one_hot = torch.nn.functional.one_hot(seq, num_classes=alphabet_size)
    # if args.mode == 'dirichlet':
    alphas = torch.from_numpy(1 + scipy.stats.expon().rvs(size=B) * args.alpha_scale).to(seq.device).float()
    if args.fix_alpha:
        alphas = torch.ones(B, device=seq.device) * args.fix_alpha
    alphas_ = torch.ones(B, L, alphabet_size, device=seq.device)
    alphas_ = alphas_ + seq_one_hot * (alphas[:,None,None] - 1)
    dist = torch.distributions.Dirichlet(alphas_)
    torch.manual_seed(args.seed)
    s = torch.get_rng_state()
    torch.set_rng_state(s)
    xt = dist.sample()
    # elif args.mode == 'distill':
    #     alphas = torch.zeros(B, device=seq.device)
    #     xt = torch.distributions.Dirichlet(torch.ones(B, L, alphabet_size, device=seq.device)).sample()
    # elif args.mode == 'riemannian':
    #     t = torch.rand(B, device=seq.device)
    #     dirichlet = torch.distributions.Dirichlet(torch.ones(alphabet_size, device=seq.device))
    #     x0 = dirichlet.sample((B,L))
    #     x1 = seq_one_hot
    #     xt = t[:,None,None] * x1 + (1 - t[:,None,None]) * x0
    #     alphas = t
    # elif args.mode == 'ardm' or args.mode == 'lrar':
    #     mask_prob = torch.rand(1, device=seq.device)
    #     mask = torch.rand(seq.shape, device=seq.device) < mask_prob
    #     if args.mode == 'lrar': mask = ~(torch.arange(L, device=seq.device) < (1-mask_prob) * L)
    #     xt = torch.where(mask, alphabet_size, seq) # mask token index
    #     xt = torch.nn.functional.one_hot(xt, num_classes=alphabet_size + 1).float() # plus one to include index for mask token
    #     alphas = mask_prob.expand(B)
    return xt, alphas

# from genomic_benchmarks.models.torch import CNN
# class CNNModel(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.cnn = CNN(device="cuda",
#                         number_of_classes=2,
#                        vocab_size=5,
#                        embedding_dim=100,
#                        input_len=200,)
#     def forward(self, seq, t=None, cls = None, return_embedding=False, state=None):
#         seq = self.cnn(seq)
#         return seq, None


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class xBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.dwconv(x)
        x = x.permute(0, 2, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2,1) # (N, H, W, C) -> (N, C, H, W)

        x = self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
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
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[None, :, None] * x + self.bias[None, :, None]
            return x




class CNNModel(nn.Module):
    def __init__(self, args, alphabet_size, num_cls, classifier=False, for_representation=False, pretrain=False, dilation=2, kernel_size=9, mlp=True, out_dim=2, length=248, use_outlinear=False, forget=False, num_conv1d=5, d_inner=2, final_conv=False, use_comp=True, **kwargs):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.args = args
        self.classifier = classifier
        self.num_cls = num_cls
        self.for_representation = for_representation
        self.d_model = args.hidden_dim
        self.pretrain = pretrain
        self.mlp = mlp
        self.use_outlinear = use_outlinear
        self.forget = forget
        self.num_conv1d = num_conv1d
        self.d_inner = d_inner
        self.use_final_conv = final_conv
        self.use_comp = use_comp
        self.num_layers = self.num_conv1d * args.num_cnn_stacks
        self.num_cnn_stacks = args.num_cnn_stacks
        self.hidden_dim = int(1.42*args.hidden_dim)
        self.mode = args.mode

        if self.args.clean_data:
            self.linear = nn.Embedding(self.alphabet_size, embedding_dim=args.hidden_dim)
        else:
            expanded_simplex_input = args.cls_expanded_simplex or not classifier and (args.mode == 'dirichlet' or args.mode == 'riemannian')
            inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1)
            inp_size = self.alphabet_size

            if self.mode=="pure_gate":
                self.linear = nn.Conv1d(inp_size, self.hidden_dim, kernel_size=9, padding=4)
            else:
                self.linear = nn.Conv1d(inp_size, args.hidden_dim, kernel_size=9, padding=4)
            
            # if self.use_comp:
            if self.mode != "pure_gate":
                self.rc_linear = nn.Conv1d(inp_size, args.hidden_dim, kernel_size=9, padding=4)
            # self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim= args.hidden_dim),nn.Linear(args.hidden_dim, args.hidden_dim))
        if self.mode=="dilation":
            self.convs = [nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                        nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                        nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, dilation=dilation**1, padding=(kernel_size-1)//2*(dilation**1)),
                                        nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, dilation=dilation**2, padding=(kernel_size-1)//2*(dilation**2)),
                                        nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, dilation=dilation**3, padding=(kernel_size-1)//2*(dilation**3))][:self.num_conv1d]
            self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(args.num_cnn_stacks)])
            self.gates = [nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                        nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                        nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, dilation=dilation**1, padding=(kernel_size-1)//2*(dilation**1)),
                                        nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, dilation=dilation**2, padding=(kernel_size-1)//2*(dilation**2)),
                                        nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, dilation=dilation**3, padding=(kernel_size-1)//2*(dilation**3))][:self.num_conv1d]
            self.gates = nn.ModuleList([copy.deepcopy(layer) for layer in self.gates for i in range(args.num_cnn_stacks)])
        elif self.mode=="up_down":
            self.down_convs = [nn.Conv1d(args.hidden_dim, int(args.hidden_dim/1.8), kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
                            nn.Conv1d(int(args.hidden_dim/1.8), int(args.hidden_dim/1.4), kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
                            nn.Conv1d(int(args.hidden_dim/1.4), int(args.hidden_dim/1.2), kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
                            nn.Conv1d(int(args.hidden_dim/1.2), args.hidden_dim, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
                            nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)][:num_conv1d]
            self.down_convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.down_convs for i in range(args.num_cnn_stacks)])
            self.down_gates = [nn.Conv1d(args.hidden_dim, int(args.hidden_dim/1.8), kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
                            nn.Conv1d(int(args.hidden_dim/1.8), int(args.hidden_dim/1.4), kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
                            nn.Conv1d(int(args.hidden_dim/1.4), int(args.hidden_dim/1.2), kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
                            nn.Conv1d(int(args.hidden_dim/1.2), args.hidden_dim, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
                            nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2)][:num_conv1d]
            self.down_gates = nn.ModuleList([copy.deepcopy(layer) for layer in self.down_gates for i in range(args.num_cnn_stacks)])

            # Upsampling layers
            self.up_convs = [nn.ConvTranspose1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1),
                            nn.ConvTranspose1d(args.hidden_dim, int(args.hidden_dim/1.2), kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1),
                            nn.ConvTranspose1d(int(args.hidden_dim/1.2), int(args.hidden_dim/1.4), kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1),
                            nn.ConvTranspose1d(int(args.hidden_dim/1.4), int(args.hidden_dim/1.8), kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1),
                            nn.ConvTranspose1d(int(args.hidden_dim/1.8), args.hidden_dim, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1)][:num_conv1d]
            self.up_convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.up_convs for i in range(args.num_cnn_stacks)])
            self.up_gates = [nn.ConvTranspose1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1),
                            nn.ConvTranspose1d(args.hidden_dim, int(args.hidden_dim/1.2), kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1),
                            nn.ConvTranspose1d(int(args.hidden_dim/1.2), int(args.hidden_dim/1.4), kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1),
                            nn.ConvTranspose1d(int(args.hidden_dim/1.4), int(args.hidden_dim/1.8), kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1),
                            nn.ConvTranspose1d(int(args.hidden_dim/1.8), args.hidden_dim, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2, output_padding=1)][:num_conv1d]
            self.up_gates = nn.ModuleList([copy.deepcopy(layer) for layer in self.up_gates for i in range(args.num_cnn_stacks)])
        elif self.mode=="convnext":
            dims=[64, 72, 108, 132]
            depths=[1, 1, 4, 2]
            drop_path_rate = 0
            layer_scale_init_value=1e-6
            stride = 1
            dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
            self.convs = [ nn.Sequential(
                                    nn.Conv1d(self.d_model, dims[0], kernel_size=2, stride=stride),
                                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                            ),
                            nn.Sequential(
                                *[xBlock(dim=dims[0], drop_path=dp_rates[0 + j], 
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[0])]
                            ),
                            nn.Sequential(
                                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                                    nn.Conv1d(dims[0], dims[1], kernel_size=2, stride=stride),
                            ),
                            nn.Sequential(
                                *[xBlock(dim=dims[1], drop_path=dp_rates[depths[0] + j], 
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[1])]
                            ),
                            nn.Sequential(
                                    LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
                                    nn.Conv1d(dims[1], dims[2], kernel_size=2, stride=stride),
                            ),
                            nn.Sequential(
                                *[xBlock(dim=dims[2], drop_path=dp_rates[depths[0]+depths[1] + j], 
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[2])]
                            ),
                            nn.Sequential(
                                    LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
                                    nn.Conv1d(dims[2], dims[3], kernel_size=2, stride=stride),
                            ),
                            nn.Sequential(
                                *[xBlock(dim=dims[3], drop_path=dp_rates[depths[0]+depths[1]+depths[2] + j], 
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[3])]
                            ),
                            ]
            self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(args.num_cnn_stacks)])
            self.gates = [      nn.Sequential(
                                    nn.Conv1d(self.d_model, dims[0], kernel_size=2, stride=stride),
                                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                            ),
                            nn.Sequential(
                                *[xBlock(dim=dims[0], drop_path=dp_rates[0 + j], 
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[0])]
                            ),
                            nn.Sequential(
                                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                                    nn.Conv1d(dims[0], dims[1], kernel_size=2, stride=stride),
                            ),
                            nn.Sequential(
                                *[xBlock(dim=dims[1], drop_path=dp_rates[depths[0] + j], 
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[1])]
                            ),
                            nn.Sequential(
                                    LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
                                    nn.Conv1d(dims[1], dims[2], kernel_size=2, stride=stride),
                            ),
                            nn.Sequential(
                                *[xBlock(dim=dims[2], drop_path=dp_rates[depths[0]+depths[1] + j], 
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[2])]
                            ),
                            nn.Sequential(
                                    LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
                                    nn.Conv1d(dims[2], dims[3], kernel_size=2, stride=stride),
                            ),
                            nn.Sequential(
                                *[xBlock(dim=dims[3], drop_path=dp_rates[depths[0]+depths[1]+depths[2] + j], 
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[3])]
                            ),  
                            ]
            self.gates = nn.ModuleList([copy.deepcopy(layer) for layer in self.gates for i in range(args.num_cnn_stacks)])
            self.ll = nn.Linear(dims[-1], self.d_model)
        if self.mode == "pure_gate":
            self.convs = [nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                        nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                        nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, dilation=dilation**1, padding=(kernel_size-1)//2*(dilation**1)),
                                        nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, dilation=dilation**2, padding=(kernel_size-1)//2*(dilation**2)),
                                        nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, dilation=dilation**3, padding=(kernel_size-1)//2*(dilation**3))][:self.num_conv1d]
            self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(args.num_cnn_stacks)])
            self.ll = nn.Linear(self.hidden_dim, args.hidden_dim)

        if self.mlp:
            self.milinear = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim*self.d_inner),
                                          nn.GELU(),
                                          nn.Linear(args.hidden_dim*self.d_inner, args.hidden_dim*self.d_inner),
                                          nn.LayerNorm(args.hidden_dim*self.d_inner),
                                          nn.Linear(args.hidden_dim*self.d_inner, args.hidden_dim*self.d_inner),
                                          nn.GELU(),
                                          nn.Linear(args.hidden_dim*self.d_inner, args.hidden_dim),
                                          nn.LayerNorm(args.hidden_dim))
        if self.mode=="dilation":
            self.norms = nn.ModuleList([nn.LayerNorm(args.hidden_dim) for _ in range(self.num_layers)])
            # self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
            self.rc_norms = nn.ModuleList([nn.LayerNorm(args.hidden_dim) for _ in range(self.num_layers)])
        elif self.mode=="up_down":
            self.norms = nn.ModuleList([nn.LayerNorm(args.hidden_dim), nn.LayerNorm(int(args.hidden_dim/1.8)), nn.LayerNorm(int(args.hidden_dim/1.4)), nn.LayerNorm(int(args.hidden_dim/1.2)), nn.LayerNorm(int(args.hidden_dim))])
            tmp = self.norms
            self.norms.append(nn.LayerNorm(args.hidden_dim))
            self.norms.extend(reversed(tmp[:-1]))
            self.rc_norms = nn.ModuleList([nn.LayerNorm(args.hidden_dim), nn.LayerNorm(int(args.hidden_dim/1.8)), nn.LayerNorm(int(args.hidden_dim/1.4)), nn.LayerNorm(int(args.hidden_dim/1.2)), nn.LayerNorm(int(args.hidden_dim))])
            tmp = self.rc_norms
            self.rc_norms.append(nn.LayerNorm(args.hidden_dim))
            self.rc_norms.extend(reversed(tmp[:-1]))
        elif self.mode == "convnext":
            pass
        elif self.mode == "pure_gate":
            self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        if self.use_final_conv:
            self.final_conv = nn.Sequential(nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=1),
                                    nn.GELU(),
                                    nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=1))
        if pretrain:
            self.out_linear = nn.Linear(args.hidden_dim, self.alphabet_size)
        elif self.use_outlinear:
            self.out = nn.Linear(args.hidden_dim*length, out_dim)
        self.dropout = nn.Dropout(args.dropout)
        if classifier:
            self.cls_head = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(args.hidden_dim, self.num_cls))

        if self.args.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=args.hidden_dim)
            self.cls_layers = nn.ModuleList([Dense(args.hidden_dim, args.hidden_dim) for _ in range(self.num_layers)])
    def forward(self, seq, t=None, cls = None, return_embedding=False, state=None):
        # if t is None and self.for_representation:
        #     # t = torch.tensor([self.args.alpha_max])[None].expand(seq.shape[0]).to(seq.device)
        #     seq, alphas = sample_cond_prob_path(self.args, seq, self.alphabet_size)
        #     seq, prior_weights = expand_simplex(seq,alphas, self.args.prior_pseudocount)
        #     t = alphas
        if not self.pretrain:
            if self.use_comp:
                # ACGTN - 01234
                N = seq==4
                rc_seq = 3-seq
                rc_seq[N] = 4
            else:
                rc_seq = seq
            seq = torch.nn.functional.one_hot(seq, num_classes=self.alphabet_size).type(torch.float32)
            if self.mode!="pure_gate":
                rc_seq = torch.nn.functional.one_hot(rc_seq, num_classes=self.alphabet_size).type(torch.float32)
            
            # time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.gelu(self.linear(feat))
            if self.mode != "pure_gate":
                rc_feat = rc_seq.permute(0,2,1)
                rc_feat = F.gelu(self.rc_linear(rc_feat))

            # if self.args.cls_free_guidance and not self.classifier:
            #     cls_emb = self.cls_embedder(cls)
            if self.mode=="dilation":
                for i in range(self.num_layers):
                    h = self.dropout(feat.clone())
                    rc_h = self.dropout(rc_feat.clone())
                    h = self.norms[i]((h).permute(0, 2, 1))
                    rc_h = self.rc_norms[i]((rc_h).permute(0,2,1))
                    if self.forget:
                        g = F.sigmoid(self.gates[i](rc_h.permute(0, 2, 1)))
                    else:
                        g = F.gelu(self.gates[i](rc_h.permute(0, 2, 1)))
                    h = F.gelu(self.convs[i](h.permute(0, 2, 1)))
                    if h.shape == feat.shape:
                        if self.forget:
                            feat = h*g + feat
                        else:
                            feat = h + g + feat
                        rc_feat = g + rc_feat
                    else:
                        feat = h
            elif self.mode == "convnext":
                for i in range(self.num_layers):
                    h = feat.clone()
                    rc_h = rc_feat.clone()
                    feat = self.convs[2*i](h)
                    rc_feat = self.gates[2*i](rc_h)
                    h = F.gelu(self.convs[2*i+1](feat))
                    if self.forget:
                        g = F.sigmoid(self.gates[2*i+1](rc_feat))
                    else:
                        g = F.gelu(self.gates[2*i+1](rc_feat))
                    if h.shape == feat.shape:
                        if self.forget:
                            feat = h*g + feat
                        else:
                            feat = h + g + feat
                        rc_feat = g + rc_feat
                    else:
                        feat = h
                feat = self.ll(feat.permute(0,2,1)).permute(0,2,1)
            elif self.mode=="pure_gate":
                for i in range(self.num_layers):
                    h = self.dropout(feat.clone())
                    h = self.norms[i]((h).permute(0, 2, 1))
                    h = self.convs[i](h.permute(0, 2, 1))
                    if self.forget:
                        g = F.sigmoid(h)
                    else:
                        g = F.gelu(h)
                    h = F.gelu(h)
                    if h.shape == feat.shape:
                        if self.forget:
                            feat = h*g + feat
                        else:
                            feat = h + g + feat
                    else:
                        feat = h
                feat = self.ll(feat.permute(0,2,1)).permute(0,2,1)
            if self.mlp:
                feat = self.milinear(feat.permute(0,2,1)).permute(0,2,1)+feat
            
            if self.use_final_conv:
                feat = self.final_conv(feat)
            feat = feat.permute(0, 2, 1)
            if self.for_representation:
                if self.use_outlinear:
                    return self.out(feat.reshape(feat.shape[0], -1)), None
                else:
                    return feat, None
            if self.classifier:
                feat = feat.mean(dim=1)
                if return_embedding:
                    embedding = self.cls_head[:1](feat)
                    return self.cls_head[1:](embedding), embedding
                else:
                    return self.cls_head(feat)
            else:
                feat = self.out_linear(feat)
            return feat
        else:
            mask = seq[1]
            seq = seq[0]
            if self.use_comp:
                N = seq==4
                rc_seq = 3-seq
                rc_seq[N] = 4
            else:
                rc_seq = seq
                
            inference_params = None
            ColumnParallelLinear = None
            if self.mode!="pure_gate":
                rc_seq = torch.nn.functional.one_hot(rc_seq, num_classes=self.alphabet_size).type(torch.float32)
            seq = torch.nn.functional.one_hot(seq, num_classes=self.alphabet_size).type(torch.float32)
            # time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))
            if self.mode!="pure_gate":
                rc_feat = rc_seq.permute(0,2,1)
                rc_feat = F.gelu(self.rc_linear(rc_feat))


            if self.mode=="dilation":
                for i in range(self.num_layers):
                    h = self.dropout(feat.clone())
                    rc_h = self.dropout(rc_feat.clone())
                    h = self.norms[i]((h).permute(0, 2, 1))
                    rc_h = self.rc_norms[i]((rc_h).permute(0,2,1))
                    if self.forget:
                        g = F.sigmoid(self.gates[i](rc_h.permute(0, 2, 1)))
                    else:
                        g = F.gelu(self.gates[i](rc_h.permute(0, 2, 1)))
                    h = F.gelu(self.convs[i](h.permute(0, 2, 1)))
                    if h.shape == feat.shape:
                        if self.forget:
                            feat = h*g + feat
                        else:
                            feat = h + g + feat
                        rc_feat = g + rc_feat
                    else:
                        feat = h
            elif self.mode == "up_down":
                h_list = []
                g_list = []
                for i in range(self.num_layers):
                    h = self.dropout(feat.clone())
                    rc_h = self.dropout(rc_feat.clone())
                    h_list.append(h)
                    g_list.append(rc_h)
                    h = self.norms[i]((h).permute(0, 2, 1))
                    rc_h = self.rc_norms[i]((rc_h).permute(0,2,1))
                    if self.forget:
                        g = F.sigmoid(self.down_gates[i](rc_h.permute(0, 2, 1)))
                    else:
                        g = F.gelu(self.down_gates[i](rc_h.permute(0, 2, 1)))
                    h = F.gelu(self.down_convs[i](h.permute(0, 2, 1)))
                    if self.forget:
                        feat = h*g
                    else:
                        feat = h + g
                    rc_feat = g
                for i in range(self.num_layers):
                    h = self.dropout(feat.clone())
                    rc_h = self.dropout(rc_feat.clone())
                    h = self.norms[i+self.num_layers]((h).permute(0, 2, 1))
                    rc_h = self.rc_norms[i+self.num_layers]((rc_h).permute(0,2,1))
                    if self.forget:
                        g = F.sigmoid(self.up_gates[i](rc_h.permute(0, 2, 1)))
                    else:
                        g = F.gelu(self.up_gates[i](rc_h.permute(0, 2, 1)))
                    h = F.gelu(self.up_convs[i](h.permute(0, 2, 1)))
                    if self.forget:
                        feat = h*g + h_list[-i-1]
                    else:
                        feat = h + g
                    rc_feat = g + g_list[-i-1]
            elif self.mode == "convnext":
                for i in range(self.num_layers):
                    h = feat.clone()
                    rc_h = rc_feat.clone()
                    feat = self.convs[2*i](h)
                    rc_feat = self.gates[2*i](rc_h)
                    h = F.gelu(self.convs[2*i+1](feat))
                    if self.forget:
                        g = F.sigmoid(self.gates[2*i+1](rc_feat))
                    else:
                        g = F.gelu(self.gates[2*i+1](rc_feat))
                    if h.shape == feat.shape:
                        if self.forget:
                            feat = h*g + feat
                        else:
                            feat = h + g + feat
                        rc_feat = g + rc_feat
                    else:
                        feat = h
                feat = self.ll(feat.permute(0,2,1)).permute(0,2,1)
            elif self.mode=="pure_gate":
                for i in range(self.num_layers):
                    h = self.dropout(feat.clone())
                    h = self.norms[i]((h).permute(0, 2, 1))
                    h = self.convs[i](h.permute(0, 2, 1))
                    if self.forget:
                        g = F.sigmoid(h)
                    else:
                        g = F.gelu(h)
                    h = F.gelu(h)
                    if h.shape == feat.shape:
                        if self.forget:
                            feat = h*g + feat
                        else:
                            feat = h + g + feat
                    else:
                        feat = h
                feat = self.ll(feat.permute(0,2,1)).permute(0,2,1)

            if self.mlp:
                feat = self.milinear(feat.permute(0,2,1)).permute(0,2,1)+feat

            if self.use_final_conv: 
                feat = self.final_conv(feat)
            feat = feat.permute(0, 2, 1)
            lm_logits = self.out_linear(feat)
            # During inference, we want the full logit for sampling
            if ColumnParallelLinear is not None and inference_params is not None:
                if isinstance(self.out_linear, ColumnParallelLinear):
                    lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
                    lm_logits = rearrange(
                        lm_logits, "(n b) s d -> b s (n d)", b=feat.shape[0]
                    )
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            return CausalLMOutput(logits=(lm_logits,mask)), None
    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model


class TransformerModel(nn.Module):
    def __init__(self, args, alphabet_size, num_cls, classifier=False, for_representation=False, **kwargs):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.classifier = classifier
        self.num_cls = num_cls
        self.args = args
        self.for_representation = for_representation
        self.d_model = args.hidden_dim
        if self.args.clean_data:
            self.embedder = nn.Embedding(self.alphabet_size, args.hidden_dim)
        else:
            expanded_simplex_input = args.cls_expanded_simplex or not classifier and (args.mode == 'dirichlet' or args.mode == 'riemannian')
            self.embedder = nn.Linear((2 if expanded_simplex_input  else 1) *  self.alphabet_size,  args.hidden_dim)
            self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim=args.hidden_dim), nn.Linear(args.hidden_dim, args.hidden_dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.hidden_dim, nhead=4, dim_feedforward=args.hidden_dim, dropout=args.dropout), num_layers=args.num_layers, norm=nn.LayerNorm(args.hidden_dim))

        if self.classifier:
            self.cls_head = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(args.hidden_dim, self.num_cls))
        else:
            self.out = nn.Linear(args.hidden_dim, self.alphabet_size)

        if self.args.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=args.hidden_dim)

    def forward(self, seq, t=None, cls = None, return_embedding=False, state=None):
        if t is None and self.for_representation:
            # t = torch.tensor([self.args.alpha_max])[None].expand(seq.shape[0]).to(seq.device)
            seq, alphas = sample_cond_prob_path(self.args, seq, self.alphabet_size)
            seq, prior_weights = expand_simplex(seq,alphas, self.args.prior_pseudocount)
            t = alphas
        feat = self.embedder(seq)
        if not self.args.clean_data:
            time_embed = F.relu(self.time_embedder(t))
            feat = feat + time_embed[:,None,:]
        if self.args.cls_free_guidance and not self.classifier:
            feat = feat + self.cls_embedder(cls)[:,None,:]
        feat = self.transformer(feat)
        if self.for_representation:
            return feat, None
        if self.classifier:
            feat = feat.mean(dim=1)
            return self.cls_head(feat)
        else:
            return self.out(feat)
    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model
    









import torch
from einops import rearrange, repeat
import math
try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None

try:
    from fftconv import fftconv_ref, fftconv_func, fftconv_heads_ref
    # from src.ops.fftconv import fftconv_ref, fftconv_func

except ImportError:
    fftconv_func = None

from flash_attn.modules.mlp import Mlp, FusedMLP
from flash_attn.modules.block import Block
from functools import partial
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm, dropout_add_rms_norm = None, None


def Activation(activation=None, size=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.Softplus()
    # elif activation in ['sqrelu', 'relu2']:
    #     return SquaredReLU()
    # elif activation == 'laplace':
    #     return Laplace()
    # elif activation == 'ln':
    #     return TransposedLN(dim)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

class FFTConvFuncv2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, k):
        seqlen = u.shape[-1]
        if len(u.shape) > 3:
            k = k.unsqueeze(1)
        fft_size = 2 * seqlen

        k_f = torch.fft.rfft(k, n=fft_size) / fft_size
        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
        y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
        ctx.save_for_backward(u_f, k_f)
        return y

    @staticmethod
    def backward(ctx, dout):
        u_f, k_f = ctx.saved_tensors
        seqlen = dout.shape[-1]
        fft_size = 2 * seqlen

        dout_f = torch.fft.rfft(dout, n=fft_size)
        du = torch.fft.irfft(dout_f * k_f.conj(), n=fft_size, norm="forward")[
            ..., :seqlen
        ]
        dk = torch.fft.irfft(dout_f * u_f.conj(), n=fft_size, norm="forward")[
            ..., :seqlen
        ]
        return du, dk.squeeze()


def fftconv_ref(u, k, D, dropout_mask, gelu=True, k_rev=None, bidirectional=False):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()

    if bidirectional:
        # we need to pad 1/2 before and 1/2 after the input
        padded_length = seqlen + 2 * (seqlen // 2)
        pad_before = padded_length // 2 - (seqlen // 2)
        pad_after = padded_length - seqlen - pad_before
        padded_u = F.pad(u, (pad_before, pad_after), mode='constant', value=0)
        u_f = torch.fft.rfft(padded_u.to(dtype=k.dtype), n=fft_size)
    else:
        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3 and len(k_f.shape) != 2: # use input as filter
        k_f = k_f.reshape(u_f.shape).contiguous()
    elif len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
        # k_f = k_f.repeat(u_f.shape[0], 1, 1).reshape(u_f.shape).contiguous()

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = (
            nn.Parameter(w * torch.ones(1, dim))
            if train_freq
            else w * torch.ones(1, dim)
        )

    def forward(self, x):
        return torch.sin(self.freq * x)

class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)

class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        shift: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        decay = torch.exp(-t * self.deltas.abs())
        x = x * (decay + self.shift)
        return x

def auto_assign_attrs(cls, **kwargs):
    for k, v in kwargs.items():
        setattr(cls, k, v)

class HyenaFilter(OptimModule):
    def __init__(
        self,
        d_model,
        emb_dim=3,  # dim of input to MLP, augments with positional encoding
        order=16,  # width of the implicit MLP
        fused_fft_conv=False,
        seq_len=1024,
        lr=1e-3,
        lr_pos_emb=1e-5,
        dropout=0.0,
        w=1,  # frequency of periodic activations
        wd=0,  # weight decay of kernel parameters
        bias=True,
        num_inner_mlps=2,
        linear_mixer=False,
        modulate: bool = True,
        normalized=False,
        bidirectional=False,
        **kwargs,
    ):
        """
        Implicit long filter with modulation.
        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        Note:
            filter_dropout is not implemented
        """
        super().__init__()
        auto_assign_attrs(
            self, d_model=d_model, emb_dim=emb_dim, seq_len=seq_len, modulate=modulate
        )
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional

        act = Sin(dim=order, w=w)
        assert (
            emb_dim % 2 != 0 and emb_dim >= 3
        ), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        # uses a variable number of inner linear layers
        if linear_mixer is False:
            self.implicit_filter = nn.Sequential(
                nn.Linear(emb_dim, order),
                act,
            )
            for i in range(num_inner_mlps):
                self.implicit_filter.append(nn.Linear(order, order))
                self.implicit_filter.append(act)
            # final linear layer
            self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
        else:
            self.implicit_filter = nn.Sequential(
                nn.Linear(emb_dim, d_model, bias=False),
            )

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        if self.modulate:
            h = self.modulation(t, h)

        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)

        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None:
            k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k
        if bias is None:
            bias = self.bias
        bias = bias if self.use_bias else 0 * bias

        if self.fused_fft_conv:
            bias = bias.to(dtype=torch.float32)
            y = fftconv_func(
                x,
                k,
                bias,
                dropout_mask=None,
                gelu=False,
                force_fp16_output=torch.is_autocast_enabled(),
            )
        else:
            y = fftconv_ref(x, k, bias, dropout_mask=None, gelu=False, bidirectional=self.bidirectional)
            # y = (
            #     FFTConvFuncv2.apply(x, k.to(dtype=torch.float32))
            #     + bias.unsqueeze(-1) * x
            # )

        return y.to(dtype=x.dtype)


class HyenaOperator(nn.Module):
    def __init__(
        self,
        d_model,
        l_max,
        order=2,
        filter_order=64,
        num_heads=1,
        inner_factor=1,
        num_blocks=1,
        fused_bias_fc=False,
        outer_mixing=False,
        dropout=0.0,
        filter_dropout=0.0,
        filter_cls="hyena-filter",
        post_order_ffn=False,
        jit_filter=False,
        short_filter_order=3,
        activation="id",
        return_state=False,
        bidirectional=False,
        **filter_args,
    ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf
        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            filter_order: (int): Width of the FFN parametrizing the implicit filter. Defaults to 64
            num_heads: (int): Number of heads. Defaults to 1
            inner_factor: (int): Width multiplier. Defaults to 1
            num_blocks: (int): Number of blocks in sequence length. Defaults to 1
            fused_bias_fc: (bool): Whether to use fused bias FC. Defaults to False
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
            post_order_ffn: (bool): Apply a dense layer between steps of the recurrence. Defaults to False
            jit_filter: (bool): Whether JIT the implicit filter function. Defaults to False
            short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3
            activation: (str): type of act between kernel output and FF (default identity)
            return_state: (bool): whether to return a state
        """
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"Model dimension {d_model} must be divisible by num heads {num_heads}"
        assert (
            l_max % num_blocks == 0
        ), f"Maximum signal length {l_max} must be divisible by block dimension {num_blocks}"
        block_dim = l_max // num_blocks
        head_dim = d_model // num_heads

        auto_assign_attrs(
            self,
            d_model=d_model,
            order=order,
            l_max=l_max,
            num_heads=num_heads,
            inner_factor=inner_factor,
            block_dim=block_dim,
            head_dim=head_dim,
            filter_order=filter_order,
            post_order_ffn=post_order_ffn,
            short_filter_order=short_filter_order,
            num_blocks=num_blocks,
            filter_dropout=filter_dropout,
            jit_filter=jit_filter,
            outer_mixing=outer_mixing,
            activation=activation,
            return_state=return_state,
            bidirectional = bidirectional
        )
        self.activation = Activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.setup_projections(fused_bias_fc, inner_factor)
        self.setup_filters(filter_cls, filter_args)
        # self.modulate = ExponentialModulation( self.head_dim * self.inner_factor * (self.order - 1), kwargs=filter_args)

    def setup_projections(self, fused_bias_fc, inner_factor):
        "Initializes input and output projections (over the width dimension)"
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.out_proj = linear_cls(self.d_model * inner_factor, self.d_model)
        self.in_proj = linear_cls(self.d_model, (self.order + 1) * self.d_model)
        if self.post_order_ffn:
            self.ord_proj_w = nn.Parameter(
                torch.randn(self.order, self.num_heads, self.num_heads)
                / math.sqrt(self.head_dim)
            )

    def setup_filters(self, filter_cls, filter_args):
        "Initializes the explicit and implicit filters"
        assert self.order >= 2, f"Order must be at least 2, (got {self.order})"
        total_width = self.d_model * self.inner_factor * (self.order + 1)

        self.short_filter = nn.Conv1d(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=self.short_filter_order,
            groups=total_width,
            padding=self.short_filter_order - 1,
        )

        # filter_cls = instantiate(registry.layer, filter_cls, partial=True)

        self.filter_fn = HyenaFilter(
            self.head_dim * self.inner_factor * (self.order - 1),
            order=self.filter_order,
            emb_dim=5,
            seq_len=self.l_max,
            channels=1,
            lr_pos_emb=0,
            w=10,
            dropout=self.filter_dropout,
            bidirectional=self.bidirectional,
            **filter_args,
        )
        if self.jit_filter:
            self.filter_fn = torch.jit.script(self.filter_fn, self.L)
        # self.input_filter = input_filter(self.head_dim * self.inner_factor * (self.order - 1), "SiLU")
        # self.q_c = nn.Linear(512, 128, dtype=torch.complex64)
        # self.k_c = nn.Linear(512, 128, dtype=torch.complex64)

    def recurrence(self, u, state):
        "Fast inference mode via distilled recurrence"
        raise NotImplementedError("Working on it!")

    def forward(self, u, *args, **kwargs):

        l = u.size(-2)
        # B, D, L = u.shape[0], u.shape[2], u.shape[1]
        l_filter = min(l, self.l_max)
        # self.t = torch.linspace(0, 1, l_filter)[None, :, None].to('cuda')
        u = self.in_proj(u)
        u = rearrange(u, "b l d -> b d l")

        uc = self.short_filter(u)[..., :l_filter]
        # k_ = self.input_filter(uc)

        uc = rearrange(
            uc,
            "b (ho v) (z l) -> b ho v z l",
            z=self.num_blocks,
            ho=self.num_heads,
            v=self.head_dim * (self.order + 1),
        )

        *x, v = uc.split(self.d_model, dim=2)
        k = self.filter_fn.filter(l_filter)
        # k = self.modulate(self.t, k_+k)

        # `c` is always 1 by default
        k = rearrange(k, "c l (v o) -> c o v l", v=self.head_dim, o=self.order - 1)[0]
        # k = rearrange(k, "c l (v o) -> c o v l", v=self.head_dim, o=self.order - 1)

        bias = rearrange(
            self.filter_fn.bias, "(v o) -> o v", v=self.head_dim, o=self.order - 1
        )

        # freq = v.reshape(B, D, L)
        # freq = torch.fft.rfft(freq.to(dtype=k.dtype), n=2*l_filter)
        # q_c = self.q_c(freq)
        # k_c = self.k_c(freq)
        # att_c = torch.einsum("bdl, bDl ->bdD", q_c, k_c)/128
        # att_c = F.softmax(att_c, dim=-1)
        

        for o, x_i in enumerate(reversed(x[1:])):
            if self.outer_mixing:
                v = rearrange(v, "b h v z l -> b h 1 v z l")
                v = self.dropout(v * rearrange(x_i, "b h v z l -> b h v 1 z l"))
                v = v.sum(dim=2)
            else:
                v = self.dropout(v * x_i)

            # the bias term is broadcasted. Last dimension (l) is handled by fftconv
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o, None, :, None])
            # v = self.filter_fn(v, l_filter, k=k[:, o], bias=bias[o, None, :, None])

            if self.post_order_ffn:
                w = self.ord_proj_w[o]
                v = mul_sum(
                    rearrange(w, "h1 h2 -> 1 h1 h2 1 1 1"),
                    rearrange(v, "b h v z l -> b h 1 v z l"),
                )
        # v = torch.fft.rfft(v.to(dtype=k.dtype), n=2*l_filter)
        # v = v@att_c
        # v = torch.fft.irfft(v, n=2*l_filter, norm="forward")[..., :l]
        y = self.activation(
            rearrange(
                v * x[0],
                "b h v z l -> b (z l) (h v)",
                z=self.num_blocks,
                h=self.num_heads,
            )
        )
        y = self.out_proj(y)

        if self.return_state:
            return y, None
        return y

    @property
    def d_output(self):
        return self.d_model

def create_mlp_cls(
    d_model,
    d_inner=None,
    process_group=None,
    fused_mlp=False,
    sequence_parallel=True,
    identity_mlp=False,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    inner_dim = d_inner if d_inner is not None else 4 * d_model
    if process_group is not None:
        assert fused_mlp, "Tensor Parallel is only implemented for FusedMLP"

    if not fused_mlp and not identity_mlp:
        mlp_cls = partial(
            Mlp,
            hidden_features=inner_dim,
            activation=partial(F.gelu, approximate="tanh"),
            **factory_kwargs,
        )
    elif fused_mlp:
        mlp_cls = FusedMLP if process_group is None else ParallelFusedMLP
        parallel_kwargs = (
            {"process_group": process_group, "sequence_parallel": sequence_parallel}
            if process_group is not None
            else {}
        )
        mlp_cls = partial(
            mlp_cls, hidden_features=inner_dim, **parallel_kwargs, **factory_kwargs
        )
    else:
        mlp_cls = nn.Identity
    return mlp_cls


def create_block(
    d_model,
    args,
    d_inner=None,
    process_group=None,
    layer=None,
    attn_layer_idx=None,
    attn_cfg=None,
    layer_norm_epsilon=1e-5,
    resid_dropout1=0.1,
    resid_dropout2=0.0,
    residual_in_fp32=True,
    fused_mlp=False,
    identity_mlp=False,
    fused_dropout_add_ln=True,
    layer_idx=None,
    sequence_parallel=True,
    checkpoint_mlp=False,
    checkpoint_mixer=False,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(HyenaOperator,
        l_max=args.max_length, 
        order=args.order,
        filter_order=args.filter_order,
        num_heads=args.num_heads,
        inner_factor=args.inner_factor,
        num_blocks=args.num_blocks,
        fused_bias_fc=args.fused_bias_fc,
        outer_mixing=args.outer_mixing,
        dropout=args.dropout,
        filter_dropout=args.filter_dropout,
        filter_cls=args.filter_cls,
        post_order_ffn=args.post_order_ffn,
        jit_filter=args.jit_filter,
        short_filter_order=args.short_filter_order,
        activation=args.activation,
        return_state=args.return_state,
        bidirectional=args.bidirectional,
        **factory_kwargs,
    )
    mlp_cls = create_mlp_cls(
        d_model,
        d_inner=d_inner,
        process_group=process_group,
        fused_mlp=fused_mlp,
        identity_mlp=identity_mlp,
        sequence_parallel=sequence_parallel,
        **factory_kwargs,
    )
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs)
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=True,
        resid_dropout1=resid_dropout1,
        resid_dropout2=resid_dropout2,
        fused_dropout_add_ln=fused_dropout_add_ln,
        residual_in_fp32=residual_in_fp32,
        sequence_parallel=sequence_parallel and process_group is not None,
        mark_shared_params=process_group is not None,
    )

    block.layer_idx = layer_idx

    return block

def set_args(args, **params):
    for key, value in params.items():
        setattr(args, key, value)

from flash_attn.modules.embedding import GPT2Embeddings

class Hyena(nn.Module):
    def __init__(
        self,
        d_model,
        args,
        order=2,
        filter_order=64,
        num_heads=1,
        inner_factor=1,
        num_blocks=1,
        fused_bias_fc=False,
        outer_mixing=False,
        dropout=0.0,
        filter_dropout=0.0,
        filter_cls="hyena-filter",
        post_order_ffn=False,
        jit_filter=False,
        short_filter_order=3,
        activation="id",
        return_state=False,
        alphabet_size=5,
        num_cls=None,
        bidirectional=False,
        cls_expanded_simplex = False,
        classifier = False,
        mode = "dirichlet",
        num_layers = 2,
        for_representation = False,
        layer_norm_epsilon: float = 1e-5,
        **filter_args,
    ):
        super().__init__()
        set_args(
                    args, order=order ,filter_order=filter_order,num_heads=num_heads,inner_factor=inner_factor,num_blocks=num_blocks,
                    fused_bias_fc=fused_bias_fc,outer_mixing=outer_mixing,dropout=dropout,filter_dropout=filter_dropout,filter_cls=filter_cls,
                    post_order_ffn=post_order_ffn,jit_filter=jit_filter,short_filter_order=short_filter_order,activation=activation,return_state=return_state,
                    bidirectional=bidirectional
                )
        self.for_representation = for_representation
        self.alphabet_size = alphabet_size
        self.classifier = classifier
        self.num_cls = num_cls
        self.d_model = d_model
        self.args = args
        expanded_simplex_input = cls_expanded_simplex or not classifier and (mode == 'dirichlet' or mode == 'riemannian')
        expanded_simplex_input = False
        inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1)
        self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim=d_model), nn.Linear(d_model, d_model))
        # self.linear = nn.Conv1d(inp_size, d_model, kernel_size=9, padding=4)
        self.linear = nn.Linear(alphabet_size, d_model)
        self.proj = nn.Linear(2*args.hidden_dim, args.hidden_dim)
        # self.embedder = nn.Linear((2 if expanded_simplex_input  else 1) *  self.alphabet_size,  args.hidden_dim)
        # self.embeddings = GPT2Embeddings(d_model, 16, max_position_embeddings=0)

        self.num_layers = num_layers
        self.hyena_layer = nn.ModuleList([create_block(
                                                        d_model=d_model, 
                                                        args=args, 
                                                        d_inner=4*d_model,
                                                        **filter_args,
                                                    ) for _ in range(self.num_layers)])
        # self.time_layers = nn.ModuleList([Dense(d_model, d_model) for _ in range(self.num_layers)])
        # self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(dropout)
        factory_kwargs = {"device": None, "dtype": None}
        # self.embeddings = GPT2Embeddings(
        #         d_model, alphabet_size, 0, **factory_kwargs
        #     )
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        if not classifier:
            self.out_linear = nn.Linear(d_model, alphabet_size)
        initializer_cfg = None
        self.apply(
            partial(
                _init_weights,
                n_layer=self.num_layers,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
    
    def forward(self, seq, t=None, cls = None, return_embedding=False, state=None):
        if t is None and self.for_representation:
            # pass
            t = torch.tensor([self.args.fix_alpha]).expand(seq.shape[0]).to(seq.device)
            # seq_, alphas = sample_cond_prob_path(self.args, seq, self.alphabet_size)
            # seq, prior_weights = expand_simplex(seq,alphas, self.args.prior_pseudocount)
            # t = alphas
        # feat = F.relu(self.linear(seq.permute(0, 2, 1))).permute(0,2,1)
        # t = torch.ones(seq.shape[0]).to(seq.device)

        seq = torch.nn.functional.one_hot(seq, num_classes=self.alphabet_size).type(self.linear.weight.dtype)
        # print(seq.shape)
        feat = F.relu(self.linear(seq))

        # feat = self.embeddings(seq)

        time_emb = F.relu(self.time_embedder(t))[:, None, :].repeat(1, feat.shape[1], 1)
        feat = torch.concatenate([feat, time_emb], dim=-1)
        feat = self.proj(feat)

        residual = None
        for i in range(self.num_layers):
            h = self.dropout(feat.clone())

            # h = self.norms[i](h)
            h, residual = self.hyena_layer[i](h, residual)
            feat = h
        feat = dropout_add_layer_norm(
            h,
            residual,
            self.ln_f.weight,
            self.ln_f.bias,
            self.dropout.p if self.training else 0.0,
            self.ln_f.eps,
            prenorm=False,
            residual_in_fp32=True,
        )
            # h = F.relu(self.hyena_layer[i](h))
            # if h.shape == feat.shape:
            #     feat = h + feat
            # else:
            #     feat = h
        if self.for_representation:
            return feat, None
        if not self.classifier:
            feat = self.out_linear(feat)
        return feat

    @property
    def d_output(self):
        return self.d_model

    # def forward(self, seq, t=None, cls = None, return_embedding=False, state=None):
    #     if t is None and self.for_representation:
    #         # t = torch.tensor([self.args.alpha_max])[None].expand(seq.shape[0]).to(seq.device)
    #         seq, alphas = sample_cond_prob_path(self.args, seq, self.alphabet_size)
    #         seq, prior_weights = expand_simplex(seq,alphas, self.args.prior_pseudocount)
    #         t = alphas
    #     time_emb = F.relu(self.time_embedder(t))
    #     feat = seq.permute(0, 2, 1)
    #     feat = F.relu(self.linear(feat)).permute(0, 2, 1)

    #     residual = None
    #     for i in range(self.num_layers):
    #         h = self.dropout(feat.clone())
    #         h = h + self.time_layers[i](time_emb)[:, None, :]
    #         # h = self.norms[i](h)
    #         h, residual = self.hyena_layer[i](h, residual)
    #         feat = h
    #     feat = dropout_add_layer_norm(
    #         h,
    #         residual,
    #         self.ln_f.weight,
    #         self.ln_f.bias,
    #         self.dropout.p if self.training else 0.0,
    #         self.ln_f.eps,
    #         prenorm=False,
    #         residual_in_fp32=True,
    #     )
    #         # h = F.relu(self.hyena_layer[i](h))
    #         # if h.shape == feat.shape:
    #         #     feat = h + feat
    #         # else:
    #         #     feat = h
    #     if self.for_representation:
    #         return feat, None
    #     if not self.classifier:
    #         feat = self.out_linear(feat)
    #     return feat
    
    # @property
    # def d_output(self):
    #     return self.d_model

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    glu_act=False,
):
    torch.manual_seed(2222)
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight", 'mha.in_proj_weight', 'Wqkv.weight']:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # nn.init.normal_(
                #     p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer)
                # )
                nn.init.kaiming_normal_(p)
            # elif name in ['mha.in_proj_bias', 'mha.out_proj_bias']:
            #     nn.init.kaiming_uniform_(p)
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    # nn.init.normal_(
                    #     p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer)
                    # )
                    nn.init.kaiming_normal_(p)
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    # nn.init.normal_(
                    #     p[: out_features // 2],
                    #     mean=0.0,
                    #     std=initializer_range / math.sqrt(2 * n_layer) * 2,
                    # )
                    nn.init.kaiming_normal_(p[: out_features//2])