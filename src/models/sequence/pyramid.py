import math
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm_parallel_residual
except ImportError:
    dropout_add_layer_norm_parallel_residual = None

try:
    from flash_attn.ops.rms_norm import dropout_add_rms_norm
except ImportError:
    dropout_add_rms_norm = None, None

try:
    from flash_attn.ops.rms_norm import dropout_add_rms_norm_parallel_residual
except ImportError:
    dropout_add_rms_norm_parallel_residual = None
from functools import partial
from torchvision.ops import StochasticDepth
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp
from src.models.sequence.bert_padding import (index_first_axis,
                                            index_put_first_axis, pad_input,
                                            unpad_input, unpad_input_only)

try:
    from src.models.sequence.flash_attn_triton import flash_attn_qkvpacked_func
except ImportError as e:
    flash_attn_qkvpacked_func = None

class Mamba(nn.Module):
    def __init__(
        self,
        d_model, # D
        d_state=16, # N
        d_conv=4,
        expand=2,
        dt_rank="auto", # auto is 64
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        return_last_state=False
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.return_last_state = return_last_state

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        return_last_state = self.return_last_state

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out, last_state = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=True
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y, last_state = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=True,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        if not return_last_state:
            return out
        else:
            return out, last_state

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class MambaBlock(nn.Module):

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout1=0., resid_dropout2=0.,
                 drop_path1=0., drop_path2=0., fused_dropout_add_ln=False, return_residual=False,
                 residual_in_fp32=False, sequence_parallel=False, mark_shared_params=False, return_last_state=False):
        """
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).

        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.

        return_residual: whether each of the sub-layers (mixer and mlp) will return the residual.
        This is for performance reason: for post-norm architecture, returning the input allows us
        to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        self.prenorm = prenorm
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        self.return_last_state = return_last_state
        if self.residual_in_fp32:
            assert self.prenorm, 'residual_in_fp32 is only compatible with prenorm=True'
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode='row')
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode='row')
            self.norm2 = norm_cls(dim)

        if self.fused_dropout_add_ln:
            assert dropout_add_layer_norm is not None, 'dropout_layer_norm is not installed'
            assert dropout_add_rms_norm is not None, 'dropout_layer_norm is not installed'
            assert (isinstance(self.norm1, (nn.LayerNorm, RMSNorm))
                    and isinstance(self.dropout1, nn.Dropout))

        # TD [2023-01-07]: TODO: During training, if sequence_parallel is False and dropout != 0.0,
        # then the input to each worker in the tensor parallel group will be different.
        # This would produce wrong outputs? Somehow we'd need to sync the RNG state across workers.
        # For now this is not an issue because we always use sequence_parallel=True during training
        # and only use sequence_parallel=False during inference.

        # Mark the norm parameters as "sequence_parallel" so that we run all-reduce on their grads.
        if sequence_parallel:
            for p in self.norm1.parameters():
                p._sequence_parallel = True
            if hasattr(self, 'norm2'):
                for p in self.norm2.parameters():
                    p._sequence_parallel = True
        # Mark the norm parameters as "shared_params" so that we sync their values at init.
        if mark_shared_params:
            for p in self.norm1.parameters():
                p._shared_params = True
            if hasattr(self, 'norm2'):
                for p in self.norm2.parameters():
                    p._shared_params = True

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None,
                mixer_subset=None, mixer_kwargs=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        fused_add_norm_fn = (dropout_add_rms_norm if RMSNorm and isinstance(self.norm1, RMSNorm)
                             else dropout_add_layer_norm)
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_path1(self.dropout1(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(torch.ones(
                        hidden_states.shape[:-1], device=hidden_states.device,
                        dtype=hidden_states.dtype)
                    )
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states, residual, self.norm1.weight, self.norm1.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps,
                    rowscale=rowscale1, prenorm=True, residual_in_fp32=self.residual_in_fp32
                )
            if mixer_kwargs is None:
                mixer_kwargs = {}
            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            if not self.return_last_state:
                hidden_states = self.mixer(hidden_states, **mixer_kwargs)
            else:
                hidden_states, last_ssm_state = self.mixer(hidden_states, **mixer_kwargs)
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                if not self.fused_dropout_add_ln:
                    dropped = self.drop_path2(self.dropout2(hidden_states))
                    residual = (dropped + residual) if residual is not None else dropped
                    hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(torch.ones(
                            hidden_states.shape[:-1], device=hidden_states.device,
                            dtype=hidden_states.dtype)
                        )
                    hidden_states, residual = fused_add_norm_fn(
                        hidden_states, residual, self.norm2.weight, self.norm2.bias,
                        self.dropout2.p if self.training else 0.0, self.norm2.eps,
                        rowscale=rowscale2, prenorm=True, residual_in_fp32=self.residual_in_fp32
                    )
                hidden_states = self.mlp(hidden_states)
            if not self.return_last_state:
                return hidden_states, residual
            else:
                return hidden_states, residual, last_ssm_state
        else:
            assert residual is None

            if not self.return_last_state:
                mixer_out = self.mixer(
                    hidden_states, **(mixer_kwargs if mixer_kwargs is not None else {})
                )
            else:
                mixer_out, last_ssm_state = self.mixer(
                    hidden_states, **(mixer_kwargs if mixer_kwargs is not None else {})
                )
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out
            if not self.fused_dropout_add_ln:
                hidden_states = self.norm1((self.drop_path1(self.dropout1(mixer_out))
                                            + hidden_states).to(dtype=self.norm1.weight.dtype))
            else:
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(torch.ones(
                        mixer_out.shape[:-1], device=mixer_out.device, dtype=mixer_out.dtype)
                    )
                hidden_states = fused_add_norm_fn(
                    mixer_out, hidden_states, self.norm1.weight, self.norm1.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps,
                    rowscale=rowscale1, prenorm=False
                )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
                if not self.fused_dropout_add_ln:
                    hidden_states = self.norm2((self.drop_path2(self.dropout2(mlp_out))
                                                + hidden_states).to(dtype=self.norm2.weight.dtype))
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(torch.ones(
                            mlp_out.shape[:-1], device=mlp_out.device, dtype=mlp_out.dtype)
                        )
                    hidden_states = fused_add_norm_fn(
                        mlp_out, hidden_states, self.norm2.weight, self.norm2.bias,
                        self.dropout2.p if self.training else 0.0, self.norm2.eps,
                        rowscale=rowscale2, prenorm=False
                    )
            if not self.return_last_state:
                return hidden_states
            else:
                return hidden_states, last_ssm_state

if __name__=='__main__':
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    x = torch.rand(3, 100, 1024).to(device)
    mamba = Mamba(1024, use_fast_path=False, return_last_state=True).to(device)
    y, last_sate = mamba(x)
    from torch.optim import Adam
    from torch.nn import MSELoss
    lossfn = MSELoss()
    loss = lossfn(y.ravel(), x.ravel())
    mamba.zero_grad()
    loss.backward()



    assert x.shape==y.shape


class BertLayer(nn.Module):
    """Composes the Mosaic BERT attention and FFN blocks into a single layer."""

    def __init__(self, d_model, num_heads, intermediate_size, layer_idx, device=None, dtype=None):
        super(BertLayer, self).__init__()
        self.attention = BertUnpadAttention(d_model=d_model, num_heads=num_heads)
        # self.mlp = BertGatedLinearUnitMLP(d_model=d_model, intermediate_size=intermediate_size)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seqlen: int,
        subset_idx: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.
        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            seqlen: int
            subset_idx: () set of indices whose values we care about at the end of the layer
                        (e.g., the masked tokens, if this is the final layer).
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen_in_batch)
            bias: None or (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
        """
        origin_shape = hidden_states.shape
        attention_output = self.attention(hidden_states, cu_seqlens, seqlen,
                                          subset_idx, indices, attn_mask, bias).reshape(origin_shape).contiguous()
        # layer_output = self.mlp(attention_output)
        return attention_output

class BertSelfOutput(nn.Module):

    def __init__(self, d_model, layer_norm_eps=1e-5, hidden_dropout_prob=0.1):
        super().__init__()
        hidden_size = d_model
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size,
                                      eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertUnpadSelfAttention(nn.Module):

    def __init__(self, d_model, num_attention_heads, attention_probs_dropout_prob=0.1):
        super().__init__()
        hidden_size = d_model
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f'The hidden size ({hidden_size}) is not a multiple of the number of attention '
                f'heads ({num_attention_heads})')

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size /
                                       num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.p_dropout = attention_probs_dropout_prob
        self.Wqkv = nn.Linear(self.all_head_size, 3 * hidden_size)

        # Warn if defaulting to pytorch because of import issues
        # if flash_attn_qkvpacked_func is None:
        #     warnings.warn(
        #         'Unable to import Triton; defaulting MosaicBERT attention implementation to pytorch (this will reduce throughput when using this model).'
        #     )

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                max_seqlen_in_batch: int, indices: torch.Tensor,
                attn_mask: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Perform self-attention.
        If dropout is zero, then we can use the Triton kernel, so we do that. However, if not, we send through a standard PyTorch
        implementation of self-attention.
        The arguments are unpadded, and our implementations of attention require padded arguments,
        so we first call `pad_input`. Once we compute attention, we re-unpad our outputs for the other layers.
        The pad/unpad operations add overhead, but not sending pad tokens through ffs saves compute.
        It is possible to write an unpadded implementation of attention (in Triton and PyTorch), which we will eventually do.
        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen_in_batch: int
            indices: (total_nnz,)
            attn_mask: (batch, max_seqlen_in_batch)
            bias: (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
        Returns:
            attention: (total_nnz, dim)
        """
        qkv = self.Wqkv(hidden_states)
        qkv = pad_input(qkv, indices, cu_seqlens.shape[0] - 1,
                        max_seqlen_in_batch)  # batch, max_seqlen_in_batch, thd
        qkv = rearrange(qkv,
                        'b s (t h d) -> b s t h d',
                        t=3,
                        h=self.num_attention_heads)
        if self.p_dropout or flash_attn_qkvpacked_func is None:
            # if we have nonzero attention dropout (e.g. during fine-tuning) or no Triton, compute attention in PyTorch
            q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3)  # b h s d
            k = qkv[:, :, 1, :, :].permute(0, 2, 3, 1)  # b h d s
            v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)  # b h s d
            attention_scores = torch.matmul(q, k) / math.sqrt(
                self.attention_head_size)
            attention_scores = attention_scores + bias
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            attention = torch.matmul(attention_probs, v).permute(0, 2, 1,
                                                                 3)  # b s h d
        else:
            # Triton implementation only supports 0 attention dropout
            convert_dtype = qkv.dtype not in [torch.float16, torch.bfloat16]
            if convert_dtype:
                # Triton implementation only supports fp16 and bf16
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.float16)
                bias_dtype = bias.dtype
                bias = bias.to(torch.float16)
                attention = flash_attn_qkvpacked_func(qkv, bias)
                attention = attention.to(orig_dtype)
                bias = bias.to(bias_dtype)
            else:
                attention = flash_attn_qkvpacked_func(qkv, bias)

        # attn_mask is 1 for attend and 0 for don't
        attention = unpad_input_only(attention, torch.squeeze(attn_mask) == 1)
        return rearrange(attention, 'nnz h d -> nnz (h d)')

class BertUnpadAttention(nn.Module):
    """Chains attention, Dropout, and LayerNorm for Mosaic BERT."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self = BertUnpadSelfAttention(d_model=d_model, num_attention_heads=num_heads)
        self.output = BertSelfOutput(d_model=d_model)

    def forward(
        self,
        input_tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_s: int,
        subset_idx: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for scaled self-attention without padding.
        Arguments:
            input_tensor: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_s: int
            subset_idx: () set of indices whose values we care about at the end of the layer
                        (e.g., the masked tokens, if this is the final layer).
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen_in_batch)
            bias: None or (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
        """
        self_output = self.self(input_tensor, cu_seqlens, max_s, indices,
                                attn_mask, bias)
        if subset_idx is not None:
            return self.output(index_first_axis(self_output, subset_idx),
                               index_first_axis(input_tensor, subset_idx))
        else:
            return self.output(self_output, input_tensor)


class BertGatedLinearUnitMLP(nn.Module):
    """Applies the FFN at the end of each Mosaic BERT layer.
    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality, but
    introduces Gated Linear Units.
    Note: Mosaic BERT adds parameters in order to implement Gated Linear Units. To keep parameter count consistent with that of a
    standard Hugging Face BERT, scale down `config.intermediate_size` by 2/3. For example, a Mosaic BERT constructed with
    `config.intermediate_size=2048` will have the same parameter footprint as its Hugging Face BERT counterpart constructed
    with the `config.intermediate_size=3072`.
    However, in most cases it will not be necessary to adjust `config.intermediate_size` since, despite the increased
    parameter size, Mosaic BERT typically offers a net higher throughput than a Hugging Face BERT built from the same `config`.
    """

    def __init__(self, d_model, intermediate_size, hidden_dropout_prob=0.1, layer_norm_eps=1e-5, device=None, dtype=None):
        super().__init__()
        hidden_size = d_model
        self.gated_layers = nn.Linear(hidden_size,
                                      intermediate_size * 2,
                                      bias=False)
        self.act = nn.GELU(approximate='none')
        self.wo = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(hidden_size,
                                      eps=layer_norm_eps)
        self.intermediate_size = intermediate_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.
        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """
        residual_connection = hidden_states
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)
        gated = hidden_states[:, :self.intermediate_size]
        non_gated = hidden_states[:, self.intermediate_size:]
        hidden_states = self.act(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + residual_connection)
        return hidden_states

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torch.nn import functional as F
# from einops import rearrange
# from tqdm import tqdm

# import math
# import os
# import urllib.request
# from zipfile import ZipFile

# from transformers import AutoTokenizer

# torch.autograd.set_detect_anomaly(True)

# def talor_normalize(x, method:str="xavier"):
#     if method=="euclidean":
#         return F.normalize(x, p=2., dim=-1)
#     elif method=="normal":
#         return torch.rand(x.shape[0], x.shape1)
#     elif method=="kaiming":
#         return nn.init.kaiming_normal_(x)
#     elif method=="xavier":
#         return nn.init.xavier_normal_(x)
#     else:
#         raise NotImplementedError("Invalid Normalize method!")

# class S6(nn.Module):
#     def __init__(self, d_state, batch_size, embed_dim, max_length, norm_method=None):
#         super().__init__()
#         self.N = d_state
#         self.B = batch_size
#         self.D = embed_dim
#         self.L = max_length
#         self.A = nn.Parameter(talor_normalize(torch.ones(self.D, self.N), method=norm_method))
#         self.B = torch.zeros(self.B, self.L, self.N) # fake, actually useless
#         self.C = torch.zeros(self.B, self.L, self.N) # fake, actually useless
#         self.delta = torch.zeros(self.B, self.L, self.D) # fake, actually useless

#         self.fc_b = nn.Linear(self.D, self.N)
#         self.fc_c = nn.Linear(self.D, self.N)
#         self.fc_delta = nn.Linear(self.D, self.D)
    
#     def discretize(self, ):
#         delta_A = torch.einsum("bld, bln->bldn", self.delta, self.A)
#         self.dA = torch.exp(delta_A) # BLDN
#         # self.dB = 1./(delta_A)*(torch.exp(delta_A)) #TODO how to write the formula if its not square?
#         self.dB = torch.einsum("bld, bln->bldn", self.delta, self.B)
#         return self.dA, self.dB
    
#     def forward(self, u):
#         self.B = self.fc_b(u)
#         self.C = self.fc_c(u)
#         self.delta = F.softplus(self.fc_delat(u))
#         dA, dB = self.discretize()



