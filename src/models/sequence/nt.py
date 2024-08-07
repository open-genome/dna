import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from typing import NewType
from dataclasses import dataclass, field
import torch.nn.functional as F
from collections import namedtuple

T = NewType("T", torch.Tensor)

SUPPORTED_FFN_ACTIVATIONS = ["GELU", "ReLU", "SiLU", 'gelu-no-approx']

class RMSNorm(nn.Module):
     def __init__(self,
                  d_model: int,
                  eps: float = 1e-5,
                  device: str ='cuda'):
         super().__init__()
         self.eps = eps
         self.weight = nn.Parameter(torch.ones(d_model, device=device))
 
 
     def forward(self, x):
         output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
 
         return output

def get_norm(norm_type, embed_dim, eps):
    if norm_type == "rms":
        return RMSNorm(embed_dim, eps)
    if norm_type == "layer":
        return nn.LayerNorm(embed_dim, eps)
    else:
        raise NotImplementedError("Illegal Norm")

def get_activation_fn(activation_name):
    if activation_name not in SUPPORTED_FFN_ACTIVATIONS:
        raise NotImplementedError("Illegal activation function")
    elif activation_name == "gelu-no-approx":
        return nn.GELU(approximate="none")
    return getattr(nn, activation_name)

class GLU_MLP(nn.Module):
    def __init__(self, fc1, fc2, act_fn):
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2
        self.act_fn = act_fn
    def forward(self, x):
        x1, x2 = torch.chunk(self.fc1(x), chunks=2, dim=-1)
        x = self.act_fn(x1) * x2
        return self.fc2(x)

def init_linear(input_dim, output_dim, method="kaiming"):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    lin = nn.Linear(input_dim, output_dim).to(device)
    if method=="kaiming":
        nn.init.kaiming_uniform_(lin.weight, a=2.0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(lin.bias)
    else:
        raise NotImplementedError("Illegal init method for linear layer")
    return lin

class SelfAttentionBlock(nn.Module):
    """
    Attention block made of self-attention.
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        key_size: Optional[int] = None,
        use_rotary_embedding: bool = False,
        add_bias_kv: bool = False,
        ffn_activation_name: str = "GELU",
        use_glu_in_ffn: bool = False,
        norm_type = "rms", # "rms, layer"
        pre_layer_norm: bool = True,
        name: Optional[str] = None,
        add_bias_ffn: bool= True,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        # Add checks on dimensions
        if key_size is None:
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"The embedding dimension should be divisible by the number of "
                    f"heads, however provided embedding dimension is {embed_dim} and "
                    f"the number of heads is {num_heads}."
                )
            else:
                key_size = embed_dim // num_heads

        # Get ffn activation function
        self._pre_layer_norm = pre_layer_norm
        self._use_glu_in_fnn = use_glu_in_ffn

        # Define layers
        if use_glu_in_ffn:
            # user should multiply ffn_embed_dim by 2/3 when using GLU
            # to keep total number of parameters equal
            # see https://arxiv.org/pdf/2002.05202.pdf. for more details
            # we multiply by 2 here as the output will be split in 2 for GLU
            ffn_embed_dim = int(2 * ffn_embed_dim)

        self.norm_mlp = get_norm(norm_type,embed_dim,eps=layer_norm_eps)
        self.norm_attention = get_norm(norm_type, embed_dim, eps=layer_norm_eps)
        self.self_atten = MultiHeadAttention(
            num_heads=num_heads,
            key_size=key_size,
            model_size=embed_dim,
            add_bias_kv=add_bias_kv,
            use_rotary_embedding=use_rotary_embedding,
            embed_size=embed_dim,
        )

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ffn_embed_dim, bias=add_bias_ffn),
            get_activation_fn(activation_name=ffn_activation_name),
            nn.Linear(ffn_embed_dim, embed_dim, bias=add_bias_ffn)
        ).to(device)

    def self_attention(
        self,
        x: T,
        attention_mask: Optional[T] = None,
        attention_weight_bias: Optional[T] = None,
    ):
        """
        Applies the self attention mechanism.

        Args:
            x: Input token embeddings of shape (batch_size, seq_len, embed_dim).
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            Dictionary containing the output embeddings and the attention weights.
        """

        return self.self_atten(
            x,
            x,
            x,
            attention_mask=attention_mask,
            attention_weight_bias=attention_weight_bias,
        )

    def forward(
        self,
        x: T,
        attention_mask: Optional[T] = None,
        attention_weight_bias: Optional[T] = None,
    ):
        """
        Computes the output of the attention layer.

        Args:
            x: Input token embeddings of shape (batch_size,seq_len,embed_dim).
            attention_mask: Attention mask of shape (batch_size, 1,seq_len, seq_len).

        Returns:
            A dictionary containing the output embeddings and the attention weights.
        """

        # Self-Attention
        res = x
        if self._pre_layer_norm:
            x = self.norm_attention(x)

        output = self.self_attention(
            x=x,
            attention_mask=attention_mask,
            attention_weight_bias=attention_weight_bias,
        )

        if not self._pre_layer_norm:
            output["embeddings"] = self.norm_attention(
                output["embeddings"] + res
            )
            x = output["embeddings"]
        else:
            x = output["embeddings"]
            x = res + x

        # MLP
        if self._pre_layer_norm:
            x = x+self.norm_mlp(self.mlp(x))
        else:
            x = self.norm_mlp(x+self.mlp(x))

        output["embeddings"] = x
        return output  # type: ignore

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with masking applied. Modified from the core implementation to
    support biases in keys and values.
    """

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        embed_size: int,
        use_rotary_embedding: bool = False,
        add_bias_kv: bool = False,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        init_method: str = "kaiming"
    ):
        """
        Args:
            num_heads: Number of independent attention heads.
            key_size: The size of keys and queries used for attention.
            use_rotary_embedding: If true, adds rotary embeddings to the key and query
                heads (see RoFormer https://arxiv.org/pdf/2104.09864.pdf).
            add_bias_kv: If True, appends biases to key and query heads, used in ESM
                model (https://www.biorxiv.org/content/10.1101/622803v4.full.pdf).
            value_size: Optional size of the value projection. If None, defaults
                to the key size.
            model_size: Optional size of the output embedding. If None, defaults
                to the key size multiplied by the number of heads.
            name: Optional name for this module.
        """
        # 1. positional embedding
        # 2. normal embedding
        # 3. qkv projection
        # 4. self attention in each head
        # 5. concatenate
        # 6. Linear
        super().__init__()
        self.num_heads=num_heads
        self.key_size=key_size
        self.value_size=value_size
        self.model_size=model_size
        if model_size == None:
            self.model_size = embed_size
        self.embed_size=embed_size

        self.q_proj = init_linear(self.embed_size, self.key_size*num_heads, init_method)
        self.k_proj = init_linear(self.embed_size, self.key_size*num_heads, init_method)
        if self.value_size != None:
            self.v_proj = init_linear(self.embed_size, self.value_size*num_heads, init_method)
        else:
            self.v_proj = init_linear(self.embed_size, self.key_size*num_heads, init_method)


        if add_bias_kv:
            self._bias_k = nn.parameter(torch.zeros(1, 1, self.num_heads, self.key_size))
            self._bias_v = nn.parameter(torch.zeros(1, 1, self.num_heads, self.value_size))
        else:
            self._bias_k = None
            self._bias_v = None

        if self.value_size != None:
            self.out_proj = init_linear(self.value_size*num_heads, self.model_size, init_method)
        else:
            self.out_proj = init_linear(self.key_size*num_heads, self.model_size, init_method)
        self._use_rotary_embedding = use_rotary_embedding
    
    def attention_weights(
        self,
        query: T,
        key: T,
        attention_mask: Optional[T] = None,
        attention_weight_bias: Optional[T] = None,
    ):
        """
        Computes the attention weights.

        Args:
            query: Embedding sequence to compute queries.
            key: Embedding sequence to compute keys.
            attention_mask: Input attention_mask. Defaults to None.

        Returns:
            Attention weights.
        """

        query_heads = self.q_proj(query).reshape(-1, query.shape[1], self.num_heads, self.key_size) # B, L, H, K
        key_heads = self.k_proj(key).reshape(-1,key.shape[1], self.num_heads, self.key_size)

        # Add bias for key (see ESM architecture)
        # jmp_policy = hk.mixed_precision.current_policy()
        # if jmp_policy is None:
        #     # default float32
        #     compute_dtype = jnp.float32
        # else:
        #     # cast to jmp policy if specified
        #     compute_dtype = jmp_policy.compute_dtype

        if self._bias_k is not None:
            batch_size = key_heads.shape[0]

            attention_bias = self._bias_k.repeat(batch_size, 1, 1, 1)

            key_heads = torch.cat((key_heads, attention_bias), dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones_like(attention_mask[..., :1], dtype=torch.bool),
                    ),
                    dim=-1,
                )

        if self._use_rotary_embedding:
            query_heads, key_heads = RotaryEmbedding(
                self.key_size, name="rotary_embed"
            )(query_heads, key_heads)

        attention_logits = torch.einsum("b L h k,b l h k->b h L l", query_heads, key_heads)
        sqrt_key_size = torch.sqrt(torch.tensor(self.key_size)).type(query.dtype)
        attention_logits = attention_logits / sqrt_key_size

        if attention_mask is not None:
            assert len(attention_mask.shape) == len(attention_logits.shape)
            attention_logits = torch.where(attention_mask, attention_logits, torch.tensor(-1e30))

        if attention_weight_bias is None:
            attention_weights = F.softmax(attention_logits, dim=-1)
        else:
            attention_weights = F.softmax(attention_logits + attention_weight_bias, dim=-1)

        return attention_weights

    def compute_embeddings(
        self,
        value: T,
        attention_weights: T,
    ):
        """
        Computes the output embeddings.

        Args:
            value: Embedding sequence to compute values.
            attention_weights: Attention weights.

        Returns:
            Output embeddings.
        """

        value_heads = self.v_proj(value).reshape(-1, value.shape[1], self.num_heads, self.key_size)

        if self._bias_v is not None:
            batch_size = value_heads.shape[0]
            # Add bias for key (see ESM architecture)
            # jmp_policy = hk.mixed_precision.current_policy()
            # if jmp_policy is None:
            #     # default float32
            #     compute_dtype = jnp.float32
            # else:
            #     # cast to jmp policy if specified
            #     compute_dtype = jmp_policy.compute_dtype

            attention_bias = self._bias_v.repeat(batch_size, 1, 1, 1)
            value_heads = torch.concatenate((value_heads, attention_bias), dim=1)

        attention = torch.einsum("...htT,...Thd->...thd", attention_weights, value_heads)
        # attention = torch.einsum("...hlL,...Lhd->...Lhd", attention_weights, value_heads)

        # Concatenate attention matrix of all heads into a single vector.
        attention_vec = torch.reshape(attention, (*attention.shape[:-2], -1))
        return self.out_proj(attention_vec)

    def forward(
        self,
        query: T,
        key: T,
        value: T,
        attention_mask: Optional[T] = None,
        attention_weight_bias: Optional[T] = None,
    ):
        """
        Computes both the embeddings and the attention weights.

        Args:
            query: Embedding sequence to compute queries.
            key: Embedding sequence to compute keys.
            value: Embedding sequence to compute values.
            attention_mask: Mask to be applied during the attention layers.
                Triangular for autoregressive models. Defaults to None.

        Returns:
            Dictionary containing the output embeddings and the attention weights.
        """

        attention_weights = self.attention_weights(
            query,
            key,
            attention_mask=attention_mask,
            attention_weight_bias=attention_weight_bias,
        )
        embeddings = self.compute_embeddings(value, attention_weights)

        return {"embeddings": embeddings, "attention_weights": attention_weights}
    
# Constant used in Sinusoidal/Rotary Embeddings, reference to this value can be found
# on page 6 of https://arxiv.org/pdf/1706.03762.pdf and page 5 of
# https://arxiv.org/abs/2104.09864
# These rotary positional embeddings are proper to ESM implementation.
# dimensions in key space are rotated 2 by 2. The key difference with
# GPT's one is that in this case each 2 dimensions rotated together are spaced
# by key_size//2
UPPER_FREQ = 10000


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding inspired by RoFormer:
    https://arxiv.org/abs/2104.09864
    https://github.com/ZhuiyiTechnology/roformer .
    """

    def __init__(
        self,
        key_size: int
    ):

        """
        Args:
            key_size: Dimension of one head.
            name: Name of the layer. Defaults to None.
        """
        super().__init__()
        self._inv_freq = 1.0 / (UPPER_FREQ ** (torch.arange(0, key_size, 2) / key_size))

    def _compute_cos_sin_tables(
        self,
        heads: T,
    ):
        """
        Computes the cosinus and sinus for rotation.

        Args:
            heads: Query or key heads of shape (batch_size, seq_len, num_heads,
            key_size).

        Returns:
            Cosinus positional embedding of shape (1, seq_len, 1,
                key_size).
            Sinus positional embedding of shape (1, seq_len, 1,
                key_size).
        """
        seq_len = heads.shape[1]

        self._seq_len_cached = seq_len
        t = torch.arange(seq_len)
        freqs = torch.einsum("i,j->ij", t, self._inv_freq)
        emb = torch.concatenate((freqs, freqs), axis=-1, dtype=heads.dtype)

        # Compute cos and cast is as (1, seq_len, 1, key_size) to be applied to queries
        # of shape (batch_size, seq_len, num_heads, key_size)
        cos_cached = torch.cos(emb)[None, :, None, :]
        sin_cached = torch.sin(emb)[None, :, None, :]

        return cos_cached, sin_cached

    def _apply_rotary_pos_emb(
        self, heads: T, cos: T, sin: T
    ):
        """
        Applies the rotary positional embedding to the heads.

        Args:
            heads: Query or key heads of shape (batch_size, seq_len, num_heads,
                key_size).
            cos: Cosinus values.
            sin: Sinus values.

        Returns:
            Embedded heads of shape (batch_size, seq_len, num_heads,
                key_size).
        """

        # Rotate x
        x1, x2 = heads[..., : heads.shape[-1] // 2], heads[..., heads.shape[-1] // 2 :]
        heads_rotated = torch.concatenate((-x2, x1), axis=-1)

        embedded_heads = (heads * cos) + (heads_rotated * sin)
        return embedded_heads

    def forward(
        self, query_heads: T, key_heads: T
    ):
        """
        Applies rotary embeddings to query_heads and key_heads.

        Args:
            query_heads: Query heads of shape
                (batch_size, seq_len, num_heads, key_size).
            key_heads: Key heads of shape (batch_size, seq_len, num_heads, key_size).

        Returns:
            Embedded query heads.
            Embedded key heads.
        """
        cos, sin = self._compute_cos_sin_tables(query_heads)

        return (
            self._apply_rotary_pos_emb(query_heads, cos, sin),
            self._apply_rotary_pos_emb(key_heads, cos, sin),
        )
    
@dataclass
class NucleotideTransformerConfig:
    """
    Parameters to initialize a Nucleotide Transformer model.

    Args:
        alphabet_size: Token vocabulary.
        pad_token_id: ID of pad token.
        mask_token_id: ID of mask token.
        max_positions: Maximum sequence length.
        embed_scale: Correction ratio applied to the embeddings to make up for the
            norm difference between the input during training and inference.
        emb_layer_norm_before: Whether to use layer norm before the first attention
            layer.
        attention_heads: Number of attention heads.
        key_size: The dimension of the query, key, and values within each attention
            head, if not specified, it is set to attention_heads//embed_dim.
            It can be useful to set a custom key size if we want to impose the size of
            the query, key and value tensor ( for example, tensors shaped with
            power of 2 are more efficiently handled on TPUs ).
            Note: Parametrizing the model with a custom key size has been done in :
            Brown, Tom, et al. "Language models are few-shot learners."
            Advances in neural information processing systems 33 (2020): 1877-1901.
        embed_dim: Embedding dimension.
        ffn_embed_dim: Feed forward embedding dimension.
        num_layers: Number of attention blocks.
        token_dropout: Token dropout.
        masking_ratio: Masking ratio (used if token dropout is enabled).
        masking_prob: Masking probability (used if token dropout is enabled).
        use_gradient_checkpointing: Whether to use gradient checkpointing (checkpoint
            gradients in the forward pass to reduce the computation in the backward).
    """

    alphabet_size: int
    pad_token_id: int
    mask_token_id: int

    max_positions: int = 1000
    embed_scale: float = 1.0

    # architecture
    emb_layer_norm_before: bool = False
    attention_heads: int = 20
    key_size: Optional[int] = None
    embed_dim: int = 1280
    ffn_embed_dim: int = 5120
    num_layers: int = 24
    positional_embedding: Optional[str] = "learned"
    add_bias_kv: bool = False
    add_bias_ffn: bool = True
    use_rotary_embedding: bool = False
    ffn_activation_name: str = "gelu-no-approx"
    use_glu_in_ffn: bool = False
    layer_norm_eps: float = 1e-5
    pre_layer_norm: bool = True

    # dropout
    token_dropout: bool = False
    masking_ratio: float = 0.1
    masking_prob: float = 0.8

    # logging
    use_gradient_checkpointing: bool = False

    # return
    embeddings_layers_to_save: Tuple[int, ...] = ()
    attention_maps_to_save: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Checks that the given values are compatible.
        """
        if self.key_size is None:
            if not self.embed_dim % self.attention_heads == 0:
                raise ValueError(
                    f"When no key size is provided, the embedding dimension should be "
                    f"divisible by the number of heads, however provided embedding "
                    f"dimension is {self.embed_dim} and the number of heads is "
                    f"{self.attention_heads}."
                )
            self.key_size = self.embed_dim // self.attention_heads

class ESMLearnedPositionalEmbeddings(nn.Module):
    """
    Learned positional embeddings to be added to token embeddings. Specific to ESM as it
    is implemented by shifting the positions by 2 (1 + padding_idx).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: int
    ):
        """
        Args:
            vocab_size: Tokenizer's vocabulary size.
            embed_dim: Embedding size.
            padding_idx: Index attributed to the padding
                token. Defaults to 1.
            name: Name of the layer. Defaults to None.
        """
        super().__init__()
        self.padding_idx = padding_idx
        self._embed_layer = nn.Embedding(vocab_size + padding_idx + 1, embed_dim)
        self.vocab_size = vocab_size + padding_idx + 1

    def forward(self, tokens: T) -> T:
        mask = tokens != self.padding_idx
        positions = torch.cumsum(mask, axis=1) * mask + self.padding_idx

        return self._embed_layer(positions)

class TokensDropout(nn.Module):
    """
    Tokens dropout layer.
    """

    def __init__(
        self,
        embed_dim: int,
        pad_token_id: int,
        mask_token_id: int,
        masking_ratio: float,
        masking_prob: float
    ):
        """
        Args:
            embed_dim: Embedding dimension.
            pad_token_id: ID of the pad token.
            mask_token_id: ID of the pad token.
            masking_ratio: Masking ratio.
            masking_prob: Probability to mask.
            name: Name of the layer. Defaults to None.
        """
        super().__init__()
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.masking_ratio = masking_ratio
        self.masking_prob = masking_prob
        self.embed_dim = embed_dim

    def forward(self, x, tokens):

        padding_mask_tokens = tokens == self.pad_token_id
        tokens_repeated = torch.repeat(
            tokens[:, :, None], repeats=self.embed_dim, axis=-1
        )
        x = torch.where(tokens_repeated == self.mask_token_id, 0.0, x)
        mask_ratio_train = self.masking_ratio * self.masking_prob
        src_lengths = (~padding_mask_tokens).sum(-1)
        mask_ratio_observed = (tokens == self.mask_token_id).sum(-1) / src_lengths
        x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
        return x

def build_padding_attention_mask(tokens, pad_token_id: int):
    """
    Builds a padding mask from a sequence of tokens by masking <pad> in the attention.

    Args:
        tokens: Batch of sequences of shape (batch_size, seq_len).
        pad_token_id: Int corresponding to the <pad> token to mask.

    Returns:
        Batch of attention masks, masking out <pad> tokens.
    """
    padding_mask = tokens != pad_token_id
    padding_mask = padding_mask[:, None, :]
    padding_mask = torch.einsum("bhT, bht->bhtT", padding_mask, padding_mask)
    return padding_mask

class RobertaLMHead(nn.Module):
    """
    Roberta Language Model head. Transform final attention layer output into a
    distribution over tokens at each position.
    """

    def __init__(self, embed_dim: int, alphabet_size: int):
        """
        Args:
            embed_dim: Embedding dimension.
            alphabet_size: Number of tokens in the alphabet.
            name: Name of the layer. Defaults to None.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.alphabet_size = alphabet_size

        # Define layers
        self._first_layer_norm =nn.LayerNorm(
            embed_dim, eps=1e-5, elementwise_affine=True
        )
        self._fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self._final_fc = nn.Linear(self.embed_dim, self.alphabet_size)
        self._second_layer_norm = nn.LayerNorm(
            embed_dim, eps=1e-5, elementwise_affine=True
        )

    def forward(self, x: T) -> Dict[str, T]:
        x = self._first_layer_norm(x)
        # Embeddings are computed after the first layer norm to be consistent with ESM
        embeddings = x
        x = self._fc1(x)
        x = F.gelu(x)
        x = self._second_layer_norm(x)

        # Compute logits
        logits = self._final_fc(x)
        return {"embeddings": embeddings, "logits": logits}

class NucleotideTransformer(nn.Module):
    """
    Jax implementation of Nucleotide Transformer models.
    """

    def __init__(
        self,
        d_model: int, # no use, just to test nt's compatibility with hyena
        alphabet_size: int,
        pad_token_id: int,
        mask_token_id: int,
        pad_vocab_size_multiple: int,
        *args,

        max_positions: int = 2048,
        embed_scale: float = 1.0,

        # architecture
        emb_layer_norm_before: bool = False,
        attention_heads: int = 8,
        key_size: Optional[int] = None,
        embed_dim: int = 128,
        ffn_embed_dim: int = 512,
        num_layers: int = 2,
        positional_embedding: Optional[str] = "learned",
        add_bias_kv: bool = False,
        add_bias_ffn: bool = True,
        use_rotary_embedding: bool = False,
        ffn_activation_name: str = "gelu-no-approx",
        use_glu_in_ffn: bool = False,
        layer_norm_eps: float = 1e-5,
        pre_layer_norm: bool = True,

        # dropout
        token_dropout: bool = False,
        masking_ratio: float = 0.1,
        masking_prob: float = 0.8,

        # logging
        use_gradient_checkpointing: bool = False,

        # return
        embeddings_layers_to_save: Tuple[int, ...] = (),
        attention_maps_to_save: List[Tuple[int, int]] = field(default_factory=list),
        **kwargs
    ):
        """
        Initialize a Nucleotide Transformer model.

        Args:
            config: Dataclass containing model hyperparameters.
            name: Name for module (custom will break weight loading).
        """

        # self._config = config
        super().__init__()
        args = locals()
        del args['self']
        for arg, val in list(args.items()):
            setattr(self, arg, val)

        self._embed_layer = nn.Embedding(self.alphabet_size, self.embed_dim)

        if alphabet_size % pad_vocab_size_multiple != 0:
            alphabet_size += pad_vocab_size_multiple - (
                alphabet_size % pad_vocab_size_multiple
            )

        if self.positional_embedding == "learned":
            self._pos_embed_layer = ESMLearnedPositionalEmbeddings(
                self.max_positions, self.embed_dim, 1
            )

        self._lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            alphabet_size=self.alphabet_size
        )

        if self.emb_layer_norm_before:
            self.emb_ln_before = nn.LayerNorm(
                self.embed_dim,
                elementwise_affine=True,
                eps=1e-5
            )

        # Process attention maps to save requirement into more suitable format
        attention_maps_to_save = self.attention_maps_to_save
        # self._attention_layers_to_save = list({t[0] for t in attention_maps_to_save})
        # self._attention_maps_per_layer_to_save = {
        #     layer: [t[1] for t in attention_maps_to_save if t[0] == layer]
        #     for layer in self._attention_layers_to_save
        # }

        # Checking user request can be executed, raise error otherwise
        # max_layer = max(self._attention_layers_to_save + [0])
        # if max_layer > self.num_layers:
        #     raise ValueError(
        #         f"You are requiring attention maps for layer {max_layer}, "
        #         f"while the model has {self.num_layers} layers only."
        #     )

        # for layer, maps in self._attention_maps_per_layer_to_save.items():
        #     max_map = max(maps)
        #     if max_map > self.attention_heads:
        #         raise ValueError(
        #             f"You are requiring attention maps number {max_map} "
        #             f"at layer {layer}, while the model has {self.attention_heads} "
        #             f"only."
        #         )

    def apply_attention_blocks(
        self,
        x: T,
        outs: Dict[str, T],
        attention_mask: Optional[T] = None,
    ) -> Tuple[T, Dict[str, T]]:
        """
        Create the blocks of attention layers and applies them.

        Args:
            x: The sequence embedding.
            outs: A dictionary to carry through the attention layers which stores the
                intermediate sequence embedding and attention maps.
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            The output sequence embedding.
            The optional intermediate results (embeddings of the layer and attention
                weights).
        """

        layers = [
            self._attention_block(layer_idx)
            for layer_idx in range(self.num_layers)
        ]

        if self.use_gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint
            layers = [checkpoint(layer) for layer in layers]

        for layer_idx, layer in enumerate(layers):
            output = layer(
                x=x, attention_mask=attention_mask, attention_weight_bias=None
            )
            x = output["embeddings"]
            # Save intermediate embeddings if needed
            # if (layer_idx + 1) in self.embeddings_layers_to_save:
            #     outs[f"embeddings_{(layer_idx + 1)}"] = output["embeddings"]
            # Save intermediate attention maps if needed
            # if (layer_idx + 1) in self._attention_layers_to_save:
            #     for map_number in self._attention_maps_per_layer_to_save[layer_idx + 1]:
            #         dkey = f"attention_map_layer_{layer_idx + 1}_number_{map_number}"
            #         outs[dkey] = output["attention_weights"][:, map_number + 1]

        return x, outs


    def _attention_block(self, layer_idx: int) -> SelfAttentionBlock:
        return SelfAttentionBlock(  # type: ignore
            num_heads=self.attention_heads,
            embed_dim=self.embed_dim,
            key_size=self.key_size,
            ffn_embed_dim=self.ffn_embed_dim,
            add_bias_kv=self.add_bias_kv,
            add_bias_ffn=self.add_bias_ffn,
            ffn_activation_name=self.ffn_activation_name,
            use_glu_in_ffn=self.use_glu_in_ffn,
            use_rotary_embedding=self.use_rotary_embedding,
            layer_norm_eps=self.layer_norm_eps,
            pre_layer_norm=self.pre_layer_norm,
        )

    def forward(
        self,
        tokens,
        attention_mask: Optional[T] = None,
        state=None
    ):
        """
        Computes the embeddings based on the input tokens.

        Args:
            tokens: Input tokens out of the tokenizer of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len).
                If no mask is provided, a mask by default which equals 1 over all non
                pad tokens and 0 over pad tokens is computed.

        Returns:
            Dictionary containing the final embeddings and logits.
        """
        # Prepare outputs dict
        outs = {}

        # Compute embeddings
        x = self._embed_layer(tokens)
        # Tokens dropout if needed
        if self.token_dropout:
            x = TokensDropout(
                embed_dim=self.embed_dim,
                mask_token_id=self.mask_token_id,
                pad_token_id=self.pad_token_id,
                masking_ratio=self.masking_ratio,
                masking_prob=self.masking_prob,
            )(x, tokens)

        # RoBERTa's mask scaling factor
        x = self.embed_scale * x

        if self.positional_embedding == "learned":
            # Add check that the sequence fed into the transformer is not longer
            # than the max positions used to instantiate the learned positional
            # embeddings layer
            max_length_authorized = (
                self._pos_embed_layer.vocab_size
                - self._pos_embed_layer.padding_idx
                - 1
            )
            assert tokens.shape[1] <= max_length_authorized, (
                "Inputs to the learned positional embeddings layer have a length "
                f"{x.shape[1]} greater than the max positions used to instantiate "
                f"it: {max_length_authorized}"
            )
            x = x + self._pos_embed_layer(tokens)

        if self.emb_layer_norm_before:
            x = self.emb_ln_before(x)

        # Attention mask
        if attention_mask is None:
            attention_mask = build_padding_attention_mask(
                tokens=tokens, pad_token_id=self.pad_token_id
            )

        # construct a tower of attention layers
        x, outs = self.apply_attention_blocks(
            x=x,
            outs=outs,
            attention_mask=attention_mask,
        )

        # Language Model Head
        lm_head_outs = self._lm_head(x)
        sequence_mask = attention_mask[:, 0, :, 0][:, :, None]
        lm_logits = torch.where(sequence_mask, lm_head_outs["logits"], 0)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])

        embeddings = lm_head_outs["embeddings"]
        # Save final embeddings if needed
        if self.num_layers in self.embeddings_layers_to_save:
            outs[f"embeddings_{self.num_layers}"] = embeddings

        return CausalLMOutput(logits=lm_logits), None  # type: ignore


# def build_nucleotide_transformer_fn(
#     model_config: NucleotideTransformerConfig,
#     compute_dtype: jnp.dtype = jnp.float32,
#     param_dtype: jnp.dtype = jnp.float32,
#     output_dtype: jnp.dtype = jnp.float32,
#     model_name: Optional[str] = None,
# ):
#     """
#     Creates the model's forward pass.

#     Args:
#         model_config: Model hyperparameters.
#         compute_dtype: the type of the activations. fp16 runs faster and is lighter in
#             memory. bf16 handles better large int, and is hence more stable ( it avoids
#             float overflows ).
#         param_dtype: if compute_dtype is fp16, the model weights will be cast to fp16
#             during the forward pass anyway. So in inference mode ( not training mode ),
#             it is better to use params in fp16 if compute_dtype is fp16 too. During
#             training, it is preferable to keep parameters in float32 for better
#             numerical stability.
#         output_dtype: the output type of the model. it determines the float precioson
#             of the gradient when training the model.
#         model_name: Model's name.

#     Returns:
#         Nucleotide Transformer model forward function.
#     """
#     policy = jmp.Policy(
#         compute_dtype=compute_dtype, param_dtype=param_dtype, output_dtype=output_dtype
#     )
#     hk.mixed_precision.set_policy(NucleotideTransformer, policy)

#     # Remove it in batch norm to avoid instabilities
#     norm_policy = jmp.Policy(
#         compute_dtype=jnp.float32, param_dtype=param_dtype, output_dtype=compute_dtype
#     )
#     hk.mixed_precision.set_policy(hk.BatchNorm, norm_policy)
#     hk.mixed_precision.set_policy(hk.LayerNorm, norm_policy)

#     def nucleotide_transformer_fn(
#         tokens: Tokens, attention_mask: Optional[AttentionMask] = None
#     ) -> TransformerOutput:
#         """Forward pass."""
#         # Run the encoder over the inputs.
#         encoder = NucleotideTransformer(config=model_config, name=model_name)
#         outs = encoder(
#             tokens=tokens,
#             attention_mask=attention_mask,
#         )
#         return outs

#     return nucleotide_transformer_fn