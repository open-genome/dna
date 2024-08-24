from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LinearSegmentedColormap
from modeling_esm import EsmForMaskedLM
from omegaconf import OmegaConf

# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species", trust_remote_code=True)
# model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species", trust_remote_code=True)
config = {
    "hidden_size": 512,
    "intermediate_size": 2048,
    "num_attention_heads": 16,
    "num_hidden_layers": 12,
    "max_position_embeddings": 2050,
    "vocab_size": 4096,
    "position_embedding_type": "rotary",
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "is_folding_model": False,
    "hyena_framework": False,
    "add_bias_fnn": False,
    "architectures": [
        "EsmForMaskedLM",
        "EsmForTokenClassification",
        "EsmForSequenceClassification"
    ],
    "attention_probs_dropout_prob": 0.0,
    "auto_map": {
        "AutoConfig": "esm_config.EsmConfig",
        "AutoModelForMaskedLM": "modeling_esm.EsmForMaskedLM",
        "AutoModelForSequenceClassification": "modeling_esm.EsmForSequenceClassification",
        "AutoModelForTokenClassification": "modeling_esm.EsmForTokenClassification"
    },
    "emb_layer_norm_before": False,
    "esmfold_config": None,
    "hidden_dropout_prob": 0.0,
    "mask_token_id": 2,
    "pad_token_id": 1,
    "tie_word_embeddings": False,
    "token_dropout": False,
    "torch_dtype": "float32",
    "transformers_version": "4.28.0",
    "use_cache": False,
    "vocab_list": None
}
config = OmegaConf.create(config)

# 创建模型实例
model = EsmForMaskedLM(config)

def reinitialize_weights(model):
    """
    Reinitialize all the weights of the model and keep track of initialized layers.
    """
    initialized_layers = set()
    
    def init_layer(layer):
        # Initialize layer weights and biases if applicable
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
            initialized_layers.add(layer)
        elif isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
            initialized_layers.add(layer)
        elif isinstance(layer, torch.nn.Embedding):
            torch.nn.init.xavier_uniform_(layer.weight)
            initialized_layers.add(layer)
        elif isinstance(layer, torch.nn.LayerNorm):
            torch.nn.init.ones_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            initialized_layers.add(layer)
        elif isinstance(layer, torch.nn.Dropout):
            # Dropout layers do not have weights
            initialized_layers.add(layer)
        elif isinstance(layer, torch.nn.GRU) or isinstance(layer, torch.nn.LSTM):
            # Initialize GRU/LSTM weights if applicable
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
            initialized_layers.add(layer)
        elif hasattr(layer, 'weight') and isinstance(layer.weight, torch.Tensor):
            # General case for layers with weights
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if hasattr(layer, 'bias') and layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
            initialized_layers.add(layer)
        
        # Recursively initialize child modules
        for child in layer.children():
            init_layer(child)
        
        return layer

    model.apply(init_layer)
    return initialized_layers

# 重新初始化权重
# initialized_layers = reinitialize_weights(model)


# Create a dummy DNA sequence and tokenize it
sequences = ["ATGCGTACGGTAGCTAGCTGAGCTAGCTGACGATCGTACGTCAGCTAGC"*10]
# Choose the length to which the input sequences are padded
max_length = len(sequences[0])
print(max_length)
tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="do_not_pad", max_length=max_length, truncation=True)["input_ids"]
print(tokens_ids.shape)

# Compute the embeddings
attention_mask = tokens_ids != tokenizer.pad_token_id
outputs = model(
    tokens_ids,
    attention_mask=attention_mask,
    encoder_attention_mask=attention_mask,
    output_attentions=True,
    output_hidden_states=True
)

# Extract attention weights
attention_weights = outputs.attentions[-1]  # Get the attention weights of the last layer
attention_weights = attention_weights[0].detach().numpy()  # Shape: [batch_size, num_heads, seq_len, seq_len]

# Number of layers and sequence length
num_layers = attention_weights.shape[0]
seq_len = attention_weights.shape[2]

# Create a grid for plotting
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
axes = axes.flatten()
cmap = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])

for layer_idx in range(num_layers):
    attention_map = attention_weights[layer_idx]
    # print(np.sum(attention_map, axis=1))
    # norm = Normalize(vmin=np.min(attention_weights), vmax=np.max(attention_weights))
    
    # Plot the attention map for this layer
    ax = axes[layer_idx]
    # cax = ax.imshow(attention_map, cmap=cmap, vmin=0, vmax=1)
    cax = ax.imshow(attention_map, cmap=cmap, vmin=np.min(attention_map), vmax=np.max(attention_map))
    ax.set_title(f"Layer {layer_idx + 1}")
    ax.set_xlabel("Key Positions")
    ax.set_ylabel("Query Positions")

# Add a colorbar
fig.colorbar(cax, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)

# Adjust layout
plt.tight_layout()
plt.savefig('attention_maps_rope.png')  # Save the attention maps as a PNG file
plt.close()

