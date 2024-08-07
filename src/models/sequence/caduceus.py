from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn


class Caduceus(nn.Module):
    def __init__(self, mode="ph", size=1.9, out_dim=2, *args, **kwargs):
        super().__init__()
        # self.d_model = config.hidden_size
        self.mode = mode
        if mode=="ph":
            if size == 470:
                model_name = "kuleshov-group/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3"
            elif size==1.9:
                model_name = "kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3"
            else:
                model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
            # tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            if size == 470:
                model_name = "kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3"
            elif size==1.9:
                model_name = "kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3"
            else:
                model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
            # tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.caduceus = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        self.d_model = self.caduceus.config.d_model
        self.score = nn.Linear(self.d_model, out_dim, bias=False)
        self.pooling_strategy = "mean"
        # self.lm_head



    def forward(self, input_ids, position_ids=None, inference_params=None, state=None): # state for the repo interface
        if isinstance(input_ids, list):
            input_ids_tensor = input_ids[0]
            attention_mask = input_ids[1]
        else:
            input_ids_tensor = torch.tensor(input_ids)
            attention_mask = None
        if position_ids is not None:
            position_ids_tensor = position_ids
        else:
            position_ids_tensor = None

        outputs = self.caduceus.caduceus(
            input_ids=input_ids,
            inputs_embeds=None,
            output_hidden_states=None,
            return_dict=None,
        )
        sequence_output = outputs[0]
        if self.mode == "ps":
            sequence_output = torch.stack(
                [
                    sequence_output[..., :self.d_model],
                    torch.flip(sequence_output[..., self.d_model:], dims=[1, 2])
                 ],
                dim=-1
            )
        # logits = sequence_output
        # Pool and get logits
        pooled_hidden_states = self.pool_hidden_states(sequence_output)
        # Potentially run `score` twice (with parameters shared) for conjoining
        if sequence_output.ndim == 4:  # bsz, seq_len, hidden_dim, 2 where last channel has the stacked fwd and rc reps
            logits_fwd = self.score(pooled_hidden_states[..., 0])
            logits_rc = self.score(pooled_hidden_states[..., 1])
            logits = (logits_fwd + logits_rc) / 2
        else:
            logits = self.score(pooled_hidden_states)
        return logits, None
        
    def pool_hidden_states(self, hidden_states, sequence_length_dim=1):
        """Pools hidden states along sequence length dimension."""
        if self.pooling_strategy == "mean":  # Mean pooling along sequence length dimension
            return hidden_states.mean(dim=sequence_length_dim)
        if self.pooling_strategy == "max":  # Max pooling along sequence length dimension
            return hidden_states.max(dim=sequence_length_dim).values
        if self.pooling_strategy == "last":  # Use embedding of last token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[-1, ...]
        if self.pooling_strategy == "first":  # Use embedding of first token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[0, ...]

    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.
        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model
    

from transformers import AutoModelForSequenceClassification

class Hyena(nn.Module):
    def __init__(self, seq_len=1, out_dim=2, *args, **kwargs):
        super().__init__()
        # self.d_model = config.hidden_size
        self.mode = seq_len
        if seq_len==1:
            model_name = "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"
            # tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif seq_len==16:
            model_name = "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"
        elif seq_len==32:
            model_name = "LongSafari/hyenadna-small-32k-seqlen-hf"
        elif seq_len==160:
            model_name = "LongSafari/hyenadna-medium-160k-seqlen-hf"
        elif seq_len==450:
            model_name = "LongSafari/hyenadna-medium-450k-seqlen-hf"
        else:
            model_name = "LongSafari/hyenadna-large-1m-seqlen-hf"
            # tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.hyena = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        self.d_model = self.hyena.config.d_model
        self.score = nn.Linear(self.d_model, out_dim, bias=False)
        # self.pooling_strategy = "mean"
        # self.lm_head



    def forward(self, input_ids, position_ids=None, inference_params=None, state=None): # state for the repo interface
        if isinstance(input_ids, list):
            input_ids_tensor = input_ids[0]
            attention_mask = input_ids[1]
        else:
            input_ids_tensor = torch.tensor(input_ids)
            attention_mask = None
        if position_ids is not None:
            position_ids_tensor = position_ids
        else:
            position_ids_tensor = None

        outputs = self.hyena.hyena(
            input_ids=input_ids,
            inputs_embeds=None,
            output_hidden_states=None,
            return_dict=None,
        )
        sequence_output = outputs[0]
        # logits = sequence_output
        # Pool and get logits
        logits = self.score(sequence_output)
        return logits, None
        
    def pool_hidden_states(self, hidden_states, sequence_length_dim=1):
        """Pools hidden states along sequence length dimension."""
        if self.pooling_strategy == "mean":  # Mean pooling along sequence length dimension
            return hidden_states.mean(dim=sequence_length_dim)
        if self.pooling_strategy == "max":  # Max pooling along sequence length dimension
            return hidden_states.max(dim=sequence_length_dim).values
        if self.pooling_strategy == "last":  # Use embedding of last token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[-1, ...]
        if self.pooling_strategy == "first":  # Use embedding of first token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[0, ...]

    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.
        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model