import torch
from typing import Optional
from transformers import AutoModel
from transformers.models.bert.configuration_bert import BertConfig
import torch.nn as nn

config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
DNABERT2 = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

class DNABERT2CustomModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.d_model = config.hidden_size
        self.bert = DNABERT2

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

        outputs = self.bert(input_ids=input_ids_tensor, position_ids=position_ids_tensor, attention_mask=attention_mask)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        return prediction_scores, None
        
    

    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.
        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model