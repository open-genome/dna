import os
import functools
import torch
from random import randrange, random
import numpy as np
from pathlib import Path
import pandas as pd

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
def coin_flip():
    return random() > 0.5
def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp
class GUEDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        max_length,
        dataset_name=None, #needed for multi datasets
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
        use_tokenizer=True, # tailored for diffusion models
        task="binary"
    ):

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask
        self.task = task
        self.use_tokenizer = use_tokenizer
        split=split.lower()
        ##data/GUE/... :train test dev.csv
        if split == "valid" or split == "val":# change "val" to "dev"
            split = "dev"
        
        data_path=Path(dest_path) / dataset_name / f"{split}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data = pd.read_csv(data_path, sep=",", header=None)
        # print(self.data)
        #refers to https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/train.py
        if(len(self.data.columns) == 2):
            # data is in the format of [text, label]
            print(f'{dataset_name}: Sequence classification')
            self.data.columns = ['seq', 'target']
        elif(len(self.data.columns) == 3):
            # data is in the format of [text1,text2 label]
            # merge first 2
            print(f'{dataset_name}: Sequence pair classification')
            self.data['seq'] = self.data[0] + self.data[1]
            self.data = self.data.drop(columns=[0, 1])
            self.data.columns = ['seq', 'target']
        else:
             raise ValueError("Data format not supported.")
        #remove header
        self.data = self.data[1:]
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        sample = self.data.iloc[idx]
        x = sample['seq']
        # print(x)
        #y to float32
        y = int(sample['target'])#convert y from str to 
        # y is a single number
        # convert to tensor
        target = torch.Tensor([y])
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)
        seq = self.tokenizer(x,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids
        seq_ids = torch.LongTensor(seq_ids)
        if not self.use_tokenizer:
            seq_ids = seq_ids-7
            mask = (seq_ids >= 4) | (seq_ids < 0)
            seq_ids[mask] = 4

        # need to wrap in list
        if len(target.shape)==2 and target.shape[0]==1:
            target = target.squeeze(0)
        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target

# from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
# if __name__ == "__main__":
#     print('ok')
#     a=GUEDataset(split='test',max_length=512,
#                  dataset_name='splice/reconstructed',
#                  dest_path='/mnt/nas/share2/home/by/hyena-dna/data/GUE',
#                  tokenizer=None,tokenizer_name=None,use_padding=True,add_eos=False,rc_aug=False,return_augs=False,return_mask=False,use_tokenizer=True)
#     print(a.__len__())
#     print(a.__getitem__(0))
#     print(a.__getitem__(1))