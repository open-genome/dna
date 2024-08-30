from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np
from mmap import mmap, ACCESS_READ
import itertools
import math
import json
from transformers import AutoTokenizer
import tracemalloc
import os
from torch.utils.data import DataLoader

# Helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# Augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        else:
            rev_comp += base
    return rev_comp

def random_mask(seq, mask_token_id, mask_prob=0.15):
    rand = torch.rand(seq.shape)
    mask = rand < mask_prob
    masked_seq = seq.clone()
    masked_seq[mask] = mask_token_id
    return masked_seq, mask

def bert_mask(seq, mask_token_id, pad_token_id, vocab_size, mask_prob=0.15, random_token_prob=0.1, unchanged_token_prob=0.1, special_token_ids=None):
    # Avoid masking padding
    rand_mask = torch.rand(seq.shape) < mask_prob
    mask = (seq != pad_token_id) & rand_mask

    labels = seq.clone()
    labels[~mask] = -100  # PyTorch ignores -100 labels

    rand = torch.rand(seq.shape)

    # 80% [MASK]
    indices_masked = mask & (rand < (1 - random_token_prob - unchanged_token_prob))
    seq[indices_masked] = mask_token_id

    # 10% random token
    indices_random = mask & (rand >= (1 - random_token_prob - unchanged_token_prob)) & (rand < (1 - unchanged_token_prob))
    random_tokens = torch.randint(0, vocab_size, seq.shape, dtype=torch.long)

    special_token_ids = torch.tensor(special_token_ids)
    while torch.isin(random_tokens, special_token_ids).any():
        random_tokens[torch.isin(random_tokens, special_token_ids)] = torch.randint(0, vocab_size, (random_tokens[torch.isin(random_tokens, special_token_ids)].shape[0],), dtype=torch.long)

    seq[indices_random] = random_tokens[indices_random]

    assert (mask == (seq != pad_token_id) & rand_mask).all()

    return seq, mask, labels

class DNABERT2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        text_file,
        max_length,
        pad_max_length=None,
        tokenizer_name=None,
        add_eos=False,
        replace_N_token=False,
        pad_interval=False,
        use_tokenizer=True,
        tokenizer=None,
        return_augs=False,
        objective="stdmlm",
    ):
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token
        self.pad_interval = pad_interval
        self.use_tokenizer = use_tokenizer
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.objective = objective

        self.split = split if split != "val" and split != "test" else "dev"
        self.text_path = Path(f"{text_file}/{self.split}.txt")
        
        self.bin_path = Path(f"{text_file}/{self.split}.bin")
        if not self.bin_path.exists():
            self.convert_dna_to_binary(self.text_path, self.bin_path)
        with open(str(self.bin_path)[:-4]+"_"+'padding_info.json', "r") as f:
            self.padding_info = json.load(f)
        
        self.length = self.calculate_length()

    def calculate_length(self):
        return len(self.padding_info)

    def base_to_bits(self, base):
        base_to_binary = {'A': '00', 'T': '01', 'C': '10', 'G': '11'}
        return base_to_binary.get(base, '00')

    def bits_to_base(self, bits):
        binary_to_base = {'00': 'A', '01': 'T', '10': 'C', '11': 'G'}
        return binary_to_base.get(bits, 'N')

    def convert_dna_to_binary(self, input_file, output_file):     
        with open(input_file, 'r') as file_in, open(output_file, 'wb') as file_out:
            padding_info = {}
            for line_number, line in enumerate(file_in, 1):
                binary_line = ''.join(self.base_to_bits(base) for base in line.strip())
                padding = (8 - len(binary_line) % 8) % 8
                binary_line += '0' * padding
                padding_info[line_number] = [math.ceil(len(line.strip()) / 4), padding]

                binary_bytes = int(binary_line, 2).to_bytes(math.ceil(len(binary_line) / 8), byteorder='big')
                
                file_out.write(binary_bytes)
            with open(str(input_file)[:-4]+"_"+'padding_info.json', 'w') as json_file:
                json.dump(padding_info, json_file)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with open(self.bin_path, 'rb') as file_in:
            padding_info = self.padding_info[f"{idx+1}"]
            marker = sum([self.padding_info[f"{i+1}"][0] for i in range(idx)])
            file_in.seek(marker)
            binary_row = file_in.read(padding_info[0])
        
        binary_string = format(int.from_bytes(binary_row, byteorder='big'), f'0{8*len(binary_row)}b')
        padding = padding_info[1]
        if padding != 0:
            binary_string = binary_string[:-padding]
        line = [self.bits_to_base(binary_string[i:i+2]) for i in range(0, len(binary_string), 2)]
        line = "".join(line)

        tokens = self.tokenizer.encode(line, add_special_tokens=False, truncation=True, max_length=self.max_length)

        if self.add_eos:
            tokens.append(self.tokenizer.eos_token_id)

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        if len(tokens) < self.pad_max_length:
            if self.pad_interval:
                tokens += [self.tokenizer.pad_token_id] * (self.pad_max_length - len(tokens))
            else:
                tokens = [self.tokenizer.pad_token_id] * (self.pad_max_length - len(tokens)) + tokens

        seq = torch.LongTensor(tokens)

        if self.replace_N_token:
            seq[seq == self.tokenizer._vocab_str_to_int['N']] = self.tokenizer.pad_token_id

        data = seq.clone()
        target = seq.clone()
        special_token_ids = self.tokenizer.all_special_ids

        if not self.use_tokenizer:
            seq = seq - 7
            mask = (seq >= 4) | (seq < 0)
            seq[mask] = 4
            data = seq.clone()
            target = seq.clone()
            return bert_mask(data, 4, 5, 5, special_token_ids=[4]), target         
        
        if self.objective == "stdmlm":
            return bert_mask(data, self.tokenizer.mask_token_id, self.tokenizer.pad_token_id, self.tokenizer.vocab_size, special_token_ids=special_token_ids), target
        else:
            return random_mask(data, self.tokenizer.mask_token_id), target

if __name__ == "__main__":
    root = os.getenv('PROJECT_ROOT', '/home/xt86/daniel/genomics/dna')

    tokenizer = AutoTokenizer.from_pretrained(f'{root}/DNABERT-2-117M')
    dataset = DNABERT2Dataset(
        split="train",
        text_file="/home/xt86/daniel/genomics/hyena-dna/data/dnabert2",
        max_length=1024,
        pad_max_length=1024,
        tokenizer_name="char",
        add_eos=True,
        replace_N_token=True,
        pad_interval=False,
        use_tokenizer=True,
        tokenizer=tokenizer,
        return_augs=False,
        objective="stdmlm"
    )

    def profile_memory_usage(dataset, batch_size=32, num_workers=0, shuffle=False, sampler=None, drop_last=False, pin_memory=False):
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, sampler=sampler, drop_last=drop_last, pin_memory=pin_memory)
        tracemalloc.start()
        for i, _ in enumerate(dataloader):
            current, peak = tracemalloc.get_traced_memory()
            print(f"Batch {i+1}: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        tracemalloc.stop()

    profile_memory_usage(dataset, batch_size=128, num_workers=1, shuffle=False, sampler=None, drop_last=False, pin_memory=False)