
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

"""

Dataset for sampling arbitrary intervals from the human genome.

"""


# helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp

def random_mask(seq, mask_token_id, mask_prob=0.15):
    rand = torch.rand(seq.shape)
    
    mask = rand < mask_prob
    
    masked_seq = seq.clone()
    masked_seq[mask] = mask_token_id
    
    return (masked_seq, mask)

def bert_mask(seq, mask_token_id, pad_token_id, vocab_size, mask_prob=0.15, random_token_prob=0.1, unchanged_token_prob=0.1, special_token_ids=None):
    """
    Applies BERT masking strategy to a sequence of BPE tokens.

    Args:
        seq: Input sequence of BPE tokens (shape: [batch_size, seq_length]).
        mask_token_id: ID of the [MASK] token.
        pad_token_id: ID of the padding token.
        vocab_size: Size of the vocabulary.
        mask_prob: Probability of masking a token.
        random_token_prob: Probability of replacing a masked token with a random token.
        unchanged_token_prob: Probability of keeping a masked token unchanged.
    
    Returns:
        A tuple containing:
            - masked_seq: The masked sequence.
            - labels: The ground truth labels for the masked positions.
            - mask: The mask used to identify which tokens were masked.
    """
    # 避免遮蔽padding
    rand_mask = torch.rand(seq.shape) < mask_prob
    mask = (seq != pad_token_id) & (rand_mask)

    # 复制一份用于保存label
    labels = seq.clone()
    labels[~mask] = -100  # PyTorch忽略-100的label

    # 生成随机数矩阵
    rand = torch.rand(seq.shape)

    # 80% [MASK]
    indices_masked = mask & (rand < (1 - random_token_prob - unchanged_token_prob))
    seq[indices_masked] = mask_token_id

    # 10% random token
    indices_random = mask & (rand >= (1 - random_token_prob - unchanged_token_prob)) & (rand < (1 - unchanged_token_prob))
    # 生成随机token，排除特殊token
    random_tokens = torch.randint(0, vocab_size, seq.shape, dtype=torch.long)
    
    # 确保不会选到special tokens
    special_token_ids = torch.tensor(special_token_ids)
    while (torch.isin(random_tokens, special_token_ids)).any():
        random_tokens[torch.isin(random_tokens, special_token_ids)] = torch.randint(0, vocab_size, (random_tokens[torch.isin(random_tokens, special_token_ids)].shape[0],), dtype=torch.long)

    seq[indices_random] = random_tokens[indices_random]

    assert (mask == (seq != pad_token_id) & (rand_mask)).all()

    # 10% unchanged
    # 不需要修改seq，因为它已经保持不变

    return (seq, mask, labels)

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
        
        # # 检查文件是否存在
        # assert self.text_path.exists(), 'path to text file must exist'

        # 计算文件的行数作为数据集的长度
        self.length = self.calculate_length()

        self.bin_path = Path(f"{text_file}/{self.split}.bin")
        if not self.bin_path.exists():
            self.convert_dna_to_binary(self.text_path, self.bin_path)
        with open(str(self.bin_path)[:-4]+"_"+'padding_info.json', "r") as f:
            self.padding_info = json.load(f)
        self.read_binary_to_list_with_markers(self.bin_path) 

    def read_binary_to_list_with_markers(self, binary_file):
        """从带有行标记的二进制文件中读取DNA序列，逐行添加到列表中"""     
        self.lines = []
    
        with open(binary_file, 'rb') as file_in:
            data = file_in.read()  # 读取整个文件到内存

        marker = 0
        for i in range(0, len(self.padding_info)):
            row_length = self.padding_info[f"{i+1}"][0]
            binary_row = data[marker:marker+row_length]
            self.lines.append(binary_row)
            marker = marker+row_length
        return   

    def calculate_length(self):
        with open(self.text_path, 'r', encoding='ISO-8859-1') as file:
            return sum(1 for _ in file)
    
    def base_to_bits(self, base):
        """将DNA碱基转换为对应的2位二进制数"""
        base_to_binary = {'A': '00', 'T': '01', 'C': '10', 'G': '11'}
        return base_to_binary.get(base, '00')  # 如果找不到对应的碱基，使用'00'作为默认值

    def bits_to_base(self, bits):
        """将2位二进制数转换回DNA碱基"""
        binary_to_base = {'00': 'A', '01': 'T', '10': 'C', '11': 'G'}
        return binary_to_base.get(bits, 'N')  # 如果找不到对应的二进制数，使用'N'作为未知碱基

    def convert_dna_to_binary(self, input_file, output_file):     
        with open(input_file, 'r') as file_in, open(output_file, 'wb') as file_out:
            padding_info = {}
            for line_number, line in enumerate(file_in, 1):
                binary_line = ''.join(self.base_to_bits(base) for base in line.strip())
                padding = (8 - len(binary_line) % 8) % 8
                binary_line += '0' * padding
                padding_info[line_number] = []
                padding_info[line_number].append(math.ceil(len(line.strip())/4))
                padding_info[line_number].append(padding)

                binary_bytes = int(binary_line, 2).to_bytes(math.ceil(len(binary_line) / 8), byteorder='big')
                
                file_out.write(binary_bytes)
            with open(str(input_file)[:-4]+"_"+'padding_info.json', 'w') as json_file:
                json.dump(padding_info, json_file)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        binary_string = self.lines[idx]
        binary_string = format(int.from_bytes(binary_string, byteorder='big'), f'0{8*len(binary_string)}b')
        padding = self.padding_info[f"{idx+1}"][1]
        if padding != 0:
            binary_string = binary_string[:-padding]
        line = []
        # 每2位二进制数转换为一个DNA碱基
        for i in range(0, len(binary_string), 2):
            line.append(self.bits_to_base(binary_string[i:i+2]))
        line = "".join(line)

        tokens = self.tokenizer.encode(line, add_special_tokens=False, truncation=True, max_length=self.max_length)

        if self.add_eos:
            tokens.append(self.tokenizer.eos_token_id)

        # 处理长度超过 max_length 的情况
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # 填充到 pad_max_length
        if len(tokens) < self.pad_max_length:
            if self.pad_interval:
                tokens += [self.tokenizer.pad_token_id] * (self.pad_max_length - len(tokens))
            else:
                tokens = [self.tokenizer.pad_token_id] * (self.pad_max_length - len(tokens)) + tokens

        # convert to tensor
        seq = torch.LongTensor(tokens)

        if self.replace_N_token:
            # replace N token with a pad token, so we can ignore it in the loss
            seq[seq == self.tokenizer._vocab_str_to_int['N']] = self.tokenizer.pad_token_id

        data = seq.clone()  # remove eos
        target = seq.clone()  # offset by 1, includes eos
        # 获取特殊 token 的 ID
        special_token_ids = self.tokenizer.all_special_ids
        if not self.use_tokenizer:
            seq = seq-7
            mask = (seq >= 4) | (seq < 0)
            seq[mask] = 4
            data = seq.clone()  # remove eos
            target = seq.clone()  # offset by 1, includes eos
            return bert_mask(data, 4, 5, 5, special_token_ids=[4]), target         
        
        if self.objective=="stdmlm":
            return bert_mask(data, self.tokenizer.mask_token_id, self.tokenizer.pad_token_id, self.tokenizer.vocab_size, special_token_ids=special_token_ids), target
        else:
            return random_mask(data, self.tokenizer.mask_token_id), target