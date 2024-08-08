
from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np
from mmap import mmap, ACCESS_READ

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

class DNABERTSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,  # split 参数可能不再需要
        text_file,
        max_length,
        seq_name='seq_a',
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

        # 读取文本文件
        assert seq_name in ['seq_a', 'seq_b'], 'seq_name must be either "seq_a" or "seq_b"'
        if split=="train":
            split="train_2m"
        elif split=="dev" or split=="test" or split=="val":
            split="val_48k"
        else:
            split="debug_train"
        # text_path = Path(text_file+"/"+split+"_"+seq_name+".txt")
        self.text_path = Path(text_file+"/"+split+".txt")
        # 检查文件是否存在
        assert self.text_path.exists(), 'path to text file must exist'

        # 创建文件的内存映射
        self.file = open(self.text_path, 'rb')  # 以二进制模式打开文件
        self.mmap_file = mmap(self.file.fileno(), 0, access=ACCESS_READ)

        # 计算文件的行数作为数据集的长度
        self.length = sum(1 for _ in open(self.text_path, 'r', encoding='ISO-8859-1'))


    def __del__(self):
        # 关闭内存映射和文件
        self.mmap_file.close()
        self.file.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 定位到行的开始
        line_start = self.mmap_file.find(b'\n', idx) + 1
        # 定位到行的结束
        line_end = self.mmap_file.find(b'\n', line_start)
        # 读取行并解码
        line = self.mmap_file[line_start:line_end].decode('ISO-8859-1').strip()

        # 使用 tokenizer 处理文本
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