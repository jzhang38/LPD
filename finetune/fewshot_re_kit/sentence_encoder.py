import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import pdb
from torch import optim
from . import network
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification, PretrainedConfig

class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50, 
            pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, 
                word_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim, 
                pos_embedding_dim, hidden_size)
        self.word2id = word2id

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        return x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        pos1_in_index = min(self.max_length, pos_head[0])
        pos2_in_index = min(self.max_length, pos_tail[0])
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(indexed_tokens)] = 1

        return indexed_tokens, pos1, pos2, mask


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, path, mode,  ): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        if path is not None and path != "None":
            self.bert.load_state_dict(torch.load(path, map_location="cuda:0")["bert-base"])
            print("We load "+ path+" to train!")
        else:
            print("Path is None, We use Bert-base!")

        self.max_length = max_length
        self.mode = mode 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
        #state = outputs[0][:,0,:]
        tensor_range = torch.arange(inputs['word'].size()[0])
        h_state = outputs[0][tensor_range, inputs["pos1"]]
        t_state = outputs[0][tensor_range, inputs["pos2"]]
        state = torch.cat((h_state, t_state), -1)

        return state

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            if token != "[MASK]" or token !="[SEP]":
                token = token.lower()
            if self.mode == "CM": 
                if cur_pos == pos_head[0]:
                    tokens.append('[unused0]')
                    pos1_in_index = len(tokens)
                if cur_pos == pos_tail[0]:
                    tokens.append('[unused1]')
                    pos2_in_index = len(tokens)
                tokens += self.tokenizer.tokenize(token)
                if cur_pos == pos_head[-1]:
                    tokens.append('[unused2]')
                if cur_pos == pos_tail[-1]:
                    tokens.append('[unused3]')
                cur_pos += 1
            elif self.mode == "OC":
                if cur_pos == pos_head[0]:
                    tokens.append('[unused0]')
                    pos1_in_index = len(tokens)
                    tokens.append('[unused4]')
                    tokens.append('[unused2]')
                if cur_pos == pos_tail[0]:
                    tokens.append('[unused1]')
                    pos2_in_index = len(tokens)
                    tokens.append('[unused5]')
                    tokens.append('[unused3]')
                cur_pos += 1
                if cur_pos >= pos_head[0] and cur_pos <= pos_head[-1]:
                    continue
                if cur_pos >= pos_tail[0] and cur_pos <= pos_tail[-1]:
                    continue
                tokens += self.tokenizer.tokenize(token)
            elif self.mode == "OM":
                if cur_pos >= pos_head[0] and cur_pos <= pos_head[-1]:
                    if cur_pos == pos_head[0]:
                        tokens.append('[unused0]')
                        pos1_in_index = len(tokens)
                    tokens += self.tokenizer.tokenize(token)
                    if cur_pos == pos_head[-1]:
                        tokens.append('[unused2]')
                    cur_pos += 1
                if cur_pos >= pos_tail[0] and cur_pos <= pos_tail[-1]:
                    if cur_pos == pos_tail[0]:
                        tokens.append('[unused1]')
                        pos2_in_index = len(tokens)
                    tokens += self.tokenizer.tokenize(token)
                    if cur_pos == pos_tail[-1]:
                        tokens.append('[unused3]')
                    cur_pos += 1

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]
    
        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length
    
        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1
        
        if pos1_in_index == 0:
            pos1_in_index = 1
        if pos2_in_index == 0:
            pos2_in_index = 1
        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
    
        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask
    

    
    
    
    
    
    