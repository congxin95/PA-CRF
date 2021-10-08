# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import BertTokenizer, BertModel

class BertEncoder(nn.Module):
    def __init__(self, bert_path, max_length):
        super(BertEncoder, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        
        self.max_length = max_length
    
    def forward(self, input_ids, **bert_args):
        output = self.bert(input_ids, **bert_args)
        output = output[0]
        return output
    
    def tokenize(self, instance, label2id):
        max_length = self.max_length
        
        raw_tokens = instance['tokens']
        raw_label = instance['trigger_label']
        raw_B_mask = instance['B-mask']
        raw_I_mask = instance['I-mask']
        
        # token -> index
        tokens = ['[CLS]']
        label = ['O']
        B_mask = [0]
        I_mask = [0]
        for i, token in enumerate(raw_tokens):
            tokenize_result = self.tokenizer.tokenize(token)
            tokens += tokenize_result
            
            if len(tokenize_result) > 1:
                label += [raw_label[i]]
                B_mask += [raw_B_mask[i]]
                I_mask += [raw_I_mask[i]]
                
                if raw_label[i][0] == "B":
                    tmp_label = "I" + raw_label[i][1:]
                    label += [tmp_label] * (len(tokenize_result) - 1)
                    B_mask += [0] * (len(tokenize_result) - 1)
                    I_mask += [1] * (len(tokenize_result) - 1)
                else:
                    label += [raw_label[i]] * (len(tokenize_result) - 1)
                    B_mask += [raw_B_mask[i]] * (len(tokenize_result) - 1)
                    I_mask += [raw_I_mask[i]] * (len(tokenize_result) - 1)
            else:
                label += [raw_label[i]] * len(tokenize_result)
                B_mask += [raw_B_mask[i]] * len(tokenize_result)
                I_mask += [raw_I_mask[i]] * len(tokenize_result)
        
        # att mask
        att_mask = torch.zeros(max_length)
        att_mask[:len(tokens)] = 1
        
        # padding
        while len(tokens) < self.max_length:
            tokens.append('[PAD]')
            label.append('O')
            B_mask.append(0)
            I_mask.append(0)
        tokens = tokens[:max_length]
        label = label[:max_length]
        B_mask = B_mask[:max_length]
        I_mask = I_mask[:max_length]
        
        tokens[-1] = '[SEP]'
        
        # to ids
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = torch.tensor(token_ids).long()
        
        label_ids = list(map(lambda x: label2id[x], label))
        label_ids = torch.tensor(label_ids).long()
        
        B_mask_ids = torch.tensor(B_mask)
        I_mask_ids = torch.tensor(I_mask)
        
        return token_ids, label_ids, B_mask_ids, I_mask_ids, att_mask

