# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

class ProtoDot(nn.Module):
    
    def __init__(self, 
                 encoder, 
                 feature_size, 
                 max_len, 
                 dropout):
        super(ProtoDot, self).__init__()
        
        self.feature_size = feature_size
        self.max_len = max_len
        
        self.encoder = encoder
        self.encoder = nn.DataParallel(self.encoder)
        
        self.cost = nn.CrossEntropyLoss(reduction="none")
        
        self.dropout_rate = dropout
        self.drop = nn.Dropout(self.dropout_rate)

    def forward(self, support_set, query_set, N, K, Q):
        # encode
        support_emb = self.encoder(support_set['tokens'])   # B*N*K, max_len, feature_size
        query_emb = self.encoder(query_set['tokens'])       # B*N*K, max_len, feature_size
        
        # dropout
        support_emb = self.drop(support_emb)                # B*N*K, max_len, feature_size
        query_emb = self.drop(query_emb)                    # B*N*K, max_len, feature_size
        
        support_emb = support_emb.view(-1, N, K, self.max_len, self.feature_size)   # B, N, K, max_len, feature_size
        query_emb = query_emb.view(-1, N*Q, self.max_len, self.feature_size)        # B, N*Q, max_len, feature_size

        B_mask = support_set['B-mask'].view(-1, N, K, self.max_len)     # B, N, K, max_len
        I_mask = support_set['I-mask'].view(-1, N, K, self.max_len)     # B, N, K, max_len

        # prototype 
        prototype = self.proto(support_emb, B_mask, I_mask)         # B, 2*N+1, feature_size
        
        # classification
        logits = self.similarity(prototype, query_emb)              # B, N*Q, max_len, 2*N+1
        _, pred = torch.max(logits.view(-1, logits.shape[-1]), 1)   # B*N*Q*max_len
        
        outputs = (logits, pred)
        
        # loss
        if query_set['trigger_label'] is not None:
            loss = self.loss(logits, query_set['trigger_label'])
            outputs = (loss,) + outputs
        
        return outputs

    def similarity(self, prototype, query):
        '''
        inputs:
            prototype: B, 2*N+1, feature_size
            query: B, N*Q, max_len, feature_size
        outputs:
            sim: B, N*Q, max_len, 2*N+1
        '''
        tag_num = prototype.shape[1]
        query_num = query.shape[1]
        
        query = query.unsqueeze(-2)                                 # B, N*Q, max_len, 1, feature_size
        query = query.expand(-1, -1, -1, tag_num, -1)               # B, N*Q, max_len, 2*N+1, feature_size
        
        prototype = prototype.unsqueeze(1)                          # B, 1, 2*N+1, feature_size
        prototype = prototype.unsqueeze(2)                          # B, 1, 1, 2*N+1, feature_size
        prototype = prototype.expand(-1, query_num, self.max_len, -1, -1) # B, N*Q, max_len, 2*N+1, feature_size
        
        sim = (prototype * query).sum(-1)                           # B, N*Q, max_len, 2*N+1

        return sim

    def proto(self, support_emb, B_mask, I_mask):
        '''
        input:
            support_emb : B, N, K, max_len, feature_size
            B_mask : B, N, K, max_len
            I_mask : B, N, K, max_len
        output:
            prototype : B, 2*N+1, feature_size # (class_num -> 2N + 1)
        '''
        B, N, K, _, _ = support_emb.shape
        prototype = torch.empty(B, 2*N+1, self.feature_size).to(support_emb)
        
        B_mask = B_mask.unsqueeze(-1)
        B_mask = B_mask.expand(-1, -1, -1, -1, self.feature_size)
        B_mask = B_mask.to(support_emb) # B, N, K, max_len, feature_size
        I_mask = I_mask.unsqueeze(-1)
        I_mask = I_mask.expand(-1, -1, -1, -1, self.feature_size)
        I_mask = I_mask.to(support_emb) # B, N, K, max_len, feature_size

        for i in range(B):
            O_mask = torch.ones_like(B_mask[i]).to(B_mask)    # N, K, max_len, feature_size
            O_mask -= B_mask[i] + I_mask[i]
            for j in range(N):
                sum_B_fea = (support_emb[i, j] * B_mask[i, j]).view(-1, self.feature_size).sum(0)
                num_B_fea = B_mask[i, j].sum() / self.feature_size + 1e-8
                prototype[i, 2*j+1] = sum_B_fea / num_B_fea
                
                sum_I_fea = (support_emb[i, j] * I_mask[i, j]).view(-1, self.feature_size).sum(0)
                num_I_fea = I_mask[i, j].sum() / self.feature_size + 1e-8
                prototype[i, 2*j+2] = sum_I_fea / num_I_fea
            
            sum_O_fea = (support_emb[i] * O_mask).reshape(-1, self.feature_size).sum(0)
            num_O_fea = O_mask.sum() / self.feature_size + 1e-8
            prototype[i, 0] = sum_O_fea / num_O_fea
        
        return prototype
    
    def loss(self, logits, label):
        logits = logits.view(-1, logits.shape[-1])
        label = label.view(-1)
        
        loss_weight = torch.ones_like(label).float()
        loss = self.cost(logits, label)
        loss = (loss_weight * loss).mean()
        return loss

