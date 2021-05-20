# -*- coding: utf-8 -*-

import time
import os
import pdb

import torch
from torch.nn import DataParallel
import numpy as np

class Framework:
    def __init__(self, 
                 train_dataset=None, 
                 dev_dataset=None, 
                 test_dataset=None,
                 metric=None,
                 device=None,
                 opt=None):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.metric = metric
        self.device = device
        self.opt = opt
        
    def to_device(self, inputs):
        for k in inputs.keys():
            inputs[k] = inputs[k].to(self.device)
        return inputs
    
    def save_model(self, model, save_ckpt):
        checkpoint = {'opt': self.opt}
        if isinstance(model, DataParallel):
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['state_dict'] = model.state_dict()
        torch.save(checkpoint, save_ckpt)
    
    def load_model(self, model, load_ckpt):
        if os.path.isfile(load_ckpt):
            checkpoint = torch.load(load_ckpt)
            print(f"Successfully loaded checkpoint : {load_ckpt}")
        else:
            raise Exception(f"No checkpoint found at {load_ckpt}")
        
        load_state = checkpoint['state_dict']
        model_state = model.state_dict()
        for name, param in load_state.items():
            if name not in model_state:
                continue
            if param.shape != model_state[name].shape:
                print(f"In load model : {name} param shape not match")
                continue
            model_state[name].copy_(param)
    
    def evaluate(self, 
                 model, 
                 eval_epoch,
                 evalN, K, Q,
                 mode="dev",
                 load_ckpt=None):
        
        if load_ckpt is not None:
            print(f"loading checkpint {load_ckpt}")
            self.load_model(model, load_ckpt)
            print(f"checkpoint {load_ckpt} loaded")
        
        self.metric.reset()
        model.to(self.device)
        
        if mode == "dev":
            eval_dataset = self.dev_dataset
        elif mode == "test":
            eval_dataset = self.test_dataset
        elif mode == "train":
            eval_dataset = self.train_dataset
        
        model.eval()        
        for i in range(eval_epoch):
            support_set, query_set, id2label = next(eval_dataset)
            support_set, query_set = self.to_device(support_set), self.to_device(query_set)
            loss, logits, pred = model(support_set, query_set, evalN, K, Q)
            
            self.metric.update_state(pred, query_set['trigger_label'], id2label)
            
        return self.metric.result()
    
    def train(self, 
              model,
              trainN, evalN, K, Q,
              optimizer,
              scheduler,
              train_epoch,
              eval_epoch,
              eval_step,
              load_ckpt=None,
              save_ckpt=None):
        
        if load_ckpt is not None:
            print(f"loading checkpint {load_ckpt}")
            self.load_model(model, load_ckpt)
            print(f"checkpoint {load_ckpt} loaded")
        
        model.to(self.device)
        
        best_p = 0
        best_r = 0
        best_f1 = 0
        best_epoch = 0
        for epoch in range(train_epoch):    
            epoch_begin = time.time()
            
            # train
            model.train()
            support_set, query_set, id2label = next(self.train_dataset)
            support_set, query_set = self.to_device(support_set), self.to_device(query_set)
            loss, logits, pred = model(support_set, query_set, trainN, K, Q)
            loss = loss.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_begin
            remain_time_s = epoch_time * ((train_epoch-epoch) + np.ceil((train_epoch-epoch) / eval_step) * eval_epoch)
            remain_time_h = remain_time_s / 3600
            print(f"Epoch : {epoch}, loss : {loss:.4f}, time : {epoch_time:.4f}s, remain time : {remain_time_s:.4f}s ({remain_time_h:.2f}h)", end="\r")
            
            # evaluate
            if (epoch+1) % eval_step == 0:
                
                eval_time = time.time()
                p, r, f1 = self.evaluate(model, eval_epoch, evalN, K, Q)
                
                print()
                print(f"Evaluate result of epoch {epoch} - eval time : {time.time()-eval_time:.4f}s, P : {p:.6f}, R : {r:.6f}, F1 : {f1:.6f}")
                if f1 >= best_f1:
                    self.save_model(model, save_ckpt)
                    best_p = p
                    best_r = r
                    best_f1 = f1
                    best_epoch = epoch
                    print(f"New best performance in epoch {epoch} - P: {best_p:.6f}, R: {best_r:.6f}, F1: {best_f1:.6f}")
                else:
                    print(f"Current best performance - P: {best_p:.6f}, R: {best_r:.6f}, F1: {best_f1:.6f} in epoch {best_epoch}")
