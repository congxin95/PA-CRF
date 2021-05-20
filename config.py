# -*- coding: utf-8 -*-

seed = None

dataset = "fewevent"
encoder_path = 'bert-base-uncased'

# model settings 
model = "pa_crf"

sample_num = 5

# encoder settings
encoder = "bert"
feature_size = 768
max_length = 128

trainN = 5
evalN = 5
K = 5
Q = 1

batch_size = 1
num_workers = 8

dropout = 0.1
optimizer = "adamw"
learning_rate = 1e-5
warmup_step = 100
scheduler_step = 1000

train_epoch = 20000
eval_epoch = 1000
eval_step = 500
test_epoch = 3000

ckpt_dir = "checkpoint/"
load_ckpt = None
save_ckpt = None

device = None
test=False

notes=""
