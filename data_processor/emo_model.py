# -*- coding: utf-8 -*-
"""Automatically annotating EmpatheticDialogues with BERT classifier
"""

import math
import csv
import numpy as np
from transformers import RobertaTokenizer
import tqdm
import pickle
import os

import datetime
import time
import faiss
from collections import Counter

import sys

# Add EmpatheticIntents to sys.path for imports (must be before importing EmpatheticIntents)
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EMPATHETIC_INTENTS_PATH = os.path.join(_BASE_DIR, "models", "pre_trained_models", "EmpatheticIntents")
if _EMPATHETIC_INTENTS_PATH not in sys.path:
    sys.path.insert(0, _EMPATHETIC_INTENTS_PATH)

from EmpatheticIntents.model import *
from EmpatheticIntents.utilities import *
from EmpatheticIntents.optimize import CustomSchedule
from EmpatheticIntents.create_datasets import create_datasets

# Safe optional import (legacy), avoid breaking when module path is absent
try:
    from data_processor.utils.data.loader import *  # type: ignore # noqa: F401,F403
except Exception:
    pass

import tensorflow as tf
 
#gpu_device = tf.config.list_physical_devices('GPU')
#print(f"gpu_device: {gpu_device}")
#tf.config.set_visible_devices(gpu_device[int(config.device_id)], 'GPU')
#print("List device:", tf.config.list_physical_devices('GPU'))

#gpus = tf.config.list_physical_devices('GPU')
#print(f"gpus: {gpus}")
#if gpus:
#    tf.config.experimental.set_memory_growth(gpus[0], True)
#    tf.config.set_visible_devices(gpus[0], 'GPU')

#os.environ["CUDA_VISIBLE DEVICES"]=f"{config.device_id}"
#gpu_options =tf.GPUOptions(allow_growth=True)
#sess =tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


tf.compat.v1.enable_eager_execution()

## After eager execution is enabled, operations are executed as they are
## defined and Tensor objects hold concrete values, which can be accessed as
## numpy.ndarray`s through the numpy() method.
#os.environ["CUDA_VISOBLE_DEVICES"] = config.device_id
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 

num_layers = 12
d_model = 768
num_heads = 12
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 514
num_emotions = 41  # Number of emotion categories

# Use local tokenizer to avoid online downloads
# _BASE_DIR is already defined above
_LOCAL_TOKENIZER = os.path.join(_BASE_DIR, "models", "pre_trained_models", "roberta_large_goEmotions")
tokenizer = RobertaTokenizer.from_pretrained(_LOCAL_TOKENIZER)
vocab_size = tokenizer.vocab_size
max_length = 100  # Maximum number of tokens
buffer_size = 100000
batch_size = 1
num_epochs = 10
peak_lr = 2e-5
total_steps = 7000
warmup_steps = 700
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

# Use absolute checkpoint path to avoid relative path issues
checkpoint_path = os.path.abspath(os.path.join(_BASE_DIR, "models", "pre_trained_models", "EmpatheticIntents", "checkpoints"))


def _configure_gpu(device_id: str = "0"):
    """Optionally pin to a specific GPU and enable memory growth."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            idx = min(max(int(device_id), 0), len(gpus) - 1)
            tf.config.set_visible_devices(gpus[idx], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[idx], True)
            print(f"Using GPU device {idx}: {gpus[idx].name}")
        else:
            print("No GPU available, fallback to CPU.")
    except Exception as e:
        print(f"GPU configuration warning: {e}. Fallback to default device placement.")

SOS_ID = tokenizer.encode('<s>')[0]
EOS_ID = tokenizer.encode('</s>')[0]

class EmoModel():
    def __init__(self, device_id="0"):
        self.device_id = device_id
        _configure_gpu(self.device_id)
        logical_gpus = tf.config.list_logical_devices('GPU')
        if logical_gpus:
            self.compute_device = logical_gpus[0].name  # e.g., '/device:GPU:0'
        else:
            self.compute_device = "/CPU:0"
        print(f"Compute device set to: {self.compute_device}")
        self.emobert = EmoBERT(self.device_id, num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
                    layer_norm_eps, max_position_embed, vocab_size, num_emotions)
        
        build_model(self.emobert, max_length, vocab_size)
        
        learning_rate = CustomSchedule(peak_lr, total_steps, warmup_steps)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
                    epsilon = adam_epsilon)
        #train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        
        # Define the checkpoint manager.
        ckpt = tf.train.Checkpoint(model = self.emobert, optimizer = optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)
        
        # Restore the checkpoint at epoch 5 (checkpoint with highest accuracy on test set)
        print("ckpt_manager.checkpoints:", ckpt_manager.checkpoints)
        ckpt.restore(ckpt_manager.checkpoints[4])
        print('Checkpoint at epoch 5 restored!!')

    def gen_emb(self, uttrs, mode):

        uttr_ids = np.ones((len(uttrs), max_length), dtype = np.int32)
        for i, u in enumerate(uttrs):
            # Clean input: remove/replace surrogate characters that cannot be UTF-8 encoded to avoid UnicodeEncodeError from tokenizer.encode
            safe_u = u.encode("utf-8", "replace").decode("utf-8")
            u_ids = [SOS_ID] + tokenizer.encode(safe_u)[:(max_length-2)] + [EOS_ID]
            uttr_ids[i, :len(u_ids)] = u_ids
    
        inp = tf.constant(uttr_ids)
        with tf.device(self.compute_device):
            enc_padding_mask = create_masks(inp)
            pred = self.emobert(inp, False, enc_padding_mask, mode)
        #print(f"pred shape: {pred.shape}, {pred}")
    
        return pred.numpy()
