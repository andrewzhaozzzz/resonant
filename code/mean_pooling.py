import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gc
import argparse
from transformers import AutoTokenizer, AutoModel, logging
from torch.cuda.amp import autocast

def mean_pooling(model_output, attention_mask, min_value = 1e-9):
    toks = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(toks.size()).float()
    pool_output = torch.sum(toks * mask, 1) / torch.clamp(mask.sum(1), min = min_value)
    return pool_output
