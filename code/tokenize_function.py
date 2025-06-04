import os
import re
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForMaskedLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling, 
    AutoTokenizer, 
    AdamW, 
    get_scheduler
)
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
import tokenize_function

def tokenize_function(examples, paragraph_col):
    return tokenizer(examples[paragraph_col], padding=True, truncation=True, return_tensors='pt')
