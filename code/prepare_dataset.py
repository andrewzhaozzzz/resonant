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

def prepare_dataset(pickle_path, paragraph_col, save_name, spot_check = False, sample_num = 5, random_state = 0, num_paras = None, messages = False):
    # Confirm you’re loading the right file
    df = pd.read_pickle(pickle_path)
    if messages = True:
        print(f"Loading DataFrame from: {pickle_path}")

    # Spot-check that URLs/RTs are gone
    if spot_check = True:
        samples = df[paragraph_col].dropna().sample(sample_num, random_state).tolist()
        for i, text in enumerate(samples, 1):
        assert 'http' not in text,   "Found a URL in cleaned data!"
        assert not text.lower().startswith('rt @'), "Found an RT in cleaned data!"
        print(f"Sample {i}: {text}")

    # Select the paragraph column
    paragraphs = df[paragraph_col].tolist()

    # replace None with ' '
    paragraphs = [' ' if para is None else para for para in paragraphs]

    # If specified number of paragraphs needed, slice the paragraph column
    if num_paras is not None:
        paragraphs = paragraphs[:num_paras]
        df = df.iloc[:num_paras]

    # Create the dataset prepared for training
    dataset = Dataset.from_dict({"labels": list(range(len(paragraphs))), "text": paragraphs})
    dataset = DatasetDict({"train": dataset, "test": dataset})

    if messages = True:
        print("Dataset ready.")

    return dataset
