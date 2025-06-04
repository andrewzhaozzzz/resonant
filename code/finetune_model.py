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

def finetune_model(prepared_dataset, paragraph_col, output_name, output_dir, prepared_dataset_path = False, messages = True, 
                   model = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                   tokenizer = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 
                   num_labels = 1, mlm_prob = 0.15, evaluation_strategy="epoch", learning_rate=1e-5, num_train_epochs=1,
                   weight_decay=0.01, save_steps=100000, per_device_train_batch_size=16, gradient_accumulation_steps=4):
    # Load the dataset prepared for finetuning
    if prepared_dataset_path == True:
        dataset = pd.read_pickle(prepared_dataset)
    else:
        dataset = prepared_dataset

    # Initialize both the model and the tokenizer
    model = AutoModelForMaskedLM.from_pretrained(model, num_labels = num_labels)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer, mlm_probability = mlm_prob
    )

    if messages == True:
        print("Model and tokenizer ready.")
    
    tokenized_datasets = dataset.map(tokenize_function.tokenize_function, fn_kwargs = {"paragraph_col": paragraph_col}, batched = True)
    tokenized_datasets = tokenized_datasets.remove_columns([paragraph_col])
    tokenized_datasets.set_format("torch")

    if messages == True:
        print("Data ready.")

    model_dir = os.path.join(output_dir, f'{output_name}_final')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    if messages == True:
        print("Starting training...")

    training_args = TrainingArguments(
        output_dir = model_dir,
        evaluation_strategy = evaluation_strategy,
        learning_rate = learning_rate,
        num_train_epochs = num_train_epochs,
        weight_decay = weight_decay,
        save_steps = save_steps,
        per_device_train_batch_size = per_device_train_batch_size,  # smaller batch size
        gradient_accumulation_steps = gradient_accumulation_steps,     # accumulate gradients over multiple steps
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_datasets["train"],
        eval_dataset = tokenized_datasets["test"],
        data_collator = data_collator,
    )

    # Check for an existing checkpoint to resume from.
    resume_checkpoint = None
    if os.path.exists(model_dir):
        checkpoints = [
            os.path.join(model_dir, d)
            for d in os.listdir(model_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            # Pick the checkpoint with the highest number (assuming the checkpoint directories are named like "checkpoint-200000")
            resume_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
            print(f"Resuming training from checkpoint: {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    if messages == True:
        print("Training complete.")

    # Save the model and tokenizer.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    if messages == True:
        print("Saved Model.")

