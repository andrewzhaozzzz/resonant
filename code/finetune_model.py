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
    get_scheduler
)
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
import resonant.prepare_dataset

def finetune_model(prepared_dataset, paragraph_col, output_name, output_dir, prepared_dataset_path = False, messages = True,
                   model = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                   tokenizer = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                   num_labels = 1, mlm_prob = 0.15, **training_args):
    """Fine-tuning of a specified LLM that would be used to generate embeddings for text.

    Parameters
    ----------
    prepared_dataset : string, DatasetDict
        If prepared_dataset_path is True, then it should be a string indicating the path of the pickle file containing the original dataset.
        If prepared_dataset_path is False, then it should be a DatasetDict containing a prepared dataset generated with the function prepare_dataset.

    paragraph_col : string
        The name of the column corresponding to text or paragraphs we aim to analyze.

    output_name : string
        A self-selected name for the output model and tokenizer.

    output_dir: string
        A self-selected path directory for saving the output model and tokenizer.
        
    prepared_dataset_path : bool, optional
        If set to True, then prepared_dataset would read in a pickle file containing the original dataset without preparation.

    messages : bool, optional
        If set to False, then messages that indicate running progress would not be printed out.

    model : string, optional
        The Large Language Model used for fine-tuning. Should be one of the Hugging Face AutoModels.

    tokenizer : string, optional
        The tokenizer used for embedding generation. Should be one of the Hugging Face AutoTokenizers.

    num_labels : int, optional

    mlm_prob : scalar, optional

    training_args : dict

    """
    # Load the dataset prepared for finetuning
    if prepared_dataset_path == True:
        dataset = pd.read_pickle(prepared_dataset)
        dataset = prepare_dataset.prepare_dataset(dataset)
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

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, return_tensors='pt')
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    if messages == True:
        print("Data ready.")

    model_dir = os.path.join(output_dir, f'{output_name}_final')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    if messages == True:
        print("Starting training...")

    training_args = TrainingArguments(
            **training_args
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

