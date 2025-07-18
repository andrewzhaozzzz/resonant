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
import wandb

wandb.init(project="embedding_resonance", name="finetune_embedding_resonance_model")

# set to 'None' to use all paragraphs
num_paras = None
criterion = nn.MSELoss()

filepath = str(os.path.dirname(os.path.realpath(__file__)))
if "hbailey" in filepath:
    data_dir = "/home/export/hbailey/data/embedding_resonance"
    model_locat_dir = "/home/export/hbailey/models/embedding_resonance"
else:
    data_dir = "/Users/HannahBailey/Documents/GitHub/embedding_resonance/data"
    model_locat_dir = "/Users/HannahBailey/Documents/GitHub/embedding_resonance/model"

def finetune_model(pickle_path, col_name, save_name, num_paras=None):
    # 1. Confirm you’re loading the right file
    print(f"Loading DataFrame from: {pickle_path}")

    df = pd.read_pickle(pickle_path)

    # 2. Spot-check that URLs/RTs are gone
    samples = df[col_name].dropna().sample(5, random_state=0).tolist()
    for i, text in enumerate(samples, 1):
        assert 'http' not in text,   "Found a URL in cleaned data!"
        assert not text.lower().startswith('rt @'), "Found an RT in cleaned data!"
        print(f"Sample {i}: {text}")
    paragraphs = df[col_name].tolist()

    # replace None with ' '
    paragraphs = [' ' if para is None else para for para in paragraphs]

    if num_paras is not None:
        paragraphs = paragraphs[:num_paras]
        df = df.iloc[:num_paras]
    
    dataset = Dataset.from_dict({"labels": list(range(len(paragraphs))), "text": paragraphs})
    dataset = DatasetDict({"train": dataset, "test": dataset})
    print("Dataset ready.")

    # initialize both the model and the tokenizer
    model = AutoModelForMaskedLM.from_pretrained(
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        num_labels=1
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    )
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    print("Model and tokenizer ready.")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, return_tensors='pt')
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    print("Data ready.")

    model_dir = os.path.join(model_locat_dir, f'{save_name}_final')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    print("Starting training...")

    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        save_steps=100000,
        per_device_train_batch_size=16,  # smaller batch size
        gradient_accumulation_steps=4,     # accumulate gradients over multiple steps
        report_to="wandb",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
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
    print("Training complete.")

    # Save the model and tokenizer.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("Saved Model.")

# First, train (or resume training) on the who leads who follows dataset.
who_leads_folder = os.path.join(data_dir, 'who_leads_who_follows')
pickle_path = os.path.join(who_leads_folder, 'cleaned_who_leads_df.pkl')
finetune_model(pickle_path, 'post_text', 'who_leads_model', num_paras)


# Second, train the model on the text as policy dataset

# load new data
# tracing_folder = os.path.join(data_dir, 'text_as_policy')
# pickle_path2 = os.path.join(tracing_folder, 'cleaned_tracing_df.pkl')

# finetune_model(pickle_path2, 'bill_text', 'tracing_model', num_paras)