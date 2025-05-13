import os
import glob
import random
import wandb
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset, DatasetDict
from nltk.tokenize import sent_tokenize
import nltk
from tqdm import tqdm

# Ensure nltk resources are available 
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

tqdm.pandas()

# --- Helper: Extract documents from posts ---
def extract_documents(df, col_name):
    """
    Extract documents (posts) from the specified column.
    """
    return df[col_name].fillna(" ").tolist()

# --- Custom SimCSE Model ---
class SimCSEModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self._keys_to_ignore_on_save = {}  # Added attribute to satisfy Trainer requirements (citation here)
        self._eval_loss_total = 0.0
        self._eval_batches = 0
    
    def forward(self, input_ids, attention_mask):
        out1 = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        out2 = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        emb1 = F.normalize(out1, p=2, dim=1)
        emb2 = F.normalize(out2, p=2, dim=1)
        cos_sim = F.cosine_similarity(emb1, emb2, dim=1)
        loss = (1 - cos_sim).mean()

        if not self.training:
            self._eval_loss_total += loss.item()
            self._eval_batches += 1

        return {"loss": loss, "logits": None}
    
    def train(self, mode=True):
        if mode and not self.training and self._eval_batches > 0:
            avg_eval_loss = self._eval_loss_total / self._eval_batches
            wandb.log({"average_eval_loss": avg_eval_loss})
            self._eval_loss_total = 0.0
            self._eval_batches = 0
        return super().train(mode)
    
    # Save the full encoder (including its config)
    def save_pretrained(self, save_directory):
        self.encoder.save_pretrained(save_directory)

# --- Tokenization function for documents ---
def tokenize_function(examples):
    tokenized = tokenizer(examples["document"], padding="max_length", truncation=True, max_length=256)
    return tokenized

# --- Callback to save a full checkpoint with model config ---
class FullModelCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_folder)
        return control

# --- Main training function using SimCSE on documents ---
def finetune_simcse_document_model(pickle_path, col_name, save_name, num_paras=None, max_test_dataset=100000, eval_every=1000):
    if os.getenv("DEBUG"):
        print("Using tiny eval dataset.")
        max_test_dataset = 100
        eval_every = 2

    df = pd.read_pickle(pickle_path)
    if num_paras is not None:
        df = df.iloc[:num_paras]
    
    print("Extracting documents from posts...")
    documents = extract_documents(df, col_name)
    print(f"Total documents extracted: {len(documents)}")
    
    dataset = Dataset.from_dict({"document": documents})
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = dataset["test"]
    if len(test_dataset) > max_test_dataset:
        test_dataset = test_dataset.select(range(max_test_dataset))
    dataset = DatasetDict({"train": dataset["train"], "test": test_dataset})
    
    print("Initializing tokenizer and tokenizing dataset...")
    global tokenizer
    tokenizer_save_path = os.path.join(model_locat_dir, "tokenizer")
    if os.path.exists(tokenizer_save_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        os.makedirs(tokenizer_save_path, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_save_path)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["document"])
    
    model = SimCSEModel('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    model.to(device)

    print("\nDropout layers in the model:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            print(f"{name}: dropout probability = {module.p}")
        
    model_dir = os.path.join(model_locat_dir, f"{save_name}_final")
    os.makedirs(model_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=model_dir,
        run_name="simcse",
        evaluation_strategy="steps",
        eval_steps=eval_every,
        learning_rate=1e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        save_steps=2000,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        report_to="wandb",
        logging_steps=1
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )
    
    trainer.add_callback(FullModelCheckpointCallback)
    
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found; starting training from scratch.")
        trainer.train()
    
    print("Training complete. Saving model and tokenizer...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("Saved Model and Tokenizer.")

# --- Directory setup (adjust paths as needed) ---
filepath = str(os.path.dirname(os.path.realpath(__file__)))
if "hbailey" in filepath:
    data_dir = "/home/export/hbailey/data/embedding_resonance"
    model_locat_dir = "/home/export/hbailey/models/embedding_resonance"
else:
    data_dir = "/home/hannah/github/embedding_resonance/data"
    model_locat_dir = "/home/hannah/github/embedding_resonance/model"

# --- Run training ---
who_leads_folder = os.path.join(data_dir, 'who_leads_who_follows')
pickle_path = os.path.join(who_leads_folder, 'cleaned_who_leads_df.pkl')
if os.getenv("DEBUG"):
    print("Using tiny dataset.")
    num_paras = 100000
else:
    num_paras = None
finetune_simcse_document_model(pickle_path, 'post_text', 'who_leads_simcse_doc_model', num_paras=num_paras)
