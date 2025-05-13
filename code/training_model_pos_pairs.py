import os
import random
import wandb
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from nltk.tokenize import sent_tokenize
import nltk
from tqdm import tqdm

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

tqdm.pandas()

# --- Helper: Create contrastive sentence pairs via intra-post sampling ---
def create_contrastive_pairs(df):
    """
    For each post with at least 2 sentences, randomly select a positive pair.
    """
    pairs = []
    for _, row in df.iterrows():
        text = row['post_text']
        sentences = sent_tokenize(text)
        if len(sentences) >= 2:
            # Randomly choose two distinct sentences from the same post
            s1, s2 = random.sample(sentences, 2)
            pairs.append({"sentence1": s1, "sentence2": s2})
    return pairs

# --- Custom contrastive model ---
class ContrastiveSentenceModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # Initialize accumulators for eval losses
        self._eval_loss_total = 0.0
        self._eval_batches = 0
    
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # Encode both sentences
        outputs1 = self.encoder(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.encoder(input_ids=input_ids2, attention_mask=attention_mask2)
        # Use the [CLS] token (first token) representation
        emb1 = outputs1.last_hidden_state[:, 0]
        emb2 = outputs2.last_hidden_state[:, 0]
        # Normalize embeddings to unit length
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        # Compute cosine similarity matrix
        logits = torch.matmul(emb1, emb2.t())
        # Scale by temperature (e.g., 0.05)
        temperature = 0.05
        logits = logits / temperature
        batch_size = input_ids1.size(0)
        # The diagonal elements are the positive pairs
        labels = torch.arange(batch_size, device=input_ids1.device)
        loss = F.cross_entropy(logits, labels)
        if not self.training:
            # Accumulate evaluation loss and count batches
            self._eval_loss_total += loss.item()
            self._eval_batches += 1
        return {"loss": loss, "logits": logits}
    
    # Override train() to log the average eval loss when switching back to training mode.
    def train(self, mode=True):
        # If we're switching to training (mode=True) and we were in eval mode,
        # log the accumulated average eval loss.
        if mode and not self.training and self._eval_batches > 0:
            avg_eval_loss = self._eval_loss_total / self._eval_batches
            wandb.log({"average_eval_loss": avg_eval_loss})
            # Reset accumulators
            self._eval_loss_total = 0.0
            self._eval_batches = 0
        return super().train(mode)
        
    # New method to allow saving the underlying transformer model
    def save_pretrained(self, save_directory):
        self.encoder.save_pretrained(save_directory)

# --- Tokenization function for sentence pairs ---
def tokenize_function(examples):
    tokenized_s1 = tokenizer(examples["sentence1"], padding="max_length", truncation=True, max_length=128)
    tokenized_s2 = tokenizer(examples["sentence2"], padding="max_length", truncation=True, max_length=128)
    return {
        "input_ids1": tokenized_s1["input_ids"],
        "attention_mask1": tokenized_s1["attention_mask"],
        "input_ids2": tokenized_s2["input_ids"],
        "attention_mask2": tokenized_s2["attention_mask"]
    }

# --- Main training function using intra-post sampling ---
def finetune_intra_post_model(pickle_path, col_name, save_name, num_paras=None, max_test_dataset=100000, eval_every=1000):
    if os.getenv("DEBUG"):
        print("Using tiny eval dataset.")
        max_test_dataset = 10000
        eval_every = 20

    # Load DataFrame and ensure missing posts are filled with an empty string
    df = pd.read_pickle(pickle_path)
    df[col_name] = df[col_name].fillna(" ")
    if num_paras is not None:
        df = df.iloc[:num_paras]
    
    print("Creating contrastive pairs from posts...")
    pairs = create_contrastive_pairs(df)
    print(f"Total contrastive pairs: {len(pairs)}")
    
    # Build a dataset from the pairs and shuffle it
    dataset = Dataset.from_dict({
        "sentence1": [p["sentence1"] for p in pairs],
        "sentence2": [p["sentence2"] for p in pairs]
    })
    dataset = dataset.shuffle(seed=42)
    
    # Split into training and test sets (80/20 split)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = dataset["test"]
    if len(test_dataset) > max_test_dataset:
        test_dataset = test_dataset.select(range(max_test_dataset))
    dataset = DatasetDict({"train": dataset["train"], "test": test_dataset})
    
    print("Initializing tokenizer and tokenizing dataset...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    # Save the tokenizer to a sensible directory for reuse
    tokenizer_save_path = os.path.join(model_locat_dir, "tokenizer")
    if os.path.exists(tokenizer_save_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        tokenizer.save_pretrained(tokenizer_save_path)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["sentence1", "sentence2"])
    
    # Initialize the contrastive model
    model = ContrastiveSentenceModel('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    model_dir = os.path.join(model_locat_dir, f"{save_name}_final")
    os.makedirs(model_dir, exist_ok=True)
    
    # Set training arguments: evaluation runs every 100 steps; logging every 10 steps.
    training_args = TrainingArguments(
        output_dir=model_dir,
        run_name="contrastive",
        evaluation_strategy="steps",
        eval_steps=eval_every,
        eval_on_start=True,
        learning_rate=1e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        save_steps=10000,  # adjust as needed
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
    
    print("Starting contrastive training with intra-post sampling...")
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
finetune_intra_post_model(pickle_path, 'post_text', 'who_leads_intra_post_model', num_paras=num_paras)