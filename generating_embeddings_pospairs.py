#!/usr/bin/env python
import os
import re
import glob
import gc
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # token-level embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def find_latest_checkpoint(base_dir):
    # Look for subdirectories that match "checkpoint-<number>"
    checkpoints = [d for d in os.listdir(base_dir)
                   if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))]
    if checkpoints:
        latest = max(checkpoints, key=lambda d: int(re.search(r'checkpoint-(\d+)', d).group(1)))
        return os.path.join(base_dir, latest)
    return base_dir

def compute_embeddings(pickle_path, col_name, base_model_directory, tokenizer_path, save_name,
                       batch_size=128, checkpoint_interval=10000):
    df = pd.read_pickle(pickle_path)
    documents = df[col_name].fillna(" ").tolist()
    print(f"Computing embeddings for {len(documents)} documents")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # Find the most up-to-date checkpoint in the finetuned model directory
    model_directory = find_latest_checkpoint(base_model_directory)
    print(f"Loading model from checkpoint: {model_directory}")
    model = AutoModel.from_pretrained(model_directory)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    output_dir = os.path.join(base_model_directory, "embedding_output")
    os.makedirs(output_dir, exist_ok=True)

    # Check for existing embedding checkpoints
    checkpoint_pattern = os.path.join(output_dir, f"{save_name}_embeddings_checkpoint_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        processed_docs = int(latest_checkpoint.split("_")[-1].split(".")[0])
        print(f"Resuming from {processed_docs} documents processed.")
        all_embeddings = [torch.load(latest_checkpoint)]
    else:
        processed_docs = 0
        all_embeddings = []

    total_docs = len(documents)
    current_count = processed_docs
    for i in range(processed_docs, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            model_output = model(**encoded)
        embeddings = mean_pooling(model_output, encoded['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
        current_count += len(batch)
        print(f"Processed {current_count} documents")

        del encoded, model_output, embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if current_count % checkpoint_interval == 0 or current_count == total_docs:
            checkpoint_embeddings = torch.cat(all_embeddings, dim=0)
            checkpoint_path = os.path.join(output_dir, f"{save_name}_embeddings_checkpoint_{current_count}.pt")
            torch.save(checkpoint_embeddings, checkpoint_path)
            print(f"Saved checkpoint at {current_count} documents to {checkpoint_path}")
            all_embeddings = []  # Clear accumulated embeddings to free memory

if __name__ == "__main__":
    try:
        filepath = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        filepath = os.getcwd()
    if "hbailey" in filepath:
        data_dir = "/home/export/hbailey/data/embedding_resonance"
        base_model_directory = "/home/export/hbailey/models/embedding_resonance/who_leads_intra_post_model_final"
        tokenizer_path = "/home/export/hbailey/models/embedding_resonance/tokenizer"
    else:
        data_dir = "/home/hannah/github/embedding_resonance/data"
        base_model_directory = "/home/hannah/github/embedding_resonance/model/who_leads_intra_post_model_final"
        tokenizer_path = "/home/hannah/github/embedding_resonance/model/tokenizer"

    who_leads_folder = os.path.join(data_dir, 'who_leads_who_follows')
    pickle_path = os.path.join(who_leads_folder, 'cleaned_who_leads_df.pkl')
    compute_embeddings(pickle_path, 'post_text', base_model_directory, tokenizer_path, "who_leads_intra_post_model")
