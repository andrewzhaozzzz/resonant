#!/usr/bin/env python
import os
import glob
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import gc

def compute_simcse_embeddings(pickle_path, col_name, model_directory, tokenizer_directory, save_name,
                              batch_size=128, checkpoint_interval=10000):
    df = pd.read_pickle(pickle_path)
    documents = df[col_name].fillna(" ").tolist()
    print(f"Computing embeddings for {len(documents)} documents using the SimCSE model")
    
    # Load tokenizer from the saved tokenizer directory.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory, local_files_only=True)
    # Load the model from the checkpoint directory.
    model = AutoModel.from_pretrained(model_directory, local_files_only=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    # Create an output folder inside the model directory.
    output_dir = os.path.join(model_directory, "simcse_embedding_output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check for an existing embedding checkpoint.
    checkpoint_pattern = os.path.join(output_dir, f"{save_name}_simcse_embeddings_checkpoint_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    processed_docs = 0
    all_embeddings = []
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True)
        print("Found the following embedding checkpoints:")
        for ckpt in checkpoint_files:
            print(f"  {ckpt}")
        for idx, ckpt in enumerate(checkpoint_files):
            try:
                print(f"Loading checkpoint (attempt {idx+1}): {ckpt}")
                all_embeddings = [torch.load(ckpt)]
                processed_docs = int(ckpt.split("_")[-1].split(".")[0])
                print(f"Loaded checkpoint with {processed_docs} documents processed")
                break
            except Exception as e:
                print(f"Failed to load {ckpt}: {e}. Removing file.")
                os.remove(ckpt)
    else:
        print("No embedding checkpoint found; starting from scratch.")

    total_docs = len(documents)
    current_count = processed_docs

    # Process documents in batches.
    for i in range(processed_docs, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        encoded = {key: val.to(device) for key, val in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        # Extract the [CLS] token embedding and normalize it.
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
        current_count += len(batch)
        print(f"Processed batch; total processed: {current_count}")

        del encoded, outputs, embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save checkpoint and then clear the accumulated embeddings to free memory.
        if current_count % checkpoint_interval == 0 or current_count == total_docs:
            checkpoint_embeddings = torch.cat(all_embeddings, dim=0)
            checkpoint_path = os.path.join(output_dir, f"{save_name}_simcse_embeddings_checkpoint_{current_count}.pt")
            torch.save(checkpoint_embeddings, checkpoint_path)
            print(f"Saved checkpoint at {current_count} documents")
            all_embeddings = []  # Clear saved embeddings

    print("All document embeddings computed.")

if __name__ == "__main__":
    try:
        filepath = os.path.dirname(os.path.realpath(__file__))
    except Exception:
        filepath = os.getcwd()

    if "hbailey" in filepath:
        data_dir = "/home/export/hbailey/data/embedding_resonance"
        # Use the actual folder name from your file structure.
        base_model_dir = "/home/export/hbailey/models/embedding_resonance/who_leads_simcse_doc_model_final"
        # Tokenizer was saved separately.
        tokenizer_dir = "/home/export/hbailey/models/embedding_resonance/tokenizer"
    else:
        data_dir = "/home/hannah/github/embedding_resonance/data"
        base_model_dir = "/home/hannah/github/embedding_resonance/model/who_leads_simcse_doc_model_final"
        tokenizer_dir = "/home/hannah/github/embedding_resonance/model/tokenizer"

    # Find the latest checkpoint if available.
    checkpoint_dirs = glob.glob(os.path.join(base_model_dir, "checkpoint-*"))
    if checkpoint_dirs:
        latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
        model_directory = latest_checkpoint
        print(f"Using latest checkpoint: {latest_checkpoint}")
    else:
        # Fallback: use the base model directory.
        model_directory = base_model_dir
        print(f"No checkpoint found; using model from {model_directory}")

    who_leads_folder = os.path.join(data_dir, 'who_leads_who_follows')
    pickle_path = os.path.join(who_leads_folder, 'cleaned_who_leads_df.pkl')
    compute_simcse_embeddings(pickle_path, 'post_text', model_directory, tokenizer_dir, "who_leads_simcse")
