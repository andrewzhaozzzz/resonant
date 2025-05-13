#!/usr/bin/env python
import os
import re
import glob
import pandas as pd
import torch
import numpy as np

def load_all_valid_checkpoints(embedding_output_dir, save_name):
    """
    Look for all files in embedding_output_dir that match the pattern
    "{save_name}_simcse_embeddings_checkpoint_<doc_count>.pt" and return a list
    of tuples: (embeddings, doc_count, checkpoint_path).
    """
    checkpoint_pattern = re.compile(f"{save_name}_simcse_embeddings_checkpoint_(\\d+)\\.pt")
    checkpoint_files = [f for f in os.listdir(embedding_output_dir) if checkpoint_pattern.match(f)]
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in the output directory.")
    # Sort in ascending order (earlier checkpoints first)
    checkpoint_files.sort(key=lambda x: int(checkpoint_pattern.match(x).group(1)))
    checkpoints = []
    for ckpt in checkpoint_files:
        checkpoint_path = os.path.join(embedding_output_dir, ckpt)
        try:
            embeddings = torch.load(checkpoint_path)
            doc_count = int(checkpoint_pattern.match(ckpt).group(1))
            print(f"Loaded checkpoint {checkpoint_path} with {doc_count} documents.")
            checkpoints.append((embeddings, doc_count, checkpoint_path))
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
    if not checkpoints:
        raise RuntimeError("No valid checkpoint found.")
    return checkpoints

def process_checkpoint(embeddings, doc_count, bins, samples_per_bin, block_size):
    """
    Process embeddings for one checkpoint.
    Compute pairwise cosine similarity in blocks and perform stratified sampling.
    Returns a list of dictionaries with the sampled pairs.
    """
    results = []
    n_docs = embeddings.size(0)
    for i in range(0, n_docs, block_size):
        i_end = min(n_docs, i + block_size)
        emb_i = embeddings[i:i_end]
        for j in range(i, n_docs, block_size):
            j_end = min(n_docs, j + block_size)
            emb_j = embeddings[j:j_end]
            # Compute cosine similarity (dot product, since embeddings are normalized)
            sim_block = torch.mm(emb_i, emb_j.T)
            sim_block_np = sim_block.cpu().numpy()
            
            if i == j:
                # Use only the upper triangular part for the same block.
                triu_idx = np.triu_indices(i_end - i, k=1)
                rows = triu_idx[0]
                cols = triu_idx[1]
                sims = sim_block_np[rows, cols]
            else:
                rows, cols = np.indices(sim_block_np.shape)
                rows = rows.flatten()
                cols = cols.flatten()
                sims = sim_block_np.flatten()
            
            # For each bin, sample a fixed number of pairs.
            for b in range(len(bins) - 1):
                lower = bins[b]
                upper = bins[b+1]
                mask = (sims >= lower) & (sims < upper)
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    num_to_sample = min(samples_per_bin, len(indices))
                    sample_indices = np.random.choice(indices, size=num_to_sample, replace=False)
                    for idx in sample_indices:
                        doc_i = i + int(rows[idx])
                        doc_j = j + int(cols[idx])
                        sim_val = sims[idx]
                        results.append({
                            "doc_i": doc_i,
                            "doc_j": doc_j,
                            "cosine_similarity": sim_val,
                            "checkpoint_doc_count": doc_count
                        })
    return results

def stratified_sample_similarity_all():
    save_name = "who_leads_simcse"  # Save name used when saving the embeddings.
    # Assume the embeddings are stored in "simcse_embedding_output" within the selected model_directory.
    embedding_output_dir = os.path.join(model_directory, "simcse_embedding_output")
    
    # Parameters for stratification.
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.01]  # Using 1.01 to include similarity 1.0
    samples_per_bin = 10
    block_size = 10000
    
    # Load all valid checkpoint files from the embedding_output_dir.
    checkpoints = load_all_valid_checkpoints(embedding_output_dir, save_name)
    all_results = []
    for embeddings, doc_count, ckpt_path in checkpoints:
        print(f"Processing checkpoint: {ckpt_path}")
        results = process_checkpoint(embeddings, doc_count, bins, samples_per_bin, block_size)
        # Save an intermediate CSV for this checkpoint.
        intermediate_csv = os.path.join(embedding_output_dir, f"stratified_similarity_{doc_count}.csv")
        pd.DataFrame(results).to_csv(intermediate_csv, index=False)
        print(f"Saved intermediate CSV for checkpoint {doc_count} at: {intermediate_csv}")
        all_results.extend(results)
        
    # Combine all results and save as one aggregated CSV file.
    df_all = pd.DataFrame(all_results)
    output_csv = os.path.join(embedding_output_dir, "stratified_similarity_all_checkpoints.csv")
    df_all.to_csv(output_csv, index=False)
    print(f"Saved aggregated stratified similarity CSV at: {output_csv}")

if __name__ == "__main__":
    try:
        filepath = os.path.dirname(os.path.realpath(__file__))
    except Exception:
        filepath = os.getcwd()
    
    # Set file paths based on the environment.
    if "hbailey" in filepath:
        data_dir = "/home/export/hbailey/data/embedding_resonance"
        base_model_dir = "/home/export/hbailey/models/embedding_resonance/who_leads_simcse_doc_model_final"
        tokenizer_path = "/home/export/hbailey/models/embedding_resonance/tokenizer"
    else:
        data_dir = "/home/hannah/github/embedding_resonance/data"
        base_model_dir = "/home/hannah/github/embedding_resonance/model/who_leads_simcse_doc_model_final"
        tokenizer_path = "/home/hannah/github/embedding_resonance/model/tokenizer"
    
    # Find checkpoint directories within the base model directory.
    checkpoint_dirs = glob.glob(os.path.join(base_model_dir, "checkpoint-*"))
    checkpoint_dirs.sort(key=os.path.getmtime, reverse=True)
    
    model_directory = None
    # Iterate over checkpoint directories (newest first) and choose the first with a simcse_embedding_output subfolder.
    for ckpt in checkpoint_dirs:
        embedding_output_path = os.path.join(ckpt, "simcse_embedding_output")
        if os.path.isdir(embedding_output_path):
            model_directory = ckpt
            print(f"Found checkpoint folder with embeddings: {ckpt}")
            break
    
    if model_directory is None:
        model_directory = base_model_dir
        print(f"No checkpoint folder with 'simcse_embedding_output' found; using model from {model_directory}")
    
    np.random.seed(42)
    stratified_sample_similarity_all()

