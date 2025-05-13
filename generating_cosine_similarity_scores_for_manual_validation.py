#!/usr/bin/env python

import os
import re
import pandas as pd
import torch
import numpy as np

def load_latest_valid_checkpoint(embedding_output_dir, save_name):
    checkpoint_pattern = re.compile(f"{save_name}_embeddings_checkpoint_(\\d+)\\.pt")
    checkpoint_files = [
        f for f in os.listdir(embedding_output_dir)
        if checkpoint_pattern.match(f)
    ]
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in the output directory.")
    # Sort by document count descending
    checkpoint_files.sort(
        key=lambda x: int(checkpoint_pattern.match(x).group(1)),
        reverse=True
    )
    for ckpt in checkpoint_files:
        checkpoint_path = os.path.join(embedding_output_dir, ckpt)
        try:
            embeddings = torch.load(checkpoint_path, map_location="cpu")
            doc_count = int(checkpoint_pattern.match(ckpt).group(1))
            print(f"Loaded checkpoint {checkpoint_path} with {doc_count} documents.")
            return embeddings, doc_count, checkpoint_path
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
    raise RuntimeError("No valid checkpoint found.")

def stratified_sample_similarity():
    save_name = "who_leads_model"
    embedding_output_dir = os.path.join(model_directory, "embedding_output")

    # smaller blocks to avoid OOM
    block_size = 1000
    samples_per_bin = 10
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.01]

    # load embeddings (always on CPU)
    embeddings, latest_doc_count, _ = load_latest_valid_checkpoint(
        embedding_output_dir, save_name
    )
    embeddings = embeddings.cpu()
    n_docs = embeddings.size(0)

    # load original df (for context or further use)
    pickle_path = os.path.join(
        data_dir, "who_leads_who_follows", "cleaned_who_leads_df.pkl"
    )
    df = pd.read_pickle(pickle_path)
    df_subset = df.iloc[:latest_doc_count].reset_index(drop=True)

    for i in range(0, n_docs, block_size):
        i_end = min(n_docs, i + block_size)
        emb_i = embeddings[i:i_end]
        checkpoint_results = []

        for j in range(0, n_docs, block_size):
            j_end = min(n_docs, j + block_size)
            emb_j = embeddings[j:j_end]

            # compute cosine similarity (dot product since embeddings are normalized)
            sim_block = emb_i @ emb_j.T

            if i == j:
                # only upper triangle, exclude diagonal
                mask = torch.triu(torch.ones_like(sim_block), diagonal=1).bool()
                sims = sim_block[mask]
                rows, cols = mask.nonzero(as_tuple=True)
            else:
                sims = sim_block.flatten()
                rows = torch.arange(sim_block.size(0)).unsqueeze(1) \
                           .expand(-1, sim_block.size(1)).flatten()
                cols = torch.arange(sim_block.size(1)).repeat(sim_block.size(0))

            # sample from each similarity bin
            for b in range(len(bins) - 1):
                lo, hi = bins[b], bins[b+1]
                in_bin = (sims >= lo) & (sims < hi)
                cand = in_bin.nonzero(as_tuple=True)[0]
                if cand.numel() == 0:
                    continue
                k = min(samples_per_bin, cand.numel())
                choice = cand[torch.randperm(cand.numel())[:k]]
                for idx in choice.tolist():
                    checkpoint_results.append({
                        "doc_i": i + int(rows[idx].item()),
                        "doc_j": j + int(cols[idx].item()),
                        "cosine_similarity": float(sims[idx].item())
                    })

        # save this outer block’s results
        out_path = os.path.join(
            embedding_output_dir,
            f"stratified_similarity_checkpoint_{i}_{i_end}.csv"
        )
        pd.DataFrame(checkpoint_results).to_csv(out_path, index=False)
        print(f"Saved checkpoint for rows {i} to {i_end} at {out_path}")

    print("Finished processing all blocks.")

if __name__ == "__main__":
    try:
        filepath = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        filepath = os.getcwd()

    if "hbailey" in filepath:
        data_dir = "/home/export/hbailey/data/embedding_resonance"
        model_directory = (
            "/home/export/hbailey/models/embedding_resonance/"
            "who_leads_model_final/checkpoint-558242"
        )
    else:
        data_dir = "/home/hannah/github/embedding_resonance/data"
        model_directory = (
            "/home/hannah/github/embedding_resonance/"
            "model/who_leads_model_final/checkpoint-558242"
        )

    # for reproducible sampling
    np.random.seed(42)

    stratified_sample_similarity()
