import os
import re
import pandas as pd
import torch
import numpy as np

def load_latest_valid_checkpoint(embedding_output_dir, save_name):
    checkpoint_pattern = re.compile(f"{save_name}_embeddings_checkpoint_(\\d+)\\.pt")
    checkpoint_files = [f for f in os.listdir(embedding_output_dir) if checkpoint_pattern.match(f)]
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in the output directory.")
    # Sort in descending order so that the most recent is first.
    checkpoint_files.sort(key=lambda x: int(checkpoint_pattern.match(x).group(1)), reverse=True)
    for ckpt in checkpoint_files:
        checkpoint_path = os.path.join(embedding_output_dir, ckpt)
        try:
            embeddings = torch.load(checkpoint_path)
            doc_count = int(checkpoint_pattern.match(ckpt).group(1))
            print(f"Loaded checkpoint {checkpoint_path} with {doc_count} documents.")
            return embeddings, doc_count, checkpoint_path
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
    raise RuntimeError("No valid checkpoint found.")

def stratified_sample_similarity():
    save_name = "who_leads_model"
    embedding_output_dir = os.path.join(model_directory, "embedding_output")
    
    # Define cosine similarity bins; adjust these edges as needed.
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.01]  # using 1.01 to capture exact 1.0 values
    samples_per_bin = 10  # number of pairs to sample per bin in each block

    # Load the latest valid checkpoint of embeddings.
    embeddings, latest_doc_count, checkpoint_path = load_latest_valid_checkpoint(embedding_output_dir, save_name)

    # Load the original dataframe and restrict to the processed documents.
    pickle_path = os.path.join(data_dir, 'who_leads_who_follows', 'cleaned_who_leads_df.pkl')
    df = pd.read_pickle(pickle_path)
    df_subset = df.iloc[:latest_doc_count].reset_index(drop=True)

    n_docs = embeddings.size(0)
    block_size = 10000  # adjust based on available memory

    # Process blocks and save checkpoint files periodically (per outer block).
    for i in range(0, n_docs, block_size):
        i_end = min(n_docs, i + block_size)
        checkpoint_results = []  # accumulate results for current outer block
        emb_i = embeddings[i:i_end]
        for j in range(i, n_docs, block_size):
            j_end = min(n_docs, j + block_size)
            emb_j = embeddings[j:j_end]
            # Compute cosine similarity (dot product, since embeddings are normalized)
            sim_block = torch.mm(emb_i, emb_j.T)
            sim_block_np = sim_block.cpu().numpy()
            
            # For diagonal blocks, use only upper triangular indices to avoid duplicates.
            if i == j:
                triu_idx = np.triu_indices(i_end - i, k=1)
                rows = triu_idx[0]
                cols = triu_idx[1]
                sims = sim_block_np[rows, cols]
            else:
                rows, cols = np.indices(sim_block_np.shape)
                rows = rows.flatten()
                cols = cols.flatten()
                sims = sim_block_np.flatten()
            
            # Convert to numpy arrays for sampling.
            sims = np.array(sims)
            rows = np.array(rows)
            cols = np.array(cols)
            
            # For each similarity bin, sample a fixed number of pairs.
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
                        checkpoint_results.append({
                            "doc_i": doc_i,
                            "doc_j": doc_j,
                            "cosine_similarity": sim_val
                        })
        # Save checkpoint file for the current outer block.
        checkpoint_output_path = os.path.join(
            embedding_output_dir,
            f"stratified_similarity_checkpoint_{i}_{i_end}.csv"
        )
        df_checkpoint = pd.DataFrame(checkpoint_results)
        df_checkpoint.to_csv(checkpoint_output_path, index=False)
        print(f"Saved checkpoint for rows {i} to {i_end} at {checkpoint_output_path}")

    print("Finished processing all blocks.")

if __name__ == "__main__":
    try:
        filepath = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        filepath = os.getcwd()
    if "hbailey" in filepath:
        data_dir = "/home/export/hbailey/data/embedding_resonance"
        model_directory = "/home/export/hbailey/models/embedding_resonance/who_leads_model_final/checkpoint-100000"
        tokenizer_path = "/home/export/hbailey/models/embedding_resonance/tokenizer"
    else:
        data_dir = "/home/hannah/github/embedding_resonance/data"
        model_directory = "/home/hannah/github/embedding_resonance/model/who_leads_model_final/checkpoint-100000"
        tokenizer_path = "/home/hannah/github/embedding_resonance/model/tokenizer"
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    stratified_sample_similarity()
