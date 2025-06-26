import os
import re
import pandas as pd
import torch
import numpy as np
import load_latest_valid_checkpoint

def stratified_sample_similarity(dataset, save_name, model_directory, dataset_path == True,
                                 similarity_thresholds = [0.3, 0.5, 0.7, 0.8, 0.9],
                                 batch_size = 700,
                                 samples_per_threshold = 10,
                                 variant = 0.1,
                                 messages = True):
    
    embedding_output_dir = os.path.join(model_directory, "embedding_output")
    # load embeddings (always on CPU)
    latest_doc_count = json.load(open(os.path.join(embedding_output_dir, f"{save_name}_progress.json")))["processed"]
    embeddings = np.memmap(os.path.join(embedding_output_dir, f"{save_name}_embeddings.npy"), dtype = "float32", mode = "r")

    embeddings = torch.from_numpy(embeddings)
    embeddings = torch.reshape(embeddings, (latest_doc_count, -1))
    embeddings = embeddings.cpu()
    n_docs = embeddings.size(0)

    # load original df (for context or further use)
    if dataset_path == True:
       df = pd.read_pickle(dataset)
    else:
       df = dataset
    df_subset = df.iloc[:latest_doc_count].reset_index(drop=True)

    for i in range(0, n_docs, batch_size):
        i_end = min(n_docs, i + batch_size)
        emb_i = embeddings[i:i_end]
        checkpoint_results = []

        for j in range(0, n_docs, batch_size):
            j_end = min(n_docs, j + batch_size)
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
            for b in range(len(similarity_thresholds) - 1):
                thres = similarity_thresholds[b]
                above_thres = (sims >= thres) & (sims <= (thres + variant))
                cand = above_thres.nonzero(as_tuple=True)[0]
                if cand.numel() == 0:
                    continue
                k = min(samples_per_threshold, cand.numel())
                choice = cand[torch.randperm(cand.numel())[:k]]
                for idx in choice.tolist():
                    checkpoint_results.append({
                        "threshold": thres,
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
        if messages == True:
           print(f"Saved checkpoint for rows {i} to {i_end} at {out_path}")

    if messages == True:
       print("Finished processing all blocks.")
      
