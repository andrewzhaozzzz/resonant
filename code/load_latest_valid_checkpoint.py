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
