import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gc
import argparse
from transformers import AutoTokenizer, AutoModel, logging
from torch.cuda.amp import autocast
import mean_pooling

def compute_embeddings(
    dataset,
    paragraph_col,
    model_dir,
    tokenizer_dir,
    save_name,
    batch_num,
    data_pickle_path = True,
    pooling_method = "mean_pooling",
    batch_size = 32,
    max_length = 128,
    messages = True
):
    # Load dataset
    if data_pickle_path = True:
        df = pd.read_pickle(dataset)
    else:
        df = dataset

    # Prepare paths for saving files
    out_dir    = os.path.join(model_dir, "embedding_output")
    os.makedirs(out_dir, exist_ok=True)
    prog_file  = os.path.join(out_dir, f"{save_name}_progress.json")
    npy_path   = os.path.join(out_dir, f"{save_name}_embeddings.npy")

    # Prepare dataset for embedding computation
    docs = df[paragraph_col].fillna(" ").tolist()
    total = len(docs)
    if messages = True:
        print(f"→ {total} documents to embed")

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model     = AutoModel.from_pretrained(model_dir)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        if messages == True:
            print(f"Using {torch.cuda.device_count()} GPUs")
    model.eval()

    # Resume if already has progress
    if os.path.exists(prog_file):
        processed = json.load(open(prog_file))["processed"]
        print(f"Resuming from {processed}")
    else:
        processed = 0

    # Open memmap
    hidden_size = (model.module.config.hidden_size
                   if hasattr(model, "module")
                   else model.config.hidden_size)
    mode = "r+" if os.path.exists(npy_path) else "w+"
    fp   = np.memmap(npy_path, dtype="float32", mode=mode, shape=(total, hidden_size))

    # Embedding loop
    for i in range(processed, total, batch_size):
        j   = min(total, i + batch_size)
        enc = tokenizer(docs[i:j],
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        if pooling_method == "mean_pooling":
            emb = mean_pooling.mean_pooling(out, enc["attention_mask"])
        
        emb = F.normalize(emb, p=2, dim=1).cpu().numpy()
        fp[i:j, :] = emb

        # Cleanup
        del enc, out, emb
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        processed = j
        with open(prog_file, "w") as f:
            json.dump({"processed": processed}, f)
        if messages == True:
            print(f" → wrote docs {i}–{j} ({processed}/{total})")

    del fp
