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


def compute_embeddings(
    pickle_path,
    col_name,
    model_dir,
    tokenizer_dir,
    save_name,
    shard,
    num_shards,
    batch_size=32,
    max_length=128,
):
    # 1) load texts and compute shard indices
    df = pd.read_pickle(pickle_path)
    total_docs = len(df)
    size       = total_docs // num_shards
    start      = shard * size
    end        = total_docs if shard == num_shards-1 else (shard+1)*size

    # 2) prepare paths
    out_dir    = os.path.join(model_dir, "embedding_output")
    os.makedirs(out_dir, exist_ok=True)
    prog_file  = os.path.join(out_dir, f"{save_name}_shard{shard}_progress.json")
    npy_path   = os.path.join(out_dir, f"{save_name}_shard{shard}_embeddings.npy")
    full_npy   = os.path.join(out_dir, f"{save_name}_embeddings.npy")

    # 3) auto-slice if needed
    if os.path.exists(prog_file) and not os.path.exists(npy_path) and os.path.exists(full_npy):
        mm_full = np.memmap(full_npy, dtype="float32", mode="r")
        D       = mm_full.size // total_docs
        mm_full = mm_full.reshape(total_docs, D)
        shard_sz = end - start
        mm_shard = np.memmap(npy_path, dtype="float32", mode="w+", shape=(shard_sz, D))
        mm_shard[:] = mm_full[start:end]
        del mm_shard, mm_full

    # 4) slice DataFrame for this shard
    df   = df.iloc[start:end]
    docs = df[col_name].fillna(" ").tolist()
    total = len(docs)
    print(f"Shard {shard}/{num_shards}: docs {start}–{end}")
    print(f"→ {total} documents to embed")

    # 5) load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model     = AutoModel.from_pretrained(model_dir)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    model.eval()

    # 6) resume progress
    if os.path.exists(prog_file):
        processed = json.load(open(prog_file))["processed"]
        print(f"Resuming from {processed}")
    else:
        processed = 0

    # 7) open memmap
    hidden_size = (model.module.config.hidden_size
                   if hasattr(model, "module")
                   else model.config.hidden_size)
    mode = "r+" if os.path.exists(npy_path) else "w+"
    fp   = np.memmap(npy_path, dtype="float32", mode=mode, shape=(total, hidden_size))

    # 8) embedding loop
    for i in range(processed, total, batch_size):
        j   = min(total, i + batch_size)
        enc = tokenizer(docs[i:j],
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        emb = mean_pooling(out, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1).cpu().numpy()
        fp[i:j, :] = emb

        # cleanup
        del enc, out, emb
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        processed = j
        with open(prog_file, "w") as f:
            json.dump({"processed": processed}, f)
        print(f" → wrote docs {i}–{j} ({processed}/{total})")

    del fp
    print(f"✅ Done shard {shard}")
