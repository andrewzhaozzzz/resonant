import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def compute_novelty(
    df, 
    embedding_path,
    start_idx,
    end_idx,
    prog_file,
    partial_file,
    dataset_path = False,
    date_col = "date",
    window_days = 14,
    save_every = 10000
):
    """
    For each post i in [start_idx, end_idx), find raw tauᵢ = max cosine(emb_i, emb_j) over all
    j in the prior window.  If there are no prior posts, tauᵢ = np.nan.  Store raw tauᵢ in taus[i].
    Checkpoint every `save_every` rows by writing a partial pickled DataFrame that includes
    the 'min_tau' column (which is just the raw tau up to that point).

    Parameters
    ----------
    df : string, DataFrame
        If dataset_path is True, then it should be a string indicating the path of the pickle file containing the original dataset.
        If dataset_path is False, then it should be the original dataset in DataFrame form.

    embedding_path : string
        The path directory of the memmap npy file containing calculated embeddings.

    start_idx : int
        The row index of the first document in dataset to consider in novelty calculation. Note that the first post might have np.nan min tau, but could affect following posts.

    end_idx : int
        The row index of the last document in dataset to consider in novelty calculation.

    prog_file : string
        A self-selected path directory for saving and reading progress file. Used to record the progress of novelty calculation, and resume from the latest progress.

    partial_file : string
        A self-selected path directory for saving min tau results. The file would be the original dataset plus an additional column with min tau information.

    dataset_path : string, optional
        If True, df should be a string indicating the path of the pickle file containing the original dataset.
        If False, df should be the original dataset in DataFrame form.

    date_col : string, optional
        The name of the dataset column that contains date information.

    window_days : int, optional
        The number of days to consider in the prior window. Documents with an earlier date than the date of current document - window_days would not be considered in novelty calculation.

    save_every : int, optional
        If specified, save the min tau information to the partial file when every save_every row has been processed.
    
    """
    if dataset_path == True:
        df = pd.read_pickle(df)
    date_raw = list(df[date_col])
    date_np = [np.datetime64(i, "D") for i in date_raw]
    dates = np.array(date_np)

    N = len(df)

    fp = np.memmap(embedding_path, dtype = "float32", mode = "r")
    fp = fp.reshape(N, -1)
    embs_t = torch.from_numpy(fp)
    
    prog = json.load(open(prog_file)) if os.path.exists(prog_file) else {}
    window_delta = np.timedelta64(window_days, 'D')
    left = 0
    right = start_idx
    taus = np.full(N, np.nan, dtype=np.float32)

    for i in tqdm(range(start_idx, end_idx), desc="novelty pass"):
        d0 = dates[i]

        # advance left pointer so that dates[left] >= d0 - window_delta
        min_date = d0 - window_delta
        while left < i and dates[left] < min_date:
            left += 1

        # advance right pointer so that dates[right] < d0
        while right < i and dates[right] < d0:
            right += 1

        # now all posts j in [left, right) are within the prior window
        if left >= right:
            tau_raw = np.nan
        else:
            idxs = np.arange(left, right)
            sims = (embs_t[idxs] @ embs_t[i]).cpu().numpy()
            tau_raw = sims.max()

        taus[i] = tau_raw

        # checkpoint JSON + partial .pkl
        if ((i + 1) % save_every == 0) or (i + 1 == end_idx):
            prog["novelty_idx"] = i + 1
            with open(prog_file, 'w') as f:
                json.dump(prog, f)

            part = df.iloc[:i + 1].copy()
            part["min_tau"] = taus[:i + 1]
            part.to_pickle(partial_file)
