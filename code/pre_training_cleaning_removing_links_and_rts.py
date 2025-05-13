#!/usr/bin/env python
import os
import re
import pandas as pd

# Improved cleaning function

def clean_post_text(text):
    if pd.isnull(text):
        return text

    # 1) remove any 'RT @username' occurrences (with optional leading dot, colon/spaces)
    text = re.sub(r"\bRT\s+@\.?[A-Za-z0-9_]+[:]*\s*", "", text, flags=re.IGNORECASE)
    # 1a) remove any 'RT @ ' leftovers (no username)
    text = re.sub(r"\bRT\s+@\s*", "", text, flags=re.IGNORECASE)

    # 2) drop any URLs with protocol-like patterns, including incomplete ones (e.g., http://, thttp://)
    text = re.sub(r"\b\w*tp[s]?://\S*", "", text, flags=re.IGNORECASE)

    # 3) remove URLs starting with www.
    text = re.sub(r"\bwww\.\S*", "", text, flags=re.IGNORECASE)

    # 3a) remove any http… fragments (catch broken or truncated links)
    text = re.sub(r"\bhttp\S*", "", text, flags=re.IGNORECASE)

    # 4) remove stray .@username or @username at start
    text = re.sub(r"^\.?@[\w_]+[:\s]*", "", text)

    # 5) trim leading ASCII quotes or backticks
    text = re.sub(r"^[\"'`]+", "", text)

    return text.strip()

if __name__ == "__main__":
    filepath = os.path.dirname(os.path.realpath(__file__))
    if "hbailey" in filepath:
        data_dir = "/home/export/hbailey/data/embedding_resonance"
    else:
        data_dir = "/Users/HannahBailey/Documents/GitHub/embedding_resonance/data"
    
    who_leads_folder = os.path.join(data_dir, 'who_leads_who_follows')
    pickle_path      = os.path.join(who_leads_folder, 'cleaned_who_leads_df.pkl')
    old_pickle_path  = os.path.join(who_leads_folder, 'cleaned_who_leads_df_old.pkl')
    
    # Load and backup original
    df = pd.read_pickle(pickle_path)
    df.to_pickle(old_pickle_path)
    print("Saved original data as old version.")
    
    # Clean the 'post_text' column
    df['post_text'] = df['post_text'].apply(clean_post_text)
    
    # Save cleaned
    df.to_pickle(pickle_path)
    print("Saved cleaned data, replacing the original file.")
    
    # Double-check for any remaining URLs or RTs
    url_mask = df['post_text'].str.contains(r"\w*tp[s]?://", na=False, case=False)
    rt_mask  = df['post_text'].str.contains(r"^RT\s+@", na=False, case=False)

    n_urls = int(url_mask.sum())
    n_rts  = int(rt_mask.sum())

    print(f"Post-cleaning: {n_urls} rows still contain URLs")
    if n_urls > 0:
        print("Examples:", df.loc[url_mask, 'post_text'].head().tolist())

    print(f"Post-cleaning: {n_rts} rows still start with RT @")
    if n_rts > 0:
        print("Examples:", df.loc[rt_mask, 'post_text'].head().tolist())

    if n_urls == 0 and n_rts == 0:
        print("✔️ All URLs and RTs have been removed.")
    else:
        print("⚠️ Some posts still contain URLs or RTs. Please review above examples.")

