import pandas as pd

# adjust this to wherever you’re running
clean_path = "/home/export/hbailey/data/embedding_resonance/who_leads_who_follows/cleaned_who_leads_df.pkl"
old_path   = "/home/export/hbailey/data/embedding_resonance/who_leads_who_follows/cleaned_who_leads_df_old.pkl"

# load both
df_clean = pd.read_pickle(clean_path)
df_old   = pd.read_pickle(old_path)

# look at the same few rows before vs after
for idx in [0, 1, 2, 3, 4]:
    print(f"--- row {idx} before ---")
    print(df_old.loc[idx, "post_text"])
    print(f"--- row {idx} after  ---")
    print(df_clean.loc[idx, "post_text"])
    print()