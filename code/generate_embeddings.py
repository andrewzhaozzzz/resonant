#!/usr/bin/env python
import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import gc

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # token-level embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_embeddings(pickle_path, col_name, model_directory, tokenizer_path, save_name, batch_size=256, checkpoint_interval=10000):
    df = pd.read_pickle(pickle_path)
    documents = df[col_name].fillna(" ").tolist()
    print(f"Computing embeddings for {len(documents)} documents")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModel.from_pretrained(model_directory)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    output_dir = os.path.join(model_directory, "embedding_output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_embeddings = []
    total_docs = len(documents)
    num_batches = (total_docs - 1) // batch_size + 1

    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            model_output = model(**encoded)
        embeddings = mean_pooling(model_output, encoded['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
        batch_num = i // batch_size + 1
        docs_processed = i + len(batch)
        print(f"Processed batch {batch_num}/{num_batches} ({docs_processed} documents)")

        del encoded, model_output, embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if docs_processed % checkpoint_interval == 0 or docs_processed == total_docs:
            checkpoint_embeddings = torch.cat(all_embeddings, dim=0)
            checkpoint_path = os.path.join(output_dir, f"{save_name}_embeddings_checkpoint_{docs_processed}.pt")
            torch.save(checkpoint_embeddings, checkpoint_path)
            print(f"Saved checkpoint at {docs_processed} documents to {checkpoint_path}")

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

    who_leads_folder = os.path.join(data_dir, 'who_leads_who_follows')
    pickle_path = os.path.join(who_leads_folder, 'cleaned_who_leads_df.pkl')
    compute_embeddings(pickle_path, 'post_text', model_directory, tokenizer_path, "who_leads_model")