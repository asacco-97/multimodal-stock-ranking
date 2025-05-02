import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import ast
import os

# === Load pretrained BERT ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

def get_bert_embedding(headlines):
    """
    Takes a list of headlines and returns a 768-dim vector (mean of [CLS] tokens).
    """
    embeddings = []

    for headline in headlines:
        if not headline.strip():
            continue
        inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embedding)

    if not embeddings:
        return torch.zeros(768).numpy()

    embeddings = torch.cat(embeddings, dim=0)
    avg_embedding = torch.mean(embeddings, dim=0)
    return avg_embedding.numpy()

def embed_news(input_path="data/processed/daily_dataset.csv", output_path="data/processed/daily_with_embeddings.parquet"):
    df = pd.read_csv(input_path)

    # Convert stringified lists into actual lists
    df["headlines"] = df["headlines"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    embeddings = []

    for headlines in tqdm(df["headlines"], desc="Encoding headlines"):
        emb = get_bert_embedding(headlines)
        embeddings.append(emb)

    # Convert list of arrays to DataFrame
    emb_df = pd.DataFrame(embeddings, columns=[f"bert_{i}" for i in range(768)])
    df_with_emb = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_with_emb.to_parquet(output_path, index=False)
    print(f"Saved embedded dataset to {output_path}")

if __name__ == "__main__":
    embed_news()
