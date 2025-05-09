import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from tqdm import tqdm
import ast
import os
import numpy as np
import torch.nn.functional as F

# === Load FinBERT for embeddings and sentiment ===
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
model_embedding = AutoModel.from_pretrained("yiyanghkust/finbert-tone")
model_sentiment.eval()
model_embedding.eval()

def get_finbert_features(headlines):
    """Takes a list of headlines and returns:
    - 768-dim average embedding vector
    - average sentiment probabilities: [positive, neutral, negative]
    """
    embeddings = []
    sentiment_scores = []

    for headline in headlines:
        if not headline.strip():
            continue

        inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True, max_length=64)

        # Sentiment
        with torch.no_grad():
            output_sent = model_sentiment(**inputs)
            probs = F.softmax(output_sent.logits, dim=1)
            sentiment_scores.append(probs.squeeze(0))

        # Embedding
        with torch.no_grad():
            output_emb = model_embedding(**inputs)
            cls_embedding = output_emb.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embedding)

    if not embeddings:
        return (np.zeros(768), np.zeros(3))

    embeddings = torch.cat(embeddings, dim=0)
    avg_embedding = torch.mean(embeddings, dim=0).numpy()

    sentiments = torch.stack(sentiment_scores, dim=0)
    avg_sentiment = torch.mean(sentiments, dim=0).numpy()

    return avg_embedding, avg_sentiment

def embed_news(input_path="data/processed/daily_dataset.csv", output_path="data/processed/daily_with_finbert.parquet"):
    df = pd.read_csv(input_path)

    # Convert stringified lists into actual lists
    df["headlines"] = df["headlines"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    embeddings = []
    sentiments = []

    for headlines in tqdm(df["headlines"], desc="Encoding headlines with FinBERT"):
        emb, sent = get_finbert_features(headlines)
        embeddings.append(emb)
        sentiments.append(sent)

    # Convert to DataFrames
    emb_df = pd.DataFrame(embeddings, columns=[f"finbert_{i}" for i in range(768)])
    sent_df = pd.DataFrame(sentiments, columns=["sent_pos", "sent_neu", "sent_neg"])

    df_with_all = pd.concat([df.reset_index(drop=True), emb_df, sent_df], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_with_all.to_parquet(output_path, index=False)
    print(f"Saved FinBERT-embedded dataset with sentiment to {output_path}")

if __name__ == "__main__":
    embed_news()