import pandas as pd
import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer

notes_file = "notes.csv"
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text)

def save_note_with_embedding(note):
    embedding = get_embedding(note)
    # Convert to JSON string
    embedding_str = json.dumps(embedding.tolist())
    df_new = pd.DataFrame([[note, embedding_str]], columns=["note", "embedding"])
    
    if os.path.exists(notes_file):
        try:
            df_existing = pd.read_csv(notes_file)
            df_existing = pd.concat([df_existing, df_new], ignore_index=True)
        except pd.errors.EmptyDataError:
            df_existing = df_new
    else:
        df_existing = df_new
    
    df_existing.to_csv(notes_file, index=False)

def load_notes_with_embeddings():
    if os.path.exists(notes_file):
        df = pd.read_csv(notes_file)
        if "embedding" in df.columns:
            df["embedding"] = df["embedding"].apply(
                lambda x: np.array(json.loads(x)) if pd.notna(x) else np.array([])
            )
        return df
    return pd.DataFrame(columns=["note", "embedding"])
