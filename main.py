import pandas as pd
import os
import numpy as np
# import openai
from sentence_transformers import SentenceTransformer

# file to store notes
notes_file = "notes.csv"

# function to save notes
def save_note(note):
    df = pd.DataFrame([[note]], columns=["note"])
    if os.path.exists(notes_file):
        try:
            df_existing = pd.read_csv(notes_file)
            df_existing = pd.concat([df_existing, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            # If file exists but empty
            df_existing = df
        df_existing.to_csv(notes_file, index=False)
    else:
        df.to_csv(notes_file, index=False)


# function to load all notes
def load_notes():
    if os.path.exists(notes_file):
        return pd.read_csv(notes_file)["note"].to_list()
    return[]

# Quick test
# save_note("Buy groceries: milk, eggs, bread")
# print("Current Notes:")
# print(load_notes())

#generate embeddings for notes
#we will use openAi embedding API 



# def get_embedding(text):
#     """generate embedding vector for a single text using OpenAi API"""
#     response = openai.embeddings.create(
#         input=text,
#         model="text-embedding-3-small"
#     )
#     return np.array(response['data'][0]['embedding'])


# def get_embedding(text):
#     """
#     Mock embedding for testing without API.
#     Returns a random vector of fixed size (1536).
#     """
#     np.random.seed(hash(text) % 2**32)  # deterministic per text
#     return np.random.rand(1536)


# hugging face
model = SentenceTransformer('all-MiniLM-L6-v2')
def get_embedding(text):
    """
    Generate embedding vector for a single note.
    Returns a NumPy array.
    """
    return model.encode(text)

# Step 4: Save note with embedding (optional)

# If you want to store embeddings in the CSV for future semantic search:

def save_note_with_embedding(note):
    embedding = get_embedding(note)
    df = pd.DataFrame([[note, embedding.tolist()]], columns=["note", "embedding"])
    
    if os.path.exists(notes_file):
        try:
            df_existing = pd.read_csv(notes_file)
            df_existing = pd.concat([df_existing, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            df_existing = df
        df_existing.to_csv(notes_file, index=False)
    else:
        df.to_csv(notes_file, index=False)
