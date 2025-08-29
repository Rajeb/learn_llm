from main import get_embedding
text = "Buy groceries: milk, eggs, bread"
embedding = get_embedding(text)
print(f"Text: {text}")
print(f"Embedding shape: {embedding.shape}")
print(f"First 5 values: {embedding[:5]}")
