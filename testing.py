from main import load_notes_with_embeddings, save_note_with_embedding
save_note_with_embedding("Visited Yosemite National Park")
save_note_with_embedding("Stayed at a cabin in Tahoe")

df = load_notes_with_embeddings()
print(df.head())
print(type(df.loc[0, "embedding"]))     
print(df.loc[0, "embedding"].shape)  

