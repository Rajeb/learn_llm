import re
sentences = [
    "Cats love playing.",
    "Dogs love playing with balls."
]
print(sentences)

# build the vocabulary
# Lowercase and remove punctuation
cleaned = [re.sub(r'[^\w\s]', '', s.lower()) for s in sentences]
# Split into words and get unique set
vocab = sorted(set(' '.join(cleaned).split()))
print("Vocabulary:", vocab)

# Step 3: Embed the sentence
def bag_of_words(sentence, vocab):
    words = sentence.split()
    return [words.count(word) for word in vocab]

bow_vectors = [bag_of_words(s, vocab) for s in cleaned]
print("Bag-of-Words vectors:")
for vec in bow_vectors:
    print(vec)