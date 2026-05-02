from text_representation import bag_of_words, tokens,vocabulary

# 4. Bag of Words Function
# -------------------------------
documents = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love coding in python"
]

# Step 1: Preprocess
processed_docs = tokens(documents)

# Step 2: Vocabulary
vocab = vocabulary(processed_docs)

# Step 3: One-Hot Encoding
bow = bag_of_words(processed_docs, vocab)

print("Processed Docs:", processed_docs)
print("Vocabulary:", vocab)
print("\n BOW:\n", bow)