from text_representation import tokens, vocabulary , one_hot_encoding


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
onehot_df = one_hot_encoding(processed_docs, vocab)

print("Processed Docs:", processed_docs)
print("Vocabulary:", vocab)
print("\nOne-Hot Encoding:\n", onehot_df)