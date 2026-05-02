from text_representation import tokens, vocabulary , compute_tfidf

documents = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love coding in python"
]

# Step 1: Preprocess
processed_docs = tokens(documents)

# Step 2: Vocabulary
vocab = vocabulary(processed_docs)


# Step 3: TF-IDF
tfidf = compute_tfidf(processed_docs, vocab)

print("\nTF-IDF:\n", tfidf)