import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')



# Performing Preprocessing

def tokens(documents):

    stop_words = set(stopwords.words('english'))
    processed_docs = []

    for doc in documents:
        tokens = word_tokenize(doc.lower())
        tokens = [word for word in tokens if word.isalpha()]  
        tokens = [word for word in tokens if word not in stop_words]
        processed_docs.append(tokens)

    return processed_docs


# Vocabulary Creation
def vocabulary(processed_docs):
    vocab = sorted(set(word for doc in processed_docs for word in doc))
    return vocab

# One Hot Encoding
def one_hot_encoding(processed_docs, vocab):
    vectors = []
    for doc in processed_docs:
        vector = [1 if word in doc else 0 for word in vocab]
        vectors.append(vector)

    return pd.DataFrame(vectors, columns=vocab)


# 4. Bag of Words Function
# -------------------------------
def bag_of_words(processed_docs, vocab):
    vectors = []
    for doc in processed_docs:
        vector = [doc.count(word) for word in vocab]
        vectors.append(vector)
    return pd.DataFrame(vectors, columns=vocab)


# -------------------------------
# TF Calculation
# -------------------------------
def compute_tf(processed_docs, vocab):
    tf = []
    for doc in processed_docs:
        doc_len = len(doc)
        vector = [doc.count(word)/doc_len if doc_len > 0 else 0 for word in vocab]
        tf.append(vector)

    return pd.DataFrame(tf, columns=vocab)


# -------------------------------
# IDF Calculation
# -------------------------------
def compute_idf(processed_docs, vocab):
    N = len(processed_docs)
    idf_dict = {}

    for word in vocab:
        df = sum(1 for doc in processed_docs if word in doc)
        idf_dict[word] = np.log(N / (df + 1))

    return idf_dict


# -------------------------------
# TF-IDF Function (MAIN)
# -------------------------------
def compute_tfidf(processed_docs, vocab):
    tf_df = compute_tf(processed_docs, vocab)
    idf_dict = compute_idf(processed_docs, vocab)

    tfidf_vectors = []

    for i, doc in enumerate(processed_docs):
        vector = []
        for word in vocab:
            tf_val = tf_df.iloc[i][word]
            tfidf_val = tf_val * idf_dict[word]
            vector.append(tfidf_val)
        tfidf_vectors.append(vector)

    return pd.DataFrame(tfidf_vectors, columns=vocab)