import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
st.title("🔤 Text Tokenization App")
st.write("Enter a paragraph and see word & sentence tokenization.")

# Input box
text = st.text_area("Enter your paragraph here:")

if st.button("Tokenize"):
    if text.strip():

        # Sentence Tokenization
        sentences = sent_tokenize(text)

        # Word Tokenization
        words = word_tokenize(text)

        st.subheader("📌 Sentence Tokens")
        st.write(sentences)

        st.subheader("📌 Word Tokens")
        st.write(words)

        # Optional: counts
        st.subheader("📊 Statistics")
        st.write(f"Total Sentences: {len(sentences)}")
        st.write(f"Total Words: {len(words)}")

    else:
        st.warning("Please enter some text first.")