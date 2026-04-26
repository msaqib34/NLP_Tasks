import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download required data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

st.title("🧒 English Tutor - Parts of Speech")

text = st.text_area("Enter a sentence:")

# POS mapping to simple categories
pos_map = {
    'NN': 'Noun', 'NNS': 'Noun', 'NNP': 'Proper Noun', 'NNPS': 'Proper Noun',
    'VB': 'Verb', 'VBD': 'Verb', 'VBG': 'Verb', 'VBN': 'Verb', 'VBP': 'Verb', 'VBZ': 'Verb',
    'JJ': 'Adjective', 'JJR': 'Adjective', 'JJS': 'Adjective',
    'RB': 'Adverb', 'RBR': 'Adverb', 'RBS': 'Adverb',
    'IN': 'Preposition',
    'DT': 'Determiner',
    'PRP': 'Pronoun', 'PRP$': 'Pronoun',
    'CC': 'Conjunction',
    'UH': 'Interjection',
    'CD': 'Number',
}

# Kid-friendly explanations
explanations = {
    "Noun": "A naming word. It tells us the name of a person, place, or thing.",
    "Proper Noun": "A special name like Ali, Lahore, or Pakistan.",
    "Verb": "An action word. It tells what someone is doing.",
    "Adjective": "A describing word. It tells how something looks or feels.",
    "Adverb": "Tells us more about a verb. Like how or when something happens.",
    "Preposition": "Shows position like in, on, under.",
    "Determiner": "Words like a, an, the that come before nouns.",
    "Pronoun": "A word used instead of a noun like he, she, it.",
    "Conjunction": "Joins words like and, but.",
    "Interjection": "Shows sudden feeling like wow!, oh!",
    "Number": "Tells counting like one, two, three."
}

if st.button("Analyze"):
    if text.strip():
        text= text.capitalize()
        #words = word_tokenize(text)
        #tagged = pos_tag(words)
        tagged = pos_tag(word_tokenize(text))

        st.subheader("📊 Analysis")

        for word, tag in tagged:
            simple_tag = pos_map.get(tag, "Other")
            explanation = explanations.get(simple_tag, "This is a special type of word.")

            st.markdown(f"""
            ### 🔹 Word: **{word}**
            - POS Tag: `{tag}`
            - Type: **{simple_tag}**
            - Explanation: {explanation}
            """)

    else:
        st.warning("Please enter a sentence.")