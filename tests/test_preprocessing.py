import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.preprocessing import clean_text
from nltk.corpus import stopwords

# Exemple de stopwords anglais
STOPWORDS = set(stopwords.words('english'))

def test_clean_empty_string():
    assert clean_text("") == "", "Le texte vide doit rester vide après nettoyage"

def test_clean_punctuation_and_digits():
    cleaned = clean_text("!!! $$$ 123")
    assert cleaned == "", "La ponctuation et les chiffres doivent être supprimés"

def test_clean_short_words():
    cleaned = clean_text("Hi I am AI in ML")
    tokens = cleaned.split()
    assert all(len(token) >= 3 for token in tokens), "Tous les tokens doivent avoir au moins 3 lettres"

def test_stopwords_removed():
    example = "This is a sample tweet with the and is of"
    cleaned = clean_text(example)
    tokens = cleaned.split()
    assert not any(token in STOPWORDS for token in tokens), "Les stopwords doivent être supprimés"

def test_vocab_reduction():
    original = "Love loving loved lover lovingly"
    cleaned = clean_text(original)
    original_tokens = original.split()
    cleaned_tokens = cleaned.split()
    assert len(set(cleaned_tokens)) < len(set(original_tokens)), "Le vocabulaire doit être réduit après le nettoyage"

def test_stemming_or_lemmatization_effect():
    text = "running runs runner"
    cleaned = clean_text(text)
    tokens = cleaned.split()
    assert len(set(tokens)) == 1 or len(set(tokens)) < 3, "Stemming/Lemmatisation doit regrouper les formes similaires"
