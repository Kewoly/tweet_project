import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Assure-toi d'avoir téléchargé punkt et stopwords une bonne fois (une seule ligne !)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Minuscule
    text = text.lower()
    # 2. Supprimer URLs/mentions/hashtags
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    # 3. Supprimer ponctuation & chiffres
    text = re.sub(r"[^a-z\s]", "", text)
    # 4. Tokenisation
    tokens = word_tokenize(text)
    # 5. Filtrage mots courts et stopwords
    tokens = [t for t in tokens if len(t) > 2 and t not in stop_words]
    # 6. Stemming (Porter)
    tokens = [stemmer.stem(t) for t in tokens]
    # 7. Reconstruction
    return " ".join(tokens)
