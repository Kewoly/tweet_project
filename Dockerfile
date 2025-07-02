FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers du projet dans l'image
COPY . /app

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger les ressources NLTK nécessaires
RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4

# Commande par défaut si aucun argument n'est passé
CMD ["python", "main.py"]