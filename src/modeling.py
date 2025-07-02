# src/modeling.py

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # Import du classifieur SVM
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

def build_pipeline(use_count: bool = False, model=None) -> Pipeline:
    """
    Construit un pipeline text → vecteur → classifieur.

    params:
      - use_count: si True, on utilise CountVectorizer, sinon TfidfVectorizer
      - model: instance d'un classifieur scikit‑learn (défaut LogisticRegression)
    """
    # Choix du vecteur de texte selon use_count
    vectorizer = CountVectorizer() if use_count else TfidfVectorizer()
    # Choix du classifieur, par défaut LogisticRegression
    clf = model if model is not None else LogisticRegression(max_iter=1000, class_weight='balanced')
    # Nom donné à la transformation textuelle dans le pipeline
    name_vect = 'count' if use_count else 'tfidf'

    # Construction du pipeline : extraction caractéristiques + classifieur
    return Pipeline([
        (name_vect, vectorizer),
        ('clf', clf)
    ])


def train_and_evaluate(
    df,
    text_col: str = 'clean_text',
    target_col: str = 'target',
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = None,
    use_count: bool = False,
    model=None
) -> dict:
    """
    Entraîne le pipeline sur df, renvoie dict de scores, prédictions, et (optionnellement) scores CV.

    params:
      - df: DataFrame contenant les données
      - text_col: nom de la colonne contenant le texte nettoyé
      - target_col: nom de la colonne cible
      - test_size: proportion des données réservées au test
      - random_state: graine pour la reproductibilité
      - cv: nombre de folds pour validation croisée, None = pas de CV
      - use_count: booléen, si True utilise CountVectorizer au lieu de TF-IDF
      - model: instance d’un classifieur scikit-learn (défaut LogisticRegression)

    return:
      dict avec clés :
        - 'pipeline': pipeline entraîné,
        - 'X_test', 'y_test', 'preds': données et prédictions test,
        - 'scores': dictionnaire des métriques sur le test,
        - 'cv_scores': dictionnaire des scores CV si cv != None
    """

    # 1. Préparation des données (texte et cible)
    X = df[text_col].fillna("")  # Remplacer les NaN par chaîne vide
    y = df[target_col]

    result = {}

    # 2. Création du pipeline avec le vecteur de texte et le modèle choisi
    pipe = build_pipeline(use_count=use_count, model=model)

    # 3. Si validation croisée demandée, calcul des scores CV
    if cv is not None and cv > 1:
        cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1')
        result['cv_scores'] = {
            'per_fold': cv_scores,
            'mean_f1': float(np.mean(cv_scores)),
            'std_f1': float(np.std(cv_scores))
        }

    # 4. Séparation train/test avec stratification selon la cible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 5. Entraînement du pipeline sur le jeu d'entraînement
    pipe.fit(X_train, y_train)

    # 6. Prédiction sur le jeu de test
    preds = pipe.predict(X_test)

    # 7. Calcul des métriques d’évaluation
    scores = {
        'accuracy':  accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall':    recall_score(y_test, preds, zero_division=0),
        'f1':        f1_score(y_test, preds, zero_division=0),
        'report':    classification_report(y_test, preds, zero_division=0, output_dict=True)
    }

    # 8. Stockage des résultats dans le dictionnaire à retourner
    result.update({
        'pipeline': pipe,
        'X_test':   X_test,
        'y_test':   y_test,
        'preds':    preds,
        'scores':   scores
    })

    return result
