import os
import joblib
import pandas as pd

# Chemin relatif vers le modèle
model_path = os.path.join("notebooks", "models", "tweet_classifier.pkl")

# Charger le pipeline
pipeline = joblib.load(model_path)

# Exemple de données à prédire
data = {
    "clean_text": [
        "There is a fire in the city center!",
        "Had a relaxing day at the beach.",
        "Tornado warning in my area, stay safe!"
    ]
}

df = pd.DataFrame(data)

# Prédictions
preds = pipeline.predict(df["clean_text"])

for text, pred in zip(df["clean_text"], preds):
    label = "Catastrophe" if pred == 1 else "Normal"
    print(f"'{text}' => {label} ({pred})")
