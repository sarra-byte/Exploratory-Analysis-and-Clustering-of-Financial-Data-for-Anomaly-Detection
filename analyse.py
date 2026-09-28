# ==========================================================
# FICHIER 1 : analyse.py (VERSION MODIFIÉE POUR DÉMO)
# ==========================================================
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, silhouette_score

print("="*60)
print(" DÉMARRAGE DU PROJET : DÉTECTION DE FRAUDE (DBSCAN)")
print("="*60)

# 1. Chargement et échantillonnage des données
# ===========================================
if not os.path.exists("creditcard.csv"):
    print(" ERREUR : Le fichier 'creditcard.csv' est introuvable !")
    exit()

print("1. Chargement et échantillonnage intelligent...")
data = pd.read_csv("creditcard.csv")

# --- MODIFICATION ICI : On force la présence de fraudes ---
# On sépare les fraudes et les normaux
df_fraud = data[data['Class'] == 1]
df_normal = data[data['Class'] == 0]

# On prend TOUTES les fraudes disponibles (492)
# Et on prend 9500 normaux au hasard
# Cela nous fait un dataset d'environ 10 000 lignes
df_normal_sample = df_normal.sample(n=9500, random_state=42)

# On recolle les morceaux
data = pd.concat([df_fraud, df_normal_sample])

# On mélange le tout pour ne pas avoir les fraudes toutes au début
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   -> Taille de l'échantillon : {len(data)} transactions")
print(f"   -> Dont fraudes réelles incluses : {len(data[data['Class']==1])}")
# ----------------------------------------------------------

# 2. PRÉTRAITEMENT
# ========================
print("2. Prétraitement et Normalisation...")
X = data.drop("Class", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. MODÉLISATION (DBSCAN)
# ========================
print("3. Exécution de DBSCAN (Clustering)...")
# Note : Avec plus de fraudes, on peut ajuster légèrement eps si besoin, 
# mais 2.5 reste une bonne base pour commencer.
dbscan = DBSCAN(eps=2.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

data["Cluster"] = clusters
anomalies_detected = np.sum(clusters == -1)



# 4. RÉDUCTION DE DIMENSION (PCA pour la viz)
# ========================
print("4. Réduction de dimension (PCA)...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)






# 5. ÉVALUATION
# ========================
print("5. Calcul des métriques de performance...")
# On considère que Cluster -1 = Fraude (1), et tout autre Cluster = Normal (0)
predictions = (clusters == -1).astype(int)

# On génère le rapport (output_dict=True pour l'utiliser dans Streamlit)
report_dict = classification_report(data["Class"], predictions, output_dict=True)

# Silhouette Score (Mesure la qualité des clusters)
# On ne le calcule que si on a au moins 2 clusters différents
mask = clusters != -1
if len(set(clusters)) > 1:
    # On calcule le score sur un sous-ensemble pour aller vite (optionnel)
    sil_score = silhouette_score(X_scaled, clusters)
else:
    sil_score = -1

# 6. SAUVEGARDE
# ========================
print("6. Sauvegarde des résultats...")
os.makedirs("results", exist_ok=True)

# Sauvegarde des tableaux Numpy
np.save("results/X_pca.npy", X_pca)
np.save("results/clusters.npy", clusters)
np.save("results/labels.npy", data["Class"].values)
np.save("results/amounts.npy", data["Amount"].values)

# Sauvegarde des données échantillonnées (CRITIQUE pour le dashboard)
data.to_csv("results/sampled_data.csv", index=False)

# Sauvegarde du rapport et métriques
with open("results/report_dict.pkl", "wb") as f:
    pickle.dump(report_dict, f)

# Création du résumé CSV
summary_df = pd.DataFrame({
    "Métrique": ["Total transactions", "Fraudes réelles", "Anomalies détectées", "Précision (fraude)", "Rappel (fraude)", "F1-score (fraude)", "Silhouette Score"],
    "Valeur": [
        len(data), 
        np.sum(data["Class"] == 1), 
        anomalies_detected, 
        report_dict['1']['precision'], 
        report_dict['1']['recall'], 
        report_dict['1']['f1-score'], 
        sil_score
    ]
})
summary_df.to_csv("results/summary.csv", index=False)

print(" SUCCÈS : Analyse terminée avec succès.")
print(" Tu peux maintenant lancer : streamlit run dashboard.py")