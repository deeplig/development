"""
Ce programme permet la création de cluster à partir d'un fichier de SMILES

Auteur : Mike MAILLASSON
"""

from rdkit import Chem
from rdkit.Chem import MACCSkeys, Draw
import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import os
import string

import argparse
# ---------------------- Définition des argument et options d'appel --------------------------------
# Créer le parser
parser = argparse.ArgumentParser(description="Ce programme simule la capacité de docking d'un groupe de molécule sur un récepteur donné.")

# Définir les arguments attendus
parser.add_argument("smiles_file", help="fichier .txt contenant des SMILES.")
parser.add_argument("dock_dico", help="fichier .txt le dictionnaire resultant du docking.")

# Analyser les arguments
args = parser.parse_args()
# --------------------------------------------------------------------------------------------------
smiles_list = []
# Charger les SMILES depuis le fichier
with open(args.smiles_file, "r") as f:
    lines = f.read()
    lines = lines[:lines.rfind(',')]
    smiles_list.extend(lines.split(',\n'))

# Charger le dictionnaire des scores de docking
with open(args.dock_dico, "r") as f:
    score_dict = json.load(f)

# Convertir les SMILES en Mol et associer les scores
mols = []
valid_smiles = []
scores = []
print(f"SMILES list: {smiles_list}")
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        mols.append(mol)
        valid_smiles.append(smi)
        scores.append(score_dict.get(smi, None))  # None si absent


# Générer les MACCS fingerprints
fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
fps_array = np.array([fp.ToList() for fp in fps])

# Réduction de dimension par PCA
pca = PCA(n_components=2)
fps_pca = pca.fit_transform(fps_array)

# Clustering hiérarchique
clusterer = AgglomerativeClustering(n_clusters=4)
labels = clusterer.fit_predict(fps_array)

# Créer un dataframe
df = pd.DataFrame(fps_pca, columns=["PC1", "PC2"])
df["SMILES"] = valid_smiles
df["Cluster"] = labels.astype(str)
df["Score"] = scores

# Affichage PCA avec scores en couleurs
plt.figure(figsize=(10, 6))
sc = plt.scatter(df["PC1"], df["PC2"], c=df["Score"], cmap="viridis", s=100, edgecolor='k')
plt.title("PCA des MACCS (coloré par score de docking)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(sc, label="Docking Score")
plt.grid(True)
plt.savefig('docking_plot.png')
os.system("explorer.exe docking_plot.png")
# plt.show()

# Sauvegarder les clusters avec scores
df.to_csv("clusters_maccs_scores.csv", index=False)
print("Fichier 'clusters_maccs_scores.csv' exporté.")

from IPython.display import display

n_per_cluster = 3
for clust in sorted(df["Cluster"].unique()):
    sub_df = df[df["Cluster"] == clust].nsmallest(n_per_cluster, "Score")
    mols_to_draw = [Chem.MolFromSmiles(smi) for smi in sub_df["SMILES"]]
    legends = [f"{smi}\nScore: {score:.2f}" for smi, score in zip(sub_df["SMILES"], sub_df["Score"])]
    print(f"Cluster {clust} - Meilleurs scores :")
    img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=3, subImgSize=(600, 600), legends=legends)
    img.save("culster.png")
    os.system("explorer.exe culster.png")
    # display(img)
