"""check.py

Programme permettant de comparer la structure d'un batch de molécule par rapport à un ligands

Auteur: MENARD Théo
"""
import argparse
# ---------------------- Définition des argument et options d'appel --------------------------------
# Créer le parser
parser = argparse.ArgumentParser(description="Ce programme simule la capacité de docking d'un groupe de molécule sur un récepteur donné.")

# Définir les arguments attendus
parser.add_argument("molecules_csv", help="fichier CSV des SMILES à analyser.")

# Analyser les arguments
args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator, rdFMCS, Draw
from IPython.display import display
import csv
import os

smiles = []
with open(args.molecules_csv, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Index"] == "0":
            smiles_ref = row["SMILES"]
            energy_ref = row["Binding Energy"]
        else:
            smiles.append((row["SMILES"],row["Binding Energy"],row["Rank"]))
            index = row["Index"] 

# Crée le générateur Morgan
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
mol1 = Chem.MolFromSmiles(smiles_ref)
fp1 = generator.GetFingerprint(mol1)

for smi,score,rang in smiles:
    print(smi,score,rang)
    mol2 = Chem.MolFromSmiles(smi)
    # Empreintes
    fp2 = generator.GetFingerprint(mol2)

    # Similarité Tanimoto
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    print(f"Similarité de Tanimoto : {similarity:.2f}")
    similarite_dice = DataStructs.DiceSimilarity(fp1, fp2)
    print(f"Similarité de Dice: {similarite_dice:.2f}")

    mcs_result = rdFMCS.FindMCS([mol1, mol2])
    print("SMARTS commun :", mcs_result.smartsString)

    # Pour voir la molécule correspondante :
    common_substructure = Chem.MolFromSmarts(mcs_result.smartsString)

    # Trouver les atomes correspondant à la sous-structure dans mol1
    match1 = mol1.GetSubstructMatch(common_substructure)
    match2 = mol2.GetSubstructMatch(common_substructure)

    # Afficher les deux avec surbrillance
    img = Draw.MolsToGridImage(
        [mol1, mol2],
        highlightAtomLists=[match1, match2],
        molsPerRow=2,
        subImgSize=(600, 600),
        legends=[f"Reference\nSMILES : {smiles_ref}\nBinding Energy : {energy_ref}", f"SMILES : {smi}\nBinding Energy : {score}\nTanimoto : {similarity:.2f} & Dice : {similarite_dice:.2f}"]
    )
    if int(rang) < 5:
        img.save(f'check_result.png')
        os.system("explorer.exe check_result.png")
    # display(img)  