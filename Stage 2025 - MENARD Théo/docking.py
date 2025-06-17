"""docking.py

Ce Programme permet un docking automatique.

Auteurs : Adaptation de MENARD Théo depuis le code original de O. Trott et A. J. Olson
"""

import argparse
# ---------------------- Définition des argument et options d'appel --------------------------------
# Créer le parser
parser = argparse.ArgumentParser(description="Ce programme simule la capacité de docking d'un groupe de molécule sur un récepteur donné.")

# Définir les arguments attendus
parser.add_argument("receptor_pdb_file", help="fichier PDB du recepteur.")
parser.add_argument("reference_pdb_file", help="fichier PDB du ligands de réference.")
parser.add_argument("smiles_file", help="fichier .txt contenant des SMILES.")

# Analyser les arguments
args = parser.parse_args()

# ------------ Importation de bibliothèques ---------------
import numpy as np
import json
import csv
import os
from Bio.PDB import *
from rdkit import Chem
from pathlib import Path
import subprocess
from tqdm import tqdm

# Fonction pour exécuter les commandes shell et vérifier les erreurs
def run_in_shell(command):
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print(result.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(e.output.decode('utf-8'))

receptor_pdb_file = args.receptor_pdb_file
ligand_ref = args.reference_pdb_file

ligand_smiles = []
ref_smile=Chem.MolToSmiles(Chem.MolFromPDBFile(ligand_ref))
if '.' not in ref_smile:
    ligand_smiles.append(ref_smile)
# Lecture des SMILES supplémentaires
with open(args.smiles_file, 'r') as f:
    lines = f.read()
    lines = lines[:lines.rfind(',')]
    ligand_smiles.extend(lines.split(',\n'))

print(f'Ligand SMILES: {ligand_smiles}')

# Préparation du fichier récepteur
print('Préparation du fichier récepteur...')

# Ajouter les charges de Gasteiger et préparer le récepteur pour le docking avec MGLTools
try:
    os.system(f"prepare_receptor4.py -r {receptor_pdb_file} -o receptor.pdbqt -A hydrogens -U nphs_lps") # -v = verbose (pour voir les détails)
except Exception as e:
    print(f"Erreur lors de la préparation du récepteur : {e}")

print("Préparation du récepteur terminée. Fichier enregistré sous receptor.pdbqt")

# Paramétrisation et ajout des charges Gasteiger aux ligands
pbar = tqdm(enumerate(ligand_smiles), total=len(ligand_smiles), desc="Préparation des ligands...")
for i, smile in pbar:
    with open(f"link.smiles", "w") as f:
        f.write(f"{smile}")
    pbar.set_postfix(file=f"pmol{i}.pdbqt")
    os.system(f'obabel link.smiles -O pmol{i}.pdb --gen3d best -p 7.4 --canonical --ff MMFF94 > /dev/null 2>&1') # > /dev/null 2>&1
    os.remove("link.smiles")
    if os.path.exists(f"pmol{i}.pdb"):
        os.system(f'prepare_ligand4.py -l pmol{i}.pdb -o pmol{i}.pdbqt -U nphs_lps') # -v = verbose (pour voir les détails)
        os.remove(f"pmol{i}.pdb")
    else:
        print(f"Erreur : Le fichier pmol{i}.pdb n'existe pas.")

# Création du dictionnaire de smiles
ligand_dict = {}
for smiles in ligand_smiles:
    ligand_dict[smiles] = {}

def get_molecule_center(pdb_file, method='geometric'):
    """
    Calcule le centre 3D d'une molécule à partir d'un fichier PDB.
    
    Args:
        pdb_file (str): Chemin vers le fichier PDB
        method (str): Méthode de calcul ('geometric' ou 'mass')
    
    Returns:
        np.array: Coordonnées [x, y, z] du centre en Angströms
    """
    # Chargement du fichier PDB en conservant les hydrogènes et les coordonnées
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    if not mol or not mol.GetNumConformers():
        raise ValueError("Molécule non valide ou sans coordonnées 3D")
    
    conformer = mol.GetConformer()
    positions = np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    
    if method == 'geometric':
        # Centre géométrique moyen
        return np.mean(positions, axis=0)
    elif method == 'mass':
        # Centre de masse (nécessite les masses atomiques)
        masses = np.array([atom.GetMass() for atom in mol.GetAtoms()])
        return np.average(positions, axis=0, weights=masses)
    else:
        raise ValueError("Méthode non reconnue. Choisir 'geometric' ou 'mass'")

# Exemple d'utilisation
center = get_molecule_center(ligand_ref, method='geometric')
print(center)

# Paramètres pour la boîte de docking
bxi, byi, bzi = round(center[0]), round(center[1]), round(center[2])  # Centre de la boîte
bxf, byf, bzf = 30, 30, 30   # Taille de la boîte

# Création des fichiers de configuration et exécution de vina
pbar = tqdm(enumerate(ligand_smiles), total=len(ligand_smiles), desc="Docking en cours...")
for i, smile in pbar: 
    ligand_pdbqt = f'pmol{i}.pdbqt'
    pbar.set_postfix(file=ligand_pdbqt)
    
    if not os.path.exists(ligand_pdbqt):
        print(f"Error: File {ligand_pdbqt} not found")
        continue
    # Créer le fichier de configuration pour le docking
    with open(f"config{i}.txt", "w") as f:
        f.write("#CONFIGURATION FILE (options not used are commented) \n")
        f.write("\n")
        f.write("#INPUT OPTIONS \n")
        f.write(f"receptor = receptor.pdbqt \n")
        f.write(f"ligand = {ligand_pdbqt} \n")
        f.write("#flex = [flexible residues in receptor in pdbqt format] \n")
        f.write("#SEARCH SPACE CONFIGURATIONS \n")
        f.write("#Center of the box (values bxi, byi and bzi) \n")
        f.write(f"center_x = {bxi} \n")
        f.write(f"center_y = {byi} \n")
        f.write(f"center_z = {bzi} \n")
        f.write("#Size of the box (values bxf, byf et bzf) \n")
        f.write(f"size_x = {bxf} \n")
        f.write(f"size_y = {byf} \n")
        f.write(f"size_z = {bzf} \n")
        f.write("#OUTPUT OPTIONS \n")
        f.write("#out = \n")
        f.write("#log = \n")
        f.write("\n")
        f.write("#OTHER OPTIONS \n")
        f.write("#cpu =  \n")
        f.write("#exhaustiveness = \n")
        f.write("#num_modes = \n")
        f.write("#energy_range = \n")
        f.write("#seed = \n")

    # Exécuter AutoDock Vina avec le fichier de configuration
    vina_command = f'vina --config config{i}.txt --out output{i}.pdbqt > /dev/null 2>&1' # > /dev/null 2>&1 : rends la commande silencieuse
    run_in_shell(vina_command)

    # Vérifier si le fichier de sortie de vina existe
    output_pdbqt = f'output{i}.pdbqt'
    if not os.path.exists(output_pdbqt):
        print(f"Error: File {output_pdbqt} not found")
        continue

    # Utiliser Open Babel pour convertir les résultats d'AutoDock Vina de pdbqt en pdb
    obabel_command = f'obabel -ipdbqt {output_pdbqt} -opdb -O output{i}.pdb -m > /dev/null 2>&1' # > /dev/null 2>&1
    run_in_shell(obabel_command)

    # Lire le fichier de sortie pour obtenir les résultats de Vina
    output_pdb = f'output{i}1.pdb'
    if os.path.exists(output_pdb):
        with open(output_pdb, 'r') as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT"):
                    words = line.split()
                    ligand_dict[list(ligand_dict.keys())[i-1]] = float(words[3])
    else:
        print(f"Error: File {output_pdb} not found")

    # Supprimer les fichiers temporaires
    os.remove(f'config{i}.txt')
    os.remove(output_pdbqt)

    for k in range(2, 10):
        try:
            os.remove(f'output{i}{k}.pdb')
        except FileNotFoundError:
            pass

# Sauvegarder le dictionnaire des ligands
    with open("ligand_dict.txt", 'w') as f:
        json.dump(ligand_dict, f)

for filename in os.listdir():
    if filename.endswith('.pdbqt'):
        os.remove(filename)
    if filename.endswith('.pdb') and filename.startswith('output'):
        os.remove(filename)    

filtered_data = {key: value for key, value in ligand_dict.items() if isinstance(value, (int, float))}

# Get the binding energy of the first SMILES string
first_smiles = list(filtered_data.keys())[0]
first_binding_energy = filtered_data[first_smiles]

# Initialize a list to store the indices of the SMILES strings with larger binding energies
indices = []
molecule_data = []
# Initialiser les variables pour trouver la SMILES avec la meilleure (plus basse) énergie de liaison
best_smiles = None
best_binding_energy = float('inf')  # Initialiser à une très grande valeur positive
best_index = 0  # Initialiser l'indice de la meilleure SMILES

# Iterate over the dictionary and find the indices of the SMILES strings with larger binding energies
for i, (smiles, energy) in enumerate(filtered_data.items()):
    if energy <= first_binding_energy:
        indices.append(i)
        molecule_data.append({'Index': i, 'SMILES': smiles, 'Binding Energy': energy})
        
    # Trouver la SMILES avec la meilleure énergie de liaison (la plus négative)
    if energy < best_binding_energy:
        best_binding_energy = energy
        best_smiles = smiles
        best_index = i

# Ajouter un classement basé sur l'énergie de liaison (ordre croissant)
for rank, molecule in enumerate(sorted(molecule_data, key=lambda x: x['Binding Energy']), start=1):
    molecule['Rank'] = rank

# Sauvegarder les résultats au format CSV
csv_filename = 'selected_molecules.csv'
csv_columns = ['Index', 'SMILES', 'Binding Energy', 'Rank']

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(molecule_data)

# Afficher la SMILES avec la meilleure énergie de liaison, sa valeur et son indice
print(f"La chaîne SMILES avec la meilleure énergie de liaison est : {best_smiles}, "
      f"avec une énergie de liaison de {best_binding_energy}, et son indice est {best_index}")
print(f"Les données des molécules sélectionnées ont été enregistrées dans le fichier : {csv_filename}")