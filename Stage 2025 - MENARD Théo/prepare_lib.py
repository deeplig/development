"""prepare_lib.py

Programme permettant le pré-traitement d'une banque de fragments de format .sdf
et retourne un dictionnaire en format .pkl à utiliser avec le générateur de pseudo-molécules

Auteur : MENARD Théo
"""

import argparse
# ---------------------- Définition des argument et options d'appel --------------------------------
# Créer le parser
parser = argparse.ArgumentParser(description="Ce Programme trie une banque de données, de fragment moléculaire, sous format SDF et la transforme en dictionnaire rechargeable sous format PKL.")
# Définir les arguments attendus
parser.add_argument("lib_sdf", help="Nom du fichier .sdf.")
# Analyser les arguments
args = parser.parse_args()

# --------------------------------------------------------------------------------------------------
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os
import random
import pickle

name = args.lib_sdf
lib = Chem.SDMolSupplier(f"{name}.sdf")
dist_dico={'1-2':[],'2-3':[],'3-4':[],'4-5':[],'5-6':[],
            '6-7':[],'7-8':[],'8-9':[],'9-10':[],'10-11':[],
            '11-12':[],'12-13':[],'13-14':[],'14-15':[],'15+':[]}
target = ['Cl.','.Cl','.Br','Br.']
for linker in lib:
  try:
    smi = Chem.MolToSmiles(linker)
    if '.' in smi:
        if target in smi:
            print(smi)
            for t in target:
                smi = smi.replace(t, '')
            print(smi)
            linker = Chem.MolFromSmiles(smi)
        else:
          continue
        if linker is None:
            continue
    linker = Chem.AddHs(linker)
    AllChem.EmbedMolecule(linker)
    AllChem.MMFFOptimizeMolecule(linker)
    conf = linker.GetConformer()
    positions = [conf.GetAtomPosition(i) for i in range(linker.GetNumAtoms())]
  except:
    continue
  # Prendre le premier et dernier atome
  first_atom_pos = positions[0]
  last_atom_pos = positions[-1]

  # Calculer la distance euclidienne
  distance = np.sqrt((first_atom_pos.x-last_atom_pos.x)**2 +
                      (first_atom_pos.y-last_atom_pos.y)**2 +
                      (first_atom_pos.z-last_atom_pos.z)**2)

  if distance > 1 and distance < 2:
    dist_dico['1-2'].append(linker)
  elif distance > 2 and distance < 3:
    dist_dico['2-3'].append(linker)
  elif distance > 3 and distance < 4:
    dist_dico['3-4'].append(linker)
  elif distance > 4 and distance < 5:
    dist_dico['4-5'].append(linker)
  elif distance > 5 and distance < 6:
    dist_dico['5-6'].append(linker)
  elif distance > 6 and distance < 7:
    dist_dico['6-7'].append(linker)
  elif distance > 7 and distance < 8:
    dist_dico['7-8'].append(linker)
  elif distance > 8 and distance < 9:
    dist_dico['8-9'].append(linker)
  elif distance > 9 and distance < 10:
    dist_dico['9-10'].append(linker)
  elif distance > 10 and distance < 11:
    dist_dico['10-11'].append(linker)
  elif distance > 11 and distance < 12:
    dist_dico['11-12'].append(linker)
  elif distance > 12 and distance < 13:
    dist_dico['12-13'].append(linker)
  elif distance > 13 and distance < 14:
    dist_dico['13-14'].append(linker)
  elif distance > 14 and distance < 15:
    dist_dico['14-15'].append(linker)
  elif distance > 15:
    dist_dico['15+'].append(linker)
with open(f'{name}_dico.pkl', 'wb') as fichier:
    pickle.dump(dist_dico, fichier)

print(f"Le dictionnaire a été enregistré avec succès dans {name}_dico.pkl")