"""
Programme permettant d'extraire les ligands et le récepteur d'un fichier PDB
"""
import argparse
# ---------------------- Définition des argument et options d'appel --------------------------------
# Créer le parser
parser = argparse.ArgumentParser(description="Ce Programme génère des pseudo-molécules à partir d'une interaction protéine-protéine (iPP).")

# Définir les arguments attendus
parser.add_argument("pdb_id", help="ID du fichier PDB.")

# Analyser les arguments
args = parser.parse_args()

from Bio.PDB import *
from pathlib import Path # Permet de gérer les chemin
import os
def load_pdb(filepath):
  """
  Charge une structure PDB à partir d'un fichier.
  Args :
    filepath: Chemin vers le fichier PDB.
  Return :
    Structure PDB.
  Raise :
    Si le fichier ne peut pas être touvé ou n'est pas en format .pdb
    renvoie None
  """
  try:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("Complex", filepath)
  except Exception as e:
     print(f"Impossible de charger le fichier {filepath}.\nErreur : {e}")
     return None
  return structure
def extract_ligname(file):
  """
  Exrtait le nom des ligands d'un fichier .pdb
  Args :
    file: Chemin vers le fichier PDB.
  Return :
    Liste des nom des ligands.
  Raise :
    None
  """
  prot = load_pdb(file)

  exception = ["HOH","NH2","SO4","ZN","UNX"]
  ligands = set()

  for model in prot:
      for chain in model:
          for residue in chain:
              # Vérifier si ce n'est pas un acide aminé ou un nucléotide
              if residue.id[0] != ' ' and residue.resname not in exception:
                  # Vérifier si le résidu est un ligand
                  ligands.add(residue.resname)  # Ajouter le nom du ligand

  # Afficher les ligands trouvés
  print("Ligands présents dans le fichier PDB :", ligands)
  return ligands

def extract_ligand(file,ligand):
  """
  Exrtait le ligand d'un fichier .pdb
  Args :
    file: Chemin vers le fichier PDB.
    ligand: Nom du ligand.
  Return :
    None
  Raise :
    None
  """
# Extract ligand from PDB file
  dossier_sortie = file.replace(".pdb","")  # Dossier où stocker les fichiers

  # Création du dossier de sortie s'il n'existe pas
  os.makedirs(dossier_sortie, exist_ok=True)

  nom_fichier_sortie = os.path.join(dossier_sortie, f"{ligand}.pdb")
  with open(nom_fichier_sortie, "w") as g:
      with open(file, 'r') as f:
          for line in f:
              row = line.split()
              if ligand in row:
                  g.write(line)
          g.write("END")
      print(f"Fichier {ligand}.pdb créé avec succès.")
  return

def extract_receptor(file):
  """
  Exrtait la protéine récepteur d'un fichier .pdb
  Args :
    file: Chemin vers le fichier PDB.
  Return :
    None
  Raise :
    None
  """
# Extract protein from PDB file
  dossier_sortie = file.replace(".pdb","")  # Dossier où stocker les fichiers

  # Création du dossier de sortie s'il n'existe pas
  os.makedirs(dossier_sortie, exist_ok=True)
  nom_fichier_sortie = os.path.join(dossier_sortie, f"receptor.pdb")
  with open(nom_fichier_sortie, "w") as g:
      with open(file, 'r') as f:
          for line in f:
              row = line.split()
              if row[0] == "ATOM":
                  g.write(line)
              elif row[0] == "TER":
                  g.write("TER\n")
          g.write("END")
      print("Fichier receptor.pdb créé avec succès.")
  return

pdb_id = args.pdb_id
pdbl = PDBList()
pdbl.retrieve_pdb_file(pdb_id, pdir='.', file_format ="pdb", overwrite=True)
os.rename("pdb"+pdb_id+".ent", pdb_id+".pdb")
pdb_file = pdb_id+".pdb"
ligands = extract_ligname(pdb_file)
for lig in ligands:
  extract_ligand(pdb_file,lig)
extract_receptor(pdb_file)