"""pseudomol_generator.py

Ce Programme à pour but la création d'un pseudo-molécule à partir d'une 
interaction prot-prot. Dans un premier temps il extraie les acides aminés 
important de l'interaction prot-prot. Ensuite, leur chaînes latérales 
sont isolée et liées entre elles via des fragment moléculaire.

Ce programme prends en entré un dossier pdb de l'interaction prot-prot
et renvoie en sortie un fichier .txt contenant les SMILES des pseudomol générer.


Auteur : MENARD Théo, theo.menard@etu.univ-nantes.fr
Dans le cadre d'un stage de 2 mois pour le laboratoire du CRCI2NA, tuteur : MAILLASSON Mike
"""
import argparse
# ---------------------- Définition des argument et options d'appel --------------------------------
# Créer le parser
parser = argparse.ArgumentParser(description="Ce Programme génère des pseudo-molécules à partir d'une interaction protéine-protéine (iPP).")

# Définir les arguments attendus
parser.add_argument("pdb_id", help="ID du fichier PDB.")
parser.add_argument("-m","--mimic", default='A', help="Protéine à imité(A ou B).")
parser.add_argument("-q","--quantity", type=int, default=20, help="Quantité de pseudo-molecule à générer.")
parser.add_argument("-o","--pdb_output", default=False, help="Générer les fichier pdb des pseudo-molécules(Ture/False).")
parser.add_argument("-l","--lib", default="Enamine_Comprehensive_Linkers_dico.pkl", help="Librairie de linker, sous format .pkl, utilisée pour la généretion des pseudo-molécules.Pour générer votre propre librairie .pkl, utilisez le script fournis.")
parser.add_argument("-s","--sortie_smi", default="molecules.txt", help="Nom du fichier de sortie .txt.")
# Analyser les arguments
args = parser.parse_args()

# ---------------------- Importation des bibliothèques nécessaires --------------------------------
import string
import pickle # Permet le chargement du dictionnaire .pkl de la BDD de linker
import os # Permet la gestion de dossier
import shutil # permet la suppression du dossier plein
from pathlib import Path # Permet de gérer les chemin
import numpy as np
from Bio.PDB import *
from Bio.PDB import PDBParser, NeighborSearch
from Bio import Align
from Bio.Align import PairwiseAligner # Permet l'alignement de deux séquences
from scipy.spatial import cKDTree
import collections
import subprocess
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdMolTransforms
import csv
from tqdm import tqdm # Permet l'integration de bar de progression
import random

rseed_value=random.randint(0,1000000000)
random.seed(rseed_value)
print(f"Voici la graine de vos tirages aléatoires : {rseed_value}")

if os.path.exists('ligands.txt'):
  os.remove('ligands.txt')
# Fonction pour charger une structure PDB
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

# Fonction d'extraction des séquences protéiques
def extract_seq(file):
  """
  Exrtait la séquence protéique d'un fichier .pdb
  Args :
    file: Chemin vers le fichier PDB.
  Return :
    La séquence protéique en lettre unique.
  Raise :
    None
  """

  # Dictionnaire de conversion des codes à 3 lettres en 1 lettre
  aa_dict = {
      'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
      'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
      'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
      'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
      'HYP': 'X','SEP' : 'J'
  } # HYP = Hydroxiproline ; SEP = Phosposérine

  structure = load_pdb(file) # Chargement du fichier pdb

  # Extraction de la séquence pour chaque chaîne

  sequence = "".join(
    aa_dict[residue.resname]
    for model in structure
    for chain in model
    for residue in chain
    if residue.resname in aa_dict
  )
  return sequence.strip()

# Fonction d'extraction des chaines d'un fichier PDB
def extract_chain(file):
  """
  Exrtait les chaines de protéine d'un fichier .pdb en plusieur fichier pdb
  Args :
    file: Chemin vers le fichier PDB.
  Return :
    None
  Raise :
    None
  """
  # Paramètres
  colonne_cible = 22  # Numéro de la colonne qui contient la position d'un AA (en comptant à partir de 1)
  dossier_sortie = file.replace(".pdb","")  # Dossier où stocker les fichiers

  aa_list = ['ALA','ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN',
             'GLY','HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO',
             'SER','THR', 'TRP', 'TYR', 'VAL', 'HYP','SEP']

  # Création du dossier de sortie s'il n'existe pas
  os.makedirs(dossier_sortie, exist_ok=True)

  # Dictionnaire pour stocker les lignes par caractère trouvé
  groupes = {}

  # Lecture du fichier d'entrée
  with open(file, "r") as f:
      for line in f:
        if line.startswith("ATOM") or line.startswith("HETATM") and line[17:20] in aa_list:
          # Vérifier que la ligne est assez longue pour contenir la colonne cible
          if len(line) >= colonne_cible:
              caractere = line[colonne_cible - 1]  # Indexer en Python (début à 0)
              if caractere not in groupes:
                  groupes[caractere] = []  # Créer une liste pour ce caractère
              groupes[caractere].append(line)
  # Écriture des fichiers de sortie
  for caractere, lines in groupes.items():
      nom_fichier_sortie = os.path.join(dossier_sortie, f"chain_{caractere}.pdb")
      with open(nom_fichier_sortie, "w") as f_out:
          f_out.writelines(lines)
      # print(f"Fichier créé : {nom_fichier_sortie} ({len(lines)} lignes)")

# Fonction pour identifier les contacts entre deux protéines
def identify_interactions(protein_a, protein_b, distance_threshold=6.5):
    """
    Identifie les acides aminés proches entre deux protéines.

    Args:
      protein_a (Bio.PDB.Structure): Structure de la protéine A.
      protein_b (Bio.PDB.Structure): Structure de la protéine B.
      distance_threshold (float): Seuil de distance (en Å) pour considérer que deux acides aminés sont en interaction.
    Return :
      interactions_a (list): Liste des interactions entre les acides aminés de la protéine A.
      interactions_b (list): Liste des interactions entre les acides aminés de la protéine B.
    Raise :
      None
    """
    interactions_a = []
    interactions_b = []
    nxt = True
    i=0
    while( nxt == True and i<2):
      for residue_a in protein_a.get_residues():
          if not residue_a.has_id("CA"):  # Filtrer uniquement les acides aminés
              continue
          for residue_b in protein_b.get_residues():
              if not residue_b.has_id("CA"):
                  continue
              distance = residue_a["CA"] - residue_b["CA"]
              if distance <= distance_threshold:
                  interactions_a.append((residue_a.get_resname(), residue_a.id[1]))
                  interactions_b.append((residue_b.get_resname(), residue_b.id[1]))
      if len(interactions_a) == 0:
        distance_threshold = distance_threshold + 2
        i = i+1
      else:
        nxt = False
    return interactions_a, interactions_b, distance_threshold

def combine_pdbs(output_file, *input_files):
    """
    Combine plusieurs fichiers PDB en un seul avec gestion des chaines.

    Args:
      output_file (str): Chemin du fichier de sortie.
      *input_files (str): Liste des chemins des fichiers PDB à combiner.
    Return :
      None
    Raise :
      None
    """
    io = PDBIO()
    structure = Structure.Structure("combined")
    model = Model.Model(0)
    structure.add(model)

    # Tous les caractères possibles pour les chaines (A-Z, a-z, 0-9)
    available_chain_ids = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    used_chain_ids = set()

    for pdb_file in input_files:
        parser = PDBParser()
        temp_structure = parser.get_structure("temp", pdb_file)

        for temp_model in temp_structure:
            for temp_chain in temp_model:
                original_id = temp_chain.id
                new_id = original_id

                # Trouver un ID de chaine disponible
                while new_id in used_chain_ids or len(new_id) != 1:
                    if not available_chain_ids:
                        raise ValueError("Plus d'identifiants de chaine disponibles")
                    new_id = available_chain_ids.pop(0)

                used_chain_ids.add(new_id)
                temp_chain.id = new_id  # Modifier l'ID de la chaine
                model.add(temp_chain)

    # Sauvegarder la structure combinée
    io.set_structure(structure)
    io.save(output_file)

def is_chain_collagen(seq):
  """
  Identifie si une séquence protéique est du collagène.

  Args:
    seq(str): La séquence protéique à analyser.
  Return :
    Boléenne : True si la séquence est du collagène, False sinon.
  Raise :
    None
  """
  if "GPXGPX" in seq:
    return True
  else:
    return False

def compare_chain(dossier):
  """
  Comparaison des chaines de protéine des fichier .pdb d'un dossier entier 
  Args :
    dossier : Chemin vers le fichier PDB.
  Return :
    None
  Raise :
    None
  """
  sequences = []
  toremove = []
  coll = []
  fab=False
  for fichier in sorted(os.listdir(dossier)):  # Liste des fichiers/dossiers
    chemin_fichier = os.path.join(dossier, fichier)

    if os.path.isfile(chemin_fichier):  # Vérifie si c'est un fichier
        seq = extract_seq(chemin_fichier)
        sequences.append((fichier,seq))

  # Comparaison des séquences
  for fichier1,seq1 in sequences:
    for fichier2,seq2 in sequences:
      if fichier1 != fichier2:
        # Alignement 2 à 2
        aligner = PairwiseAligner()
        aligner.mode = 'global'
        alignements = aligner.align(seq1, seq2)
        # Récupération du score (nombre de correspondances)
        score = alignements[0].score / max(len(seq1), len(seq2))  # Score normalisé (0 à 1)

        # Verification des scores :
        # print(f"Score de similarité entre {fichier1} et {fichier2} : {score}")

        chemin_fichier1 = os.path.join(dossier, fichier1)
        chemin_fichier2 = os.path.join(dossier, fichier2)

        # Verification si chaine est collagène
        if is_chain_collagen(seq1) and chemin_fichier1 not in coll:
          coll.append(chemin_fichier1)
        if is_chain_collagen(seq2) and chemin_fichier2 not in coll:
          coll.append(chemin_fichier2)

        # Verification de la présence de FAB
        if fichier1 == "chain_H.pdb" and fichier2 == "chain_L.pdb":
          fab=[chemin_fichier1,chemin_fichier2]

        # Suppression si les séquences sont trop similaires (exemple : seuil 90%)
        seuil = 0.9
        if score >= seuil:
          if(chemin_fichier1 not in toremove and chemin_fichier2 not in toremove and (chemin_fichier1 or chemin_fichier2) not in coll):
            toremove.append(chemin_fichier2)
        else:
          interA, interB, dist = identify_interactions(load_pdb(chemin_fichier1),load_pdb(chemin_fichier2))
          if len(interA) == 0 and chemin_fichier1 not in toremove and chemin_fichier2 not in toremove:
            toremove.append(chemin_fichier2)
  # Suppression des fichiers temporaires
  if fab:
    combine_pdbs(dossier+'/fab.pdb', chemin_fichier1, chemin_fichier2)
    for fichier in fab:
      os.remove(fichier)
  if coll:
    combine_pdbs(dossier+'/collagen.pdb', *coll)
    for fichier in coll:
      os.remove(fichier)
  # Suppression des fichier en double
  for fichier in toremove:
    os.remove(fichier)
  return

pdb_id = args.pdb_id
pdb_id = pdb_id.lower()
# Si le dossier existe déjà, il est supprimer
if os.path.exists(f"{pdb_id}.pdb"):
  os.remove(f"{pdb_id}.pdb")
if os.path.isdir(f"{pdb_id}/"):
  shutil.rmtree(f"{pdb_id}/")
pdbl = PDBList()

pdbl.retrieve_pdb_file(pdb_id, pdir='.', file_format ="pdb", overwrite=True) # Téléchargement du fichier PDB
os.rename("pdb"+pdb_id+".ent", pdb_id+".pdb")
pdb_file = pdb_id+".pdb"

extract_chain(pdb_file)
compare_chain(pdb_id)

# Charger les structures des protéines A et B
mol = []
for fichier in sorted(os.listdir(pdb_id)):  # Liste des fichiers/dossiers
  if os.path.isfile(os.path.join(pdb_id, fichier)):  # Vérifie si c'est un fichier
    mol.append(pdb_id+"/"+fichier)
receptor = load_pdb(mol[0])  # Remplacer par le chemin de la protéine A
ligand = load_pdb(mol[1]) # Remplacer par le chemin de la protéine B

# Seuils de distance par type d'interaction
CUTOFF = 3.5  # Distance maximale pour toutes les interactions (Å)
HYDROGEN_BOND_CUTOFF = 3.4
IONIC_CUTOFF = 5.0
HYDROPHOBIC_CUTOFF = 5.2

# Dictionnaires pour les propriétés des résidus
POSITIVE_RESIDUES = {'ARG', 'LYS', 'HIS'}
NEGATIVE_RESIDUES = {'ASP', 'GLU'}

AROMATIC_RESIDUES = {'PHE', 'TYR', 'TRP', 'HIS'}
HYDROPHOBIC_ALKYL = {'ALA','VAL', 'LEU', 'ILE', 'MET', 'PRO'}
HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'TYR', 'PRO', 'GLY'}

def load_pdb_structure(pdb_file):
    """Charge une structure PDB et retourne les atomes avec leurs résidus"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    return [(atom, atom.get_parent()) for atom in structure.get_atoms()]

def classify_interaction(atom1, atom2, distance):
    """Classifie l'interaction entre deux atomes"""
    res1 = atom1.get_parent().resname
    res2 = atom2.get_parent().resname
    # Amélioration possible en prenant compte de l'angle entre les deux acide aminé

    # Interactions hydrogène
    donors_acceptors = {'N', 'O', 'P', 'S'}
    elements = {atom1.element, atom2.element}
    if atom1.element in donors_acceptors and atom2.element in donors_acceptors and distance <= HYDROGEN_BOND_CUTOFF:
      if (res1 in POSITIVE_RESIDUES and res2 in NEGATIVE_RESIDUES) or \
          (res1 in NEGATIVE_RESIDUES and res2 in POSITIVE_RESIDUES):
        return 'HB + Electro : Salt Bridge'
      else:
        return 'HBond Conventional'
    elif 'C' in elements and ('N' in elements or 'O' in elements) and HYDROGEN_BOND_CUTOFF < distance < 3.8  :
      return 'HBond Carbon'

    # Interactions ioniques (court terme)
    # Pi-Cation
    if distance <= IONIC_CUTOFF:
      if (res1 or res2) in AROMATIC_RESIDUES and (res1 and res2) in POSITIVE_RESIDUES and (res1 and res2) not in AROMATIC_RESIDUES:
          return 'Electrostatic Pi-Cation'
      if (res1 in POSITIVE_RESIDUES and res2 in NEGATIVE_RESIDUES) or \
         (res1 in NEGATIVE_RESIDUES and res2 in POSITIVE_RESIDUES):
          return 'Electrostatic Attrative Charge'

    # Interactions hydrophobes
    if res1 in HYDROPHOBIC_RESIDUES and res2 in HYDROPHOBIC_RESIDUES and distance <= HYDROPHOBIC_CUTOFF:
      if res1 in HYDROPHOBIC_ALKYL and res2 in HYDROPHOBIC_ALKYL:
        if res1 in AROMATIC_RESIDUES or res2 in AROMATIC_RESIDUES:
          return 'Hydrophobic Pi-Alkyl'
        else:
          return 'Hydrophobic Alkyl'
      elif res1 in AROMATIC_RESIDUES and res2 in AROMATIC_RESIDUES:
          return 'Hydrophobic Pi-Pi'
      elif atom1.element == 'C' and atom2.element == 'C':
          return 'Hydrophobic'

    # Van der Waals (par défaut si dans le cutoff)
    if distance <= CUTOFF:
        return 'Van der Waals'

    return None

def analyze_interactions(prot1_atoms, prot2_atoms):
    """Analyse les interactions entre deux ensembles d'atomes"""
    # Préparation des données pour la protéine 2
    prot2_coords = np.array([atom[0].get_coord() for atom in prot2_atoms])
    prot2_tree = cKDTree(prot2_coords)

    interactions = []

    # Recherche des interactions
    for atom1, res1 in prot1_atoms:
        neighbors = prot2_tree.query_ball_point(atom1.get_coord(), 6.5)
        for idx in neighbors:
            atom2, res2 = prot2_atoms[idx]
            distance = np.linalg.norm(atom1.get_coord() - atom2.get_coord())
            interaction_type = classify_interaction(atom1, atom2, distance)

            if interaction_type:
                interactions.append((
                    res1, atom1,
                    res2, atom2,
                    interaction_type,
                    distance
                ))

    return interactions

def interaction_AB(pdb1, pdb2):
  """
  Analyse les interactions entre deux protéines.

  Args:
    pdb1 (str): Chemin vers le fichier PDB de la protéine A.
    pdb2 (str): Chemin vers le fichier PDB de la protéine B.
  Return :
    None
  Raise :
    None
  """
  # Chargement des structures
  prot1 = load_pdb_structure(pdb1)
  prot2 = load_pdb_structure(pdb2)

  # Analyse des interactions
  results = analyze_interactions(prot1, prot2)

  # Affichage des résultats
  displayed_pairs = set()
  interactionsA = []
  interactionsB = []

  chains=[]
  tys=[]
  for j in results:
    if j[2].parent.id not in chains:
      chains.append(j[2].parent.id)
  for j in results:
    if j[4] not in tys and j[4] != "Van der Waals":
      tys.append(j[4])
  for chain in chains:
    for ty in tys:
      print(ty)
      for res1, a1, res2, a2, types, dist in results:
        if res2.parent.id==chain:
          if types==ty:
            if ty:
                if (res1,res2) not in displayed_pairs:
                  # print(f"{(res1.resname,res1.id[1],res1.parent.id)} - {(res2.resname,res2.id[1],res2.parent.id)}")
                  interactionsA.append((res1.resname,res1.id[1]))
                  interactionsB.append((res2.resname,res2.id[1]))
                displayed_pairs.add((res1,res2))
            else:
              interactionsA.append((res1.resname,res1.id[1]))
              interactionsB.append((res2.resname,res2.id[1]))
  print(sorted(interactionsB))
  return displayed_pairs

AA_inter = interaction_AB(mol[0], mol[1])

def extract_residues(pdb_path, residue_ids, already_seen, output_path):
    """
    Extrait les chaîne latérale des acides aminés qui ont été identifier
    comme faisant partie de l'interaction entre la protéine A et B
    Args :
      matchs : un ensemble de tuple qui contient les AA d'interêt
      A_pdb : fichier pdb de la protéine A
      B_pdb : fichier pdb de la protéine B
      output_file : fichier de sortie
    Return :
      None
    Raise :
      None
    """
    residue_ids_str = {str(res.id[1]).strip() for res in residue_ids}  # Set pour rapidité
    with open(output_path, "w") as g:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    res_id = line[23:26].strip()
                    atom_name = line[12:16].strip()  # Plus fiable pour noms d'atomes
                    if res_id in residue_ids_str and atom_name not in {'N', 'C', 'O'} and line not in already_seen:
                        g.write(line)
                        already_seen.add(line)
                elif line.startswith("TER"):
                    g.write("TER\n")


# Préparation des résidus
A_AA = [A for A, _ in AA_inter]
B_AA = [B for _, B in AA_inter]
already_seen = set()
output_file = pdb_id+"/AA_inter_"
# Écriture des fichiers A.pdb et B.pdb
extract_residues(mol[0], A_AA, already_seen, output_file + "A.pdb")
extract_residues(mol[1], B_AA, already_seen, output_file + "B.pdb")


def separate_AA(file, output_dir=f"{pdb_id}/AA_"):
  """
  Sépare les acides aminés dans des fichiers pdb distincts
  Args :
    file : fichier pdb contenant les acides aminés
    output_dir : dossier de sortie
  Return :
    None
  Raise :
    None
  """
  os.makedirs(output_dir+file[14]+"/", exist_ok=True)
  resi_list = [(r,c) for m in load_pdb(file) for c in m for r in c]
  for residue, chain in resi_list:
      # Créer une nouvelle structure pour cet acide aminé
      io = PDBIO()
      io.set_structure(residue)

      # Nom du fichier de sortie (ex: 1_ALA_A.pdb)
      output_file = os.path.join(
          output_dir+file[14]+"/",
          f"{residue.id[1]}_{residue.get_resname()}_{chain.id}.pdb"
      )

      # Sauvegarder dans un fichier séparé
      io.save(output_file)

choix = args.mimic
os.system(f"cp {pdb_id}/AA_inter_{choix}.pdb ./reference.pdb")

if choix =='A':
  os.system(f"cp {mol[1]} ./receptor.pdb")
else:
   os.system(f"cp {mol[0]} ./receptor.pdb")
separate_AA(f"{pdb_id}/AA_inter_{choix}.pdb")

# ------------------- Liaison des chaînes latérales ------------------

def load_pdb_with_coords(pdb_file):
  """Charge un fichier PDB en conservant les coordonnées 3D"""
  mol = Chem.MolFromPDBFile(pdb_file, removeHs=True)  # Supprime les hydrogènes
  if not mol:
      raise ValueError(f"Impossible de charger le fichier {pdb_file}")
  return mol
def calculate_interatomic_distance(pdb_file1, atom_idx1, pdb_file2, atom_idx2):
    """
    Calcule la distance entre deux atomes de molécules différentes
    Args:
      pdb_file1/2 : Chemins des fichiers PDB
      atom_idx1/2 : Indices des atomes dans chaque molécule (0-based)
    Returns:
      distance (float) : Distance en Angströms
    Raise:
      None
    """
    # Chargement des molécules
    mol1 = load_pdb_with_coords(pdb_file1)
    mol2 = load_pdb_with_coords(pdb_file2)
    mol1 = Chem.RemoveHs(mol1)
    mol2 = Chem.RemoveHs(mol2)

    # Vérification des indices
    for mol, idx in [(mol1, atom_idx1), (mol2, atom_idx2)]:
        if idx >= mol.GetNumAtoms():
            raise ValueError(f"Index {idx} hors limites (max = {mol.GetNumAtoms()})")

    # Extraction des coordonnées
    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()

    pos1 = np.array(conf1.GetAtomPosition(atom_idx1))
    pos2 = np.array(conf2.GetAtomPosition(atom_idx2))

    # Calcul de la distance euclidienne
    distance = np.linalg.norm(pos1 - pos2)
    return distance-1

def link_mol(mol1_pdb, mol2_pdb, mol3_pdb, linker_lib_pkl, output):
  """
  Relie 3 molécules entre elles via un fragment provenant d'une banque de molécule (linker) en .sdf.
  La banque est précedement traiter (par un autre programme) pour être trié par taille.

  Args:
    mol1_pdb : Fichier .pdb de la 1er molécule
    mol2_pdb : Fichier .pdb de la 2ème molécule
    mol3_pdb : Fichier .pdb de la 3ème molécule
    linker_lib_pkl : Librairie pré-traitée sous format .pkl
    output : nom des fichiers de sortie
  Return :
    connected : la molécule finale (les 3 molécules lié par 2 linkers)
    linker1 : 1er linker utilisé
    linker2 : 2ème linker utilisé
  Raise :
    None
  """
  # Charger les molécules à lier (PDB)
  mol1 = Chem.MolFromPDBFile(mol1_pdb, removeHs=True, sanitize=True)
  mol2 = Chem.MolFromPDBFile(mol2_pdb, removeHs=True, sanitize=True)
  mol3 = Chem.MolFromPDBFile(mol3_pdb, removeHs=True, sanitize=True)

  # Charger la banque de linkers (pkl)
  with open(linker_lib_pkl, "rb") as f:
    loaded_dict = pickle.load(f)

  def get_linker_sites(mol):
      """
      Trouve les sites de liaison (ex: atomes avec valence libre)
      """
      sites = []
      for atom in mol.GetAtoms():
          if atom.GetNumImplicitHs() > 0:  # Valence libre
              sites.append(atom.GetIdx())
      return sites

  sites_mol1 = get_linker_sites(mol1)
  sites_mol2 = get_linker_sites(mol2)
  sites_mol3 = get_linker_sites(mol3)

  # Calcule des distance entre chaque chaîne latéral
  distances =[]
  dist1 = calculate_interatomic_distance(mol1_pdb, sites_mol1[0], mol2_pdb, sites_mol2[0])
  distances.append(dist1)
  dist2 = calculate_interatomic_distance(mol2_pdb, sites_mol2[0], mol3_pdb, sites_mol3[0])
  distances.append(dist2)

  # Filtrer les linker à utiliser en fonction de leur taille
  i=0
  for dist in distances:
    i+=1
    if dist > 1 and dist < 2:
      linker = random.choice(loaded_dict['1-2'])
    elif dist > 2 and dist < 3:
      linker = random.choice(loaded_dict['2-3'])
    elif dist > 3 and dist < 4:
      linker = random.choice(loaded_dict['3-4'])
    elif dist > 4 and dist < 5:
      linker = random.choice(loaded_dict['4-5'])
    elif dist > 5 and dist < 6:
      linker = random.choice(loaded_dict['5-6'])
    elif dist > 6 and dist < 7:
      linker = random.choice(loaded_dict['6-7'])
    elif dist > 7 and dist < 8:
      linker = random.choice(loaded_dict['7-8'])
    elif dist > 8 and dist < 9:
      linker = random.choice(loaded_dict['8-9'])
    elif dist > 9 and dist < 10:
      linker = random.choice(loaded_dict['9-10'])
    elif dist > 10 and dist < 11:
      linker = random.choice(loaded_dict['10-11'])
    elif dist > 11 and dist < 12:
      linker = random.choice(loaded_dict['11-12'])
    elif dist > 12 and dist < 13:
      linker = random.choice(loaded_dict['12-13'])
    elif dist > 13 and dist < 14:
      linker = random.choice(loaded_dict['13-14'])
    elif dist > 14 and dist < 15:
      linker = random.choice(loaded_dict['14-15'])
    elif dist > 15:
      linker = random.choice(loaded_dict['15+'])
    if i == 1:
      linker1 = linker
    elif i == 2:
      linker2 = linker
  linker1 = Chem.RemoveHs(linker1)
  linker2 = Chem.RemoveHs(linker2)
  # Trouver les sites de connexion du linker (ex: atomes avec groupe "attachment point")
  linker_sites1 = get_linker_sites(linker1)
  linker_sites2 = get_linker_sites(linker2)

  # Combiner les molécules (ex: mol1 + linker + mol2)
  combined = Chem.CombineMols(mol1, linker1)
  combined = Chem.CombineMols(combined, mol2)
  combined = Chem.CombineMols(combined, linker2)
  combined = Chem.CombineMols(combined, mol3)
  combined = Chem.RemoveHs(combined)

  # Créer des liaisons (ex: entre le site 0 de mol1 et le site 0 du linker)
  ed = Chem.EditableMol(combined)
  ed.AddBond(sites_mol1[0], linker_sites1[0] + mol1.GetNumAtoms() , order=Chem.BondType.SINGLE) # linker_sites[1] + mol1.GetNumAtoms()
  ed.AddBond(mol1.GetNumAtoms()+ linker_sites1[-1], sites_mol2[0] + mol1.GetNumAtoms() + linker1.GetNumAtoms(), order=Chem.BondType.SINGLE)
  ed.AddBond(sites_mol2[0] + mol1.GetNumAtoms() + linker1.GetNumAtoms(), linker_sites2[0] + mol2.GetNumAtoms() + mol1.GetNumAtoms() + linker1.GetNumAtoms(), order=Chem.BondType.SINGLE)
  ed.AddBond(linker_sites2[-1] + mol2.GetNumAtoms() + mol1.GetNumAtoms() + linker1.GetNumAtoms(), sites_mol3[0] + mol1.GetNumAtoms() + linker1.GetNumAtoms() + mol2.GetNumAtoms() + linker2.GetNumAtoms(), order=Chem.BondType.SINGLE)
  connected = ed.GetMol()

  # (optionnel) Génération des fichiers .pdb pour chaque molécules si l'utilisateur le choisit
  if args.pdb_output:
    try:
        with open(f"link.smiles", "w") as f:
          f.write(f"{Chem.MolToSmiles(connected)}")
        os.system(f'obabel link.smiles -O {output} --gen3d best -p 7.4 --canonical --ff MMFF94')
        os.remove("link.smiles")
    except:
        print(f"Echec de la génération du fichier {output}.")
        pass

  return connected, linker1, linker2

list_aa=os.listdir(f"{pdb_id}/AA_{choix}/")
ma_liste_triee = sorted(list_aa, key=lambda x: int(x.split('_')[0]))
if os.path.exists(args.sortie_smi):
      os.remove(args.sortie_smi)
# Génération d'une quantité de pseudo-molécules déterminé par l'utilisateur (default=20)
with tqdm(range(args.quantity), desc="Génération de pseudo-molécule") as pbar:
  for i in pbar:
    nb = random.randint(0,len(ma_liste_triee)-3) # Choisie 3 chaînes latérals qui se suivent pour avoir une plus petite molécule
    molecules = ma_liste_triee[nb:nb+3]
    output_filename = f"connected{i + 1}.pdb"
    pbar.set_postfix(file=output_filename) # mise à jour de la bar de progression

    # Appel de la fonction link_mol
    mol, L1, L2 = link_mol(f"{pdb_id}/AA_{choix}/"+molecules[0],
                          f"{pdb_id}/AA_{choix}/"+molecules[1],
                          f"{pdb_id}/AA_{choix}/"+molecules[2] ,
                          args.lib,
                          output_filename)

    smiM = Chem.MolToSmiles(mol)
    with open(args.sortie_smi, "a") as f:
      f.write(f"{smiM},\n")
print(f"SMILES enregistrés sous {args.sortie_smi} !")

# Suppression  des fichiers et dossier temporaires
os.remove(f"{pdb_id}.pdb")
shutil.rmtree(pdb_id)