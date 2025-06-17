# Environnement

Ce code a été tester sur Python 3.11 avec Tensorflow 2.18. 

un fichier .yaml contenant l'environnement est disponible. Il peut être mis en place via conda.

```
conda env create -f environnement.yml
conda activate analysis
```

Il est aussi nécessaire d'ajouter la commande vina au PATH:

```
export PATH="$PATH:./autodock_vina_1_1_2_linux_x86/bin"
```

# Péparation de la bibliothèque de fragments

Un script de préparation de bibliothèque de fragments est disponible.
Par défaut, la bibliothèque *Enamine comprehensive linker* est déjà pré-traité.

```
python3 prepare_lib.py <nom de la bibliothèque .sdf>
```

Renvoie un fichier .pkl à utiliser dans le pseudo-mol generator.

# Extraction du ligand de référence

Un script d'extraction du ligand et du récepteur de référence est disponible.

```
python3 clean_ligand.py <pdb_id>
```

Créer les fichiers receptor.pdb et reference.pdb dans un dossier du nom de l'ID PDB.

# Lancer le pseudo-mol generator

```
python3 pseudomol_generator.py <pdb_id> -m <chaine à imiter (A/B)> -q <nombre de molécules> -o <création des fichier PDB (TRUE/FALSE)> -l <librairie en .pkl> -s <nom du fichier de sortie>
```
Renvoie un fichier .txt contenant des SMILES.

# docking

Un script de docking automatique est disponible.

```
python3 docking.py <fichier PDB du recepteur> <fichier PDB du ligand de reférence>
```
Si aucun ligand de référence n'est disponible, un fichier est généré par le pseudo-mol generator pour faire office de ligand de référence.

Renvoie un fichier .csv avec les meilleurs SMILES et leur score de docking.

# Evaluation

Un script d'évaluation de la génération des molécules est disponible.

```
python3 check.py <fichier .csv résultant du docking>
```

# Deep Q-Learning

Un script de deep Q-learning est disponible.

```
python3 DPQ.py <fichier PDB du recepteur> <fichier PDB du ligand de reférence> <model autoencodeu .keras>
```
Si aucun ligand de référence n'est disponible, un fichier est généré par le pseudo-mol generator pour faire office de ligand de référence.

Renvoie un fichier .keras du modèle entraîné.