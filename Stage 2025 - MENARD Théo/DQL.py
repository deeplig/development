"""Deep_Q_Learning.py

Ce programme permet un apprentissage par deep-q learning 
pour optimiser le SMILES en fonction de son score de docking.

Auteurs : MENARD Théo & GUITTENY Sarah
"""

import argparse
# ---------------------- Définition des argument et options d'appel --------------------------------
# Créer le parser
parser = argparse.ArgumentParser(description="Ce Programme à pour but la création d'un pseudo-molécule à partir d'une interaction prot-prot.")

# Définir les arguments attendus
parser.add_argument("receptor_pdb_file", help="fichier PDB du recepteur.")
parser.add_argument("reference_pdb_file", help="fichier PDB du ligands de réference.")

# Analyser les arguments
args = parser.parse_args()

# ---------------------------------Bibliothèques------------------------------------

import time
import numpy as np
from rdkit import Chem, RDLogger
# Désactive complètement les logs RDKit
RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm
import random
from collections import deque
import os
import tensorflow as tf
# Désactive les WARNING tensorflow
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Dropout, GRU, GaussianNoise
from keras.optimizers import Adam

# -------------------------------Mise en place du docking----------------------------------

from Bio.PDB import *
from pathlib import Path
import subprocess

def run_in_shell(command):
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print(result.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(e.output.decode('utf-8'))

receptor_pdb_file = args.receptor_pdb_file
ligand_ref = args.reference_pdb_file

ref_smile=Chem.MolToSmiles(Chem.MolFromPDBFile(ligand_ref))


print(f'Ref SMILES: {ref_smile}')

# Préparation du fichier récepteur
print('Préparation du fichier récepteur...')

# Ajouter les charges de Gasteiger et préparer le récepteur pour le docking avec MGLTools
try:
    os.system(f"python prepare_receptor4.py -r {receptor_pdb_file} -o receptor.pdbqt -A hydrogens -U nphs_lps") # -v = verbose (pour voir les détails)
    print("Préparation du récepteur terminée. Fichier enregistré sous receptor.pdbqt")
except Exception as e:
    print(f"Erreur lors de la préparation du récepteur : {e}")

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

# Paramètres pour la boîte de docking
bxi, byi, bzi = round(center[0]), round(center[1]), round(center[2])  # Centre de la boîte
bxf, byf, bzf = 30, 30, 30   # Taille de la boîte


# ---------------------------------Mise en place de l'encodage/decodage------------------------------------

max_length = 150
unique_chars =['\n', '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', 
 '7', '8', '9', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\', 
 ']', 'c', 'l', 'n', 'o', 'p', 'r', 's']

unique_chars.sort()
char_to_num = {char: i + 2 for i, char in enumerate(unique_chars)}  # Associer chaque caractère unique à un numéro
char_to_num['!'] = 0  # Ajouter le jeton de début au dictionnaire
char_to_num['E'] = 1  # Ajouter le jeton de fin au dictionnaire

# Ajouter un jeton pour les caractères inconnus
char_to_num['UNK'] = len(char_to_num)

num_to_char = {v: k for k, v in char_to_num.items()}
vocab_size = len(char_to_num)  # Nombre de caractères uniques dans le vocabulaire

# One-hot encode une chaîne SMILES en utilisant le dictionnaire de tokenisation
def one_hot_encode(smi, to_length=max_length + 2):
    """
    Transforme un SMILES en tableau de chiffre
      Args :
      Return :
      Raise :
    """
    result = np.zeros((to_length, vocab_size), dtype=np.uint8)
    result[0, char_to_num['!']] = 1  # Jeton de début
    for i, char in enumerate(smi):
        if char not in char_to_num:
            print(f'Character {char} not found in char_to_num.')
        result[i + 1, char_to_num.get(char, char_to_num['UNK'])] = 1  # Gérer tous les caractères, y compris les inconnus
    result[i + 2:, char_to_num['E']] = 1  # Jeton de fin
    return result

# One-hot decode un tableau SMILES encodé en utilisant le dictionnaire de dé-tokenisation
def one_hot_decode(array):
    """
    Transforme un tableau de chiffre en SMILES
      Args :
      Return :
      Raise :
    """
    result = ''  # Initialiser la chaîne SMILES
    for item in array:
        item = list(item)
        index = item.index(1)
        result += num_to_char[index]  # Ajouter le caractère approprié au résultat
    result = result.replace('!', '')  # Retirer le jeton de début
    result = result.split('E')[0]  # Troncature au premier jeton de fin
    return result

# Convertir le tableau de sortie softmax du modèle en tableau one-hot
def softmax_to_one_hot(array, temp):
    """
    Convertir le tableau de sortie softmax du modèle en tableau one-hot
      Args :
      Return :
      Raise :
    """
    result = np.zeros(array.shape, dtype=np.uint8)  # Initialiser le tableau de résultats
    for i, row in enumerate(array):
        with np.errstate(divide='ignore'):
            row_with_temp = np.exp(np.log(row) / temp) / np.sum(np.exp(np.log(row) / temp))  # Calculer les probabilités de l'échantillonnage
        result[i, np.random.choice(range(len(row)), p=row_with_temp)] = 1  # Définir l'index échantillonné à 1
    return result


class DQNAgent:
    def __init__(self, state_size, action_size, model_path=None):
        self.state_size = state_size
        # self.action_size = action_size
        self.memory = deque(maxlen=2000) # deque = double-ended queue de taille 2000
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.target_model = tf.keras.models.load_model(model_path)
            self.epsilon = self.epsilon_min

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.state_size,))
        x = layers.Dense(256, activation='relu')(inputs) # 64 ou 256
        x = layers.Dense(256, activation='relu')(x) # 64 ou 256
        outputs = layers.Dense(self.state_size, activation='linear')(x)
        # outputs = layers.Dense(self.action_size, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer='adam')
        # model.compile(loss='mse', 
        #              optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
        #              metrics=['accuracy'])
        return model

    def update_target_model(self):
        """Met à jour les poids du modèle en cours de création"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Enregistre dans la memoire
        
        Args :
            state : Etat de base
            action : Modifications apporter au vecteur
            reward : score d'amélioration 
            next_state : Nouveau vecteur
        Return :
            None
        """
        self.memory.append((state, action, reward, next_state, done))

    def explo(self, state):
        """Détermine l'exporation ou l'exploitation"""
        return np.random.rand() <= self.epsilon
    
    def decay_epsilon(self):
        """Applique la décroissance d'epsilon et garantit qu'il ne descende pas en dessous du minimum"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                # Si terminal, la target est juste la récompense
                targets[i] = actions[i]  # On conserve la modification qui a mené à ce résultat
            else:
                # Sinon, on utilise la meilleure prédiction future
                best_future_action = next_q_values[i]
                targets[i] = actions[i] + self.gamma * best_future_action

         # Entraînement sur les modifications optimales pondérées par les récompenses
        self.model.fit(states, targets, sample_weight=rewards, epochs=4, verbose=0)

    def save(self, path):
        if not path.endswith('.keras'):
            path += '.keras'
        self.model.save(path)

class SMILESOptimizer:
    def __init__(self, agent, autoencoder_model, docking_scorer, target_score):
        self.agent = agent
        self.autoencoder_model = autoencoder_model
        self.docking_scorer = docking_scorer
        self.target_score = target_score
        
        # Modification importante: Récupération de la taille latente différemment
        # On utilise maintenant la forme de sortie de la couche encoder
        encoder_output = autoencoder_model.get_layer('encoder').output
        self.latent_size = encoder_output.shape[-1]
    
    def generate_around(self, latent_seed, explo,sd=0.1 ,num_attempts=10, num_attempts_per_latent=1,temp=0.5):
        """
        Appel latent_to_smile plusieur fois depuis une version muter de la seed latent
            Args :
                Latent seed vector
                standard deviation of permitted mutation (sd = 0 means no mutation)
            Return :
                SMILES string
            Raise :
                None
        """

        best_score = float('inf')
        best_smiles = None
        best_latent = None
        best_mod = None
        molecule = None
        num_success = 0
        valid_smiles = []
        valid_mols = []
        # latent_size = self.autoencoder_model.get_layer('encoder').units

        if explo: # Exploration : modification aléatoire
            for i in tqdm(range(num_attempts)):
                modification = sd * np.random.randn(self.latent_size)
                mutated_latent_seed = latent_seed + modification

                # Convert to smiles
                smiles = latent_to_smile(mutated_latent_seed, temp = temp, num_attempts = num_attempts_per_latent, model = self.autoencoder_model)

                for smile in smiles:
                    molecule = Chem.MolFromSmiles(smile)
                    if molecule:
                        valid_smiles.append((smile,modification))
                        valid_mols.append(molecule)
                        num_success += 1
            print(f"Nombre de molécules valides : {len(valid_smiles)}")
        else: # Exploitation selon la mémoire
            while molecule is None and i < 2*num_attempts:
                state = latent_seed[0] if isinstance(latent_seed, (list, np.ndarray)) else latent_seed
                modification = self.agent.model.predict(state[np.newaxis, :], verbose=0)[0]
                smiles = latent_to_smile(mutated_latent_seed,temp,num_attempts_per_latent, self.autoencoder_model)
                molecule = (Chem.MolFromSmiles(smile) for smile in smiles)
            valid_smiles.append((smile,modification))
        for smile, modification in valid_smiles:
            try:
                score = docking_scorer(smile)
                if score < best_score:
                    best_score = score
                    best_smiles = smile
                    best_mod = modification
                    best_latent = mutated_latent_seed
            except Exception as e:
                print(f"Erreur de scoring: {str(e)}")
                continue
        # Calculate success rate
        # success_rate = num_success/(num_attempts * num_attempts_per_latent)

        return best_smiles, best_latent, best_mod, best_score if best_score != float('inf') else None
    
        
    def optimize(self,auto_model, initial_smiles, max_episodes=160,batch_size=16):
        initial_latent = smile_to_latent(initial_smiles, self.autoencoder_model)
        if initial_latent is None:
            raise ValueError("Le SMILES initial n'a pas pu être converti en vecteur latent")
        
        initial_score = self.docking_scorer(initial_smiles)
        best_score = initial_score
        best_smiles = initial_smiles
        best_latent = initial_latent
        
        for episode in range(max_episodes):
            current_latent = best_latent.copy()
            current_smiles = best_smiles
            current_score = best_score

            
            print(f"\nEpisode {episode + 1}/{max_episodes}")
            print(f"SMILES d'origine: {initial_smiles}")
            print(f"Score d'origine: {initial_score}")
            print(f"Meilleur SMILES: {best_smiles}")
            print(f"Meilleur score actuel: {best_score}")
            print(f"SMILES actuel: {current_smiles}")
            print(f"Score cible: {self.target_score}")
            print(f"Epsilon: {self.agent.epsilon:.4f}")
            
            if current_score <= self.target_score:
                print("Score cible atteint!")
                break
            
            state = current_latent[0]
            exploration = self.agent.explo(state)
            print(f"Exploration : {exploration}")
            
            # new_smiles, new_latent, new_score = self.generate_optimized_smiles(current_latent)
            new_smiles, new_latent, action ,new_score = self.generate_around(current_latent,exploration)
            
            if new_score is None:
                print("Échec de génération de SMILES valide")
                continue
            
            reward = initial_score - new_score # ou current_score - new_score
            done = new_score <= self.target_score
            self.agent.remember(state, action, reward, new_latent[0], done)

            if new_score < best_score:
                best_score = new_score
                best_smiles = new_smiles
                best_latent = new_latent
                print(f"Nouveau meilleur SMILES: {best_smiles}")
                print(f"Nouveau meilleur score: {best_score}")

            self.agent.decay_epsilon()
            self.agent.replay(batch_size)
            
            if episode % 10 == 0:
                self.agent.update_target_model()
        
        return best_smiles, best_score

def load_autoencoder_model(path):
    """Charge un modèle autoencodeur et vérifie sa structure"""
    if not path.endswith('.keras'):
        path += '.keras'
    model = tf.keras.models.load_model(path)
    print(model.summary())
    
    # Vérification que le modèle a bien une couche 'encoder'
    if 'encoder' not in [layer.name for layer in model.layers]:
        raise ValueError("Le modèle autoencodeur doit contenir une couche nommée 'encoder'")
    
    return model


def smile_to_latent(smile, model):
    """
    Convertion des SMILES de format 'string' en vecteurs latents
        Args :
        SMILES string or list of strings
        Return :
        latent vector(s) in an array of dimensions
        Raise :
    """

    ohsmile = []

    # One-hot encode SMILES string or list of strings
    if isinstance(smile, str):
        ohsmile.append(one_hot_encode(smile, max_length + 2))
    else:
        for smi in smile:
            ohsmile.append(one_hot_encode(smi, max_length + 2))

    ohsmile = np.array(ohsmile)

    # Define encoder model
    encoder_model = tf.keras.Model(inputs = model.input, outputs = model.get_layer('encoder').output)

    # Predict latent vector(s) from encoder model
    latent_vec = encoder_model.predict(ohsmile[:, :-1, :], verbose = 0)

    return latent_vec

def latent_to_smile(latent_vec, temp, num_attempts, model):
    # Conversion du vecteur latent en états initiaux
    def latent_to_state(latent_vec, model):
        latent_dim = model.get_layer('encoder').units
        input = Input(shape=(latent_dim,))
        latent = model.get_layer('decoder_pre')(input)
        process_model = tf.keras.Model(inputs=input, outputs=latent)
        state = process_model.predict(latent_vec, verbose=0)
        return state.reshape(1, -1)  # Shape (1, units)

    # Définition du modèle décodeur
    decoder_input = Input(shape=(1, vocab_size), batch_size=1, name='input')
    gru_units = model.get_layer('encoder_pre').units
    decoder = GRU(gru_units, return_sequences=True, name='decoder', stateful=True)(decoder_input)
    output = model.get_layer('output')(decoder)
    decoder_model = tf.keras.Model(decoder_input, output)
    
    # Copie des poids
    decoder_model.get_layer('decoder').set_weights(model.get_layer('decoder').get_weights())

    smiles = []
    for _ in range(num_attempts):
        # Correction finale de la shape des états
        initial_state = latent_to_state(latent_vec, model)
        initial_state = np.squeeze(initial_state, axis=0) if initial_state.shape[0] == 1 else initial_state
        
        # Méthode garantie pour Keras 3
        decoder_layer = decoder_model.get_layer('decoder')
        decoder_layer.reset_states()
        
        # Assignation avec reshape final
        target_shape = decoder_layer.states[0].shape
        reshaped_state = np.reshape(initial_state, target_shape)
        decoder_layer.states[0].assign(reshaped_state)
        
        # Génération du SMILES (le reste reste inchangé)
        current_char = np.zeros((1, 1, vocab_size))
        current_char[0, 0, char_to_num['!']] = 1
        result = [current_char[0, 0]]
        
        while True:
            pred = decoder_model.predict(current_char, verbose=0)
            next_char = softmax_to_one_hot(pred[0], temp=temp)
            result.append(next_char[0])
            
            if np.argmax(next_char) == char_to_num['E'] or len(result) > max_length:
                break
                
            current_char = np.expand_dims(next_char, axis=0)
        
        smiles.append(one_hot_decode(result))
    
    return smiles

if __name__ == "__main__":
    # try:
        # Charger votre modèle autoencodeur
        autoencoder_model = load_autoencoder_model('model_e300.keras')
        
        # Récupérer la taille latente pour l'agent DQN
        encoder_output = autoencoder_model.get_layer('encoder').output
        LATENT_SIZE = encoder_output.shape[-1]
        
        ACTION_SIZE = 5
        TARGET_SCORE = -15.0
        
        # Initialiser l'agent DQN
        agent = DQNAgent(LATENT_SIZE, ACTION_SIZE)
        
        # Fonction de scoring de docking (simulée)
        def docking_scorer(smile):
            with open(f"tmp.smiles", "w") as f:
                f.write(f"{smile}")
            os.system(f'obabel tmp.smiles -O pmol.pdb --gen3d best -p 7.4 --canonical --ff MMFF94 > /dev/null 2>&1') # > /dev/null 2>&1
            os.remove("tmp.smiles")
            if os.path.exists(f"pmol.pdb"):
                os.system(f'python prepare_ligand4.py -l pmol.pdb -o pmol.pdbqt -U nphs_lps') # -v = verbose (pour voir les détails)
                os.remove(f"pmol.pdb")
            else:
                print(f"Erreur : Le fichier pmol.pdb n'existe pas.")
            ligand_pdbqt = f'pmol.pdbqt'
            
            if not os.path.exists(ligand_pdbqt):
                print(f"Error: File {ligand_pdbqt} not found")
                return 0
            # Créer le fichier de configuration pour le docking
            with open(f"config.txt", "w") as f:
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
            vina_command = f'vina --config config.txt --out output.pdbqt > /dev/null 2>&1' # > /dev/null 2>&1 : rends la commande silencieuse
            run_in_shell(vina_command)

            # Vérifier si le fichier de sortie de vina existe
            output_pdbqt = f'output.pdbqt'
            if not os.path.exists(output_pdbqt):
                print(f"Error: File {output_pdbqt} not found")
                return 0

            # Utiliser Open Babel pour convertir les résultats d'AutoDock Vina de pdbqt en pdb
            obabel_command = f'obabel -ipdbqt {output_pdbqt} -opdb -O output.pdb -m > /dev/null 2>&1' # > /dev/null 2>&1
            run_in_shell(obabel_command)

            # Lire le fichier de sortie pour obtenir les résultats de Vina
            output_pdb = f'output1.pdb'
            if os.path.exists(output_pdb):
                with open(output_pdb, 'r') as f:
                    for line in f:
                        if line.startswith("REMARK VINA RESULT"):
                            words = line.split()
                            result = float(words[3])
                            print(f"{smile}\nresult : {result}")
            else:
                print(f"Error: File {output_pdb} not found")

            # Supprimer les fichiers temporaires
            os.remove(f'config.txt')
            os.remove(output_pdbqt)

            for k in range(2, 10):
                try:
                    os.remove(f'output{k}.pdb')
                except FileNotFoundError:
                    pass
            return result
        
        # Initialiser l'optimiseur
        optimizer = SMILESOptimizer(agent, autoencoder_model, docking_scorer, TARGET_SCORE)
        
        # SMILES initial
        if '.' not in ref_smile:
            initial_smiles =ref_smile
        else:
            initial_smiles = "CCO"
        
        # Lancer l'optimisation
        best_smiles, best_score = optimizer.optimize(autoencoder_model, initial_smiles, max_episodes=200)
        
        print("\nOptimisation terminée!")
        print(f"Meilleur SMILES trouvé: {best_smiles}")
        print(f"Meilleur score: {best_score}")
        
        # Sauvegarder le modèle DQN
        agent.save("dqn_smiles_optimizer")
        
    # except Exception as e:
    #     print(f"Une erreur est survenue: {str(e)}")
    #     print("Vérifiez que:")
    #     print("- Votre modèle autoencodeur est au format .keras")
    #     print("- Votre modèle contient bien une couche nommée 'encoder'")
    #     print("- Les fonctions smile_to_latent et latent_to_smile sont correctement définies")