# ===========================================================================================================
# Ce programme à pour but de transformer des SMILES (Simplified Molecular Input Line Entry System)
# en vecteur pour y apporter des modification aléatoire (mutation) afin d'optimiser l'affinité à un recepteur.
# les ligands mutés générés seront ensuite transmis au docker.
# ===========================================================================================================
# Useful links:
# https://www.kaggle.com/code/art3mis/220221-getting-started-with-smiles
# https://machinelearningmastery.com/gentle-introduction-generative-long-short-term-memory-networks/
# https://www.cheminformania.com/master-your-molecule-generator-seq2seq-rnn-models-with-smiles-in-keras/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # supprime les message d'info de TensorFlow : 0 = all logs, 3 = errors only
import sys
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.random import seed
seed(3)
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, Lipinski, AllChem, Fragments, rdFingerprintGenerator


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from tqdm import tqdm

# Module de DeepLearning
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Dropout, GRU, GaussianNoise
from keras.optimizers import Adam
tf.random.set_seed(3)


if len(sys.argv) != 3:
    print(f"Usage: python autoencoder.py <nom_dossier> <nom_fichier>\n"
          f"  <nom_dossier> : dossier dans lequel vous souhaitez extraire les résultats,\n" 
          f"    utiliser '.' si vous souhaitez rester dans  le dossier courant,\n"
          f"    dans le cas ou le dossier n'existe pas, il sera créer.\n"
          f"  <nom_fichier> : fichier 'seed' en format . txt contenant le SMILES du ligand\n")
    sys.exit(1)  # Quitter le script si les arguments ne sont pas correctement fournis

directory_name = sys.argv[1]
seed_file = sys.argv[2]

# Créer le dossier si celui-ci n'existe pas
directory_path = os.path.join(directory_name)  # Assure un chemin valide
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
    print(f"Dossier '{directory_name}' créé pour ce projet.")
else:
    print(f"Le dossier '{directory_name}' existe déjà.")

# Lire le fichier et transformer son contenu en une seule chaîne de caractères
try:
    with open(seed_file, 'r') as file:
        seed = ''.join([line.strip() for line in file])
    print("Votre graine est plantée !.")
    print(f"Et son contenu est: {seed}")
except Exception as e:
    print(f"Erreur lors de la lecture du fichier : {e}")


print(f"TendorFlow version : {tf.__version__}") # Version de TensoFlow
print(f"RDkit version : {rdkit.__version__}") # Versionde RDkit


with open('chembl_35_clean.smi') as f:  # Read SMI file into list on local
    smiles = f.readlines()
smiles = [item.split('\n')[0] for item in smiles]  # Split by linebreak


#print("Contenu de smiles après extend:", smiles)  # Vérifiez le contenu après extend

n = 2250000 # Size of train + val set min_length = 0 
max_length = 150 # Upper limit on length of SMILES sequence in train set
min_length = 0 # Lower limit on length of SMILES sequence in train set 

smiles = [smi for smi in smiles if len(smi) <= max_length and len(smi) >= min_length]
print(len(smiles))

# Charger les indices des chaînes SMILES
if os.path.exists("indices_v2.npy"): # Charge le fichier s'il existe déjà
    print('Loading saved indices...')
    indices = np.load("indices_v2.npy")
else:
    print('Sampling new indices...')
    print(f"{len(smiles)}")
    indices = random.sample(range(len(smiles)), n)
    np.save("indices_v2.npy", indices)
smiles = [smiles[i] for i in indices]

# Dictionnaire de tokenisation =============================================================================================
# Tokenisation : Division de l'information (Chaîne de caractère) en plus petites unités (tokens ou jeton en français).

unique_chars = list(set(''.join(smiles)))  # Obtenir tous les caractères uniques dans la liste SMILES
unique_chars.sort()
char_to_num = {char: i + 2 for i, char in enumerate(unique_chars)}  # Associer chaque caractère unique à un numéro
char_to_num['!'] = 0  # Ajouter le jeton de début au dictionnaire
char_to_num['E'] = 1  # Ajouter le jeton de fin au dictionnaire

# Ajouter un jeton pour les caractères inconnus
char_to_num['UNK'] = len(char_to_num)

# Vérifier les dimensions du dictionnaire
print(f'{len(char_to_num)} characters in dictionary.')
print(f"{unique_chars}\n")

# ============================================================================================================================

# Dictionnaire de dé-tokenisation
num_to_char = {v: k for k, v in char_to_num.items()}
vocab_size = len(char_to_num)  # Nombre de caractères uniques dans le vocabulaire

# Définir des fonctions pour traiter le jeu de données avant de le charger dans le modèle ========================================

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

# =================================================================================================================================

# Créer des ensembles d'entraînement et de validation pour le modèle
if os.path.exists('train_indices_v2.npy'):  # Charger les ensembles de données enregistrés
    print('Loading saved train indices...')
    train_indices = np.load('train_indices_v2.npy')
else:
    # Diviser les ensembles d'entraînement et de validation
    # Sélectionner des indices aléatoires pour les ensembles d'entraînement et de validation
    train_indices = random.sample(range(len(smiles)), int(0.9 * len(smiles)))
    # Enregistrer les indices d'entraînement
    np.save('train_indices_v2.npy', train_indices)

val_indices = list(set(range(len(smiles))) - set(train_indices))
train_smiles = [smiles[i] for i in tqdm(train_indices) if Chem.MolFromSmiles(smiles[i])]  # Filtrer les SMILES compatibles avec rdkit
val_smiles = [smiles[i] for i in tqdm(val_indices) if Chem.MolFromSmiles(smiles[i])]  # Filtrer les SMILES compatibles avec rdkit
train_mols = [Chem.MolFromSmiles(smile) for smile in tqdm(train_smiles)]

# Créer des jeux de données one-hot encodés
X_train = np.array([one_hot_encode(smile)[:-1] for smile in tqdm(train_smiles)])  # Contient la séquence + jeton de début
X_val = np.array([one_hot_encode(smile)[:-1] for smile in tqdm(val_smiles)])  # Contient la séquence + jeton de début
y_train = np.array([one_hot_encode(smile)[1:] for smile in tqdm(train_smiles)])  # Contient la séquence + jeton de fin
y_val = np.array([one_hot_encode(smile)[1:] for smile in tqdm(val_smiles)])  # Contient la séquence + jeton de fin

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


# Construction du modele =========================================================================================================

def build_model(model_name = None,
                recurrent_unit_size = 256,
                latent_size = 256,
                epochs = 20,
                batch_size = 256,
                noise_sd = 0.01,
                latent_activation = 'relu',
                verbose = 1):
  """
  Création du modèle de réseau neuronal
    Args :
      recurrent_unit_size = 256,
      latent_size = 256,
      epochs = 20,
      batch_size = 256,
      noise_sd = 0.01,
      latent_activation = 'relu',
      verbose = 1
    Return :
      model :
      history :
    Raise :
  """
  # Charge le modèle s'il existe déjà
  if os.path.exists(f'{model_name}.keras'):
    print('Loading saved model...')
    model = tf.keras.models.load_model(f'{model_name}.keras')

    print(model.summary())

    return model, None

  else:
    # Encoder layers
    input = Input(shape = (None, vocab_size), name = 'input')
    state = GRU(recurrent_unit_size, name = 'encoder_pre')(input) # GRU state and GRU output are the same
    latent = Dense(latent_size, name = 'encoder', activation = latent_activation)(state)
    latent_noise = tf.keras.layers.GaussianNoise(noise_sd)(latent) # Gaussian layer to make decoder more robust to small deviations in latent vector
    state = Dense(recurrent_unit_size, name = 'decoder_pre', activation = 'relu')(latent_noise)

    # Decoder layers
    decoder = GRU(recurrent_unit_size, return_sequences = True, name = 'decoder')(input, initial_state = [state]) # return_sequences set as True to return SMILES sequence
    output = Dense(vocab_size, activation = 'softmax', name = 'output')(decoder)

    # Assemblage du model
    model = Model(inputs = input, outputs = output)

    # Compilation du model
    model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['acc'])
    print(model.summary())

    # Train model
    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = verbose, validation_data = (X_val, y_val))

    # Plot training history
    fig, axs = plt.subplots(1, 2, figsize = (18, 6))
    fig.tight_layout()

    for i, metric in enumerate(['acc', 'loss']):
      axs[i].plot(history.history[metric], label='train')
      axs[i].plot(history.history[f'val_{metric}'], label='val')
      axs[i].set_title(metric.upper())
      axs[i].set_ylabel(metric)
      axs[i].set_xlabel('epoch')
      axs[i].legend()

    # Save model
    model.save(f'{model_name}.keras')

    return model, history.history

# Define encoder and decoder models ==================================================================================================

# Convertion des SMILES de format 'string' en vecteurs latents
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
  encoder_model = Model(inputs = model.get_layer('input').input, outputs = model.get_layer('encoder').output)

  # Predict latent vector(s) from encoder model
  latent_vec = encoder_model.predict(ohsmile[:, :-1, :], verbose = 0)

  return latent_vec

# Convert latent vector to SMILES string
def latent_to_smile(latent_vec, temp, num_attempts, model):
  """
  Convertion un vecteur latent en SMILES
    Args :
      latent vector :
      sampling temperature :
      num_attempts :
    Return :
      SMILES string
    Raise :
  """

  # Convert latent vector to a size compatible with decoder
  def latent_to_state(latent_vec, model):
    # Input: latent vector of size latent_size
    # Output: latent vector of size recurrent_unit_size

    input = Input(shape = (model.get_layer('encoder').output_shape[-1]))
    latent = model.get_layer('decoder_pre')(input)

    process_model = Model(inputs = input, outputs = latent)
    latent_vec_processed = process_model.predict(latent_vec, verbose = 0)

    return latent_vec_processed

  # Define decoder model
  decoder_input = Input(shape = (1, vocab_size), batch_size = 1, name = 'input')
  decoder = GRU(model.get_layer('encoder_pre').output_shape[-1], return_sequences = True, name = 'decoder', stateful = True)(decoder_input) # Stateful model because one char inputted at a time, and the chars of a sequence are related. In training several molecules are processed at a time so there is no need to preserve the states from the previous *independent* SMILES string.
  output = model.get_layer('output')(decoder)
  decoder_model = Model(decoder_input, output)

  decoder_model.get_layer('decoder').set_weights(model.get_layer('decoder').get_weights()) # Transfer weights from trained model because layer is redefined

  # Use decoder model to generate SMILES string from latent_vec
  smiles = []
  for i in range(num_attempts):
    # Predict SMILES string one character at a time

    decoder_model.get_layer('decoder').reset_states(states = latent_to_state(latent_vec, model)) # Pulling the states from convertor_model (resets_states stuffs the state numpy arrays into the LSTM layer)

    # Create start token to feed into model
    vec = np.zeros(vocab_size, dtype = np.uint8)
    vec[char_to_num['!']] = 1
    result = [vec] # Initialize results list
    vec = np.expand_dims(np.expand_dims(np.array(vec), 0), 0) # Make vec 3D to keep decoder model happy

    while np.argmax(np.expand_dims(result[-1], 0)) != char_to_num['E']:
      vec = np.expand_dims(softmax_to_one_hot(decoder_model.predict(vec, verbose = 0)[0], temp = temp), 0)
      result.append(vec[0][0])

    smiles.append(one_hot_decode(result))

  return smiles

# Define seed and random generation functions

# Call latent_to_smile multiple times from mutated versions of a latent seed
def generate_around(name, latent_seed, sd, model, num_attempts, num_attempts_per_latent, temp):
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

  num_success = 0
  valid_smiles = []
  valid_mols = []
  latent_size = model.get_layer('encoder').output_shape[-1]

  with open(name, "w") as file:
    pass

  for i in tqdm(range(num_attempts)):

    # Mutate seed
    mutated_latent_seed = latent_seed + sd * np.random.randn(latent_size)

    # Convert to smiles
    smiles = latent_to_smile(mutated_latent_seed, temp = temp, num_attempts = num_attempts_per_latent, model = model)

    for smile in smiles:
      # Check if smile valid
      molecule = Chem.MolFromSmiles(smile)
      if molecule:
        valid_smiles.append(smile)
        valid_mols.append(molecule)
        num_success += 1

        with open(name, 'r') as f:
          lines = f.readlines()

          if smile not in lines:
            with open(name, 'a') as f:
                f.writelines(smile + ",\n")

  # Calculate success rate
  success_rate = num_success/(num_attempts * num_attempts_per_latent)

  return num_success, success_rate, valid_smiles, valid_mols

# Call latent_to_smile multiple times from random latent seeds
def generate_random(span, model, num_attempts, num_attempts_per_latent, temp):

  num_success = 0
  valid_smiles = []
  valid_mols = []

  latent_size = model.get_layer('encoder').output_shape[-1]

  with open("random_ligands.txt", "w") as file:
    pass

  for i in tqdm(range(num_attempts)):
    # Generate random latent seed uniformly between 0 and 1
    latent_seed = np.random.uniform(low = span[0], high = span[1], size = (1, latent_size))

    # Get smile strings from seed
    smiles = latent_to_smile(latent_seed, temp = temp, num_attempts = num_attempts_per_latent, model = model)

    for smile in smiles:
      # Check if smile valid
      molecule = Chem.MolFromSmiles(smile)
      if molecule:
        valid_smiles.append(smile)
        valid_mols.append(molecule)
        num_success += 1

        with open(f'random_ligands.txt', 'r') as f:
          lines = f.readlines()

          if smile not in lines:
            with open(f'random_ligands.txt', 'a') as f:
                f.writelines(smile + ",\n")

  success_rate = num_success/(num_attempts * num_attempts_per_latent)

  return num_success, success_rate, valid_smiles, valid_mols

"""
# Créer un générateur préconfigurer (réutilisable) : C'est une mise à jour de -> AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits = 2048)
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=3,       # Rayon = 3 
    fpSize=2048    # Taille du bit vector = 2048 (remplace `nBits`)
)
trainset_df = []
seed = seed
mol1 = Chem.MolFromSmiles(seed)

fp1 = morgan_gen.GetFingerprint(mol1)

for mol in tqdm(train_mols):
  fp2 = morgan_gen.GetFingerprint(mol)
  trainset_df.append(DataStructs.TanimotoSimilarity(fp1, fp2))

def get_upper_quartile(values):
    sorted_values = sorted(values)
    n = len(sorted_values)

    # Calculate the value of the upper quartile
    upper_quartile_index = n * 0.99
    upper_quartile_value = sorted_values[int(upper_quartile_index)]

    return upper_quartile_value

cutoff = get_upper_quartile(trainset_df)
print(f'Tanimoto cutoff: {cutoff}')

plt.hist(trainset_df, bins = 1000)
plt.axvline(x=cutoff, color='r')

# Set the limit of x-axis
plt.xlim(0, 0.3)

plt.tick_params(axis='x', labelsize=16)

plt.tick_params(axis='y', labelsize=16)

plt.title('Tanimoto Similarity of Training Set', fontsize = 20)
plt.xlabel('Score', fontsize = 20)
plt.ylabel('Frequency', fontsize = 20)
#plt.show()
plt.savefig(f'{directory_name}/tanimoto.png')
# Prepare fine-tuned datasets

tuned_smiles = []
mol1 = Chem.MolFromSmiles(seed)
fp1 = morgan_gen.GetFingerprint(mol1)

for smile in tqdm(train_smiles):
  mol2 = Chem.MolFromSmiles(smile)
  fp2 = morgan_gen.GetFingerprint(mol2)
  if DataStructs.TanimotoSimilarity(fp1, fp2) >= cutoff:
    tuned_smiles.append(smile)

tuned_train_indices = random.sample(range(len(tuned_smiles)), int(0.9 * len(tuned_smiles)))
tuned_val_indices = list(set(range(len(tuned_smiles))) - set(tuned_train_indices))
tuned_train_smiles = [tuned_smiles[i] for i in tuned_train_indices if Chem.MolFromSmiles(tuned_smiles[i])] # filter for smiles compatible with rdkit
tuned_val_smiles = [tuned_smiles[i] for i in tuned_val_indices if Chem.MolFromSmiles(tuned_smiles[i])] # filter for smiles compatible with rdkit
tuned_train_mols = [Chem.MolFromSmiles(smile) for smile in tuned_train_smiles]

# # Create one-hot-encoded datasets
tuned_X_train = np.array([one_hot_encode(smile)[:-1] for smile in tuned_train_smiles]) # Contains sequence + start token
tuned_X_val = np.array([one_hot_encode(smile)[:-1] for smile in tuned_val_smiles]) # Contains sequence + start token
tuned_y_train = np.array([one_hot_encode(smile)[1:] for smile in tuned_train_smiles]) # Contains sequence + end token
tuned_y_val = np.array([one_hot_encode(smile)[1:] for smile in tuned_val_smiles]) # Contains sequence + end token

print("X_train shape:", tuned_X_train.shape)
print("y_train shape:", tuned_y_train.shape)
print("X_val shape:", tuned_X_val.shape)
print("y_val shape:", tuned_y_val.shape)

# Build fine-tuned model
def fine_tune(model_name = None,
                epochs = 100,
                batch_size = 256,
                verbose = 1
                ):

  print('Loading saved model...')
  model = tf.keras.models.load_model(f'{model_name}.keras')
  print(model.summary())

  # Train model
  history = model.fit(tuned_X_train, tuned_y_train, batch_size = batch_size, epochs = epochs, verbose = verbose, validation_data = (tuned_X_val, tuned_y_val))

  # Plot training history
  fig, axs = plt.subplots(1, 2, figsize = (18, 6))
  fig.tight_layout()

  for i, metric in enumerate(['acc', 'loss']):
    axs[i].plot(history.history[metric], label='train')
    axs[i].plot(history.history[f'val_{metric}'], label='val')
    axs[i].set_title(metric.upper())
    axs[i].set_ylabel(metric)
    axs[i].set_xlabel('epoch')
    axs[i].legend()

  # saving the plot
  fig.savefig(f"{directory_name}/model.png")

  # Save model
  model.save(f'tuned_{model_name}.keras')

  return model, history.history
"""
# Generate seed ligands before fine-tuning
print("Building model...")

# Valeur de test local : ------------------------------------------------------------------------------------------------------

# model1, history = build_model(model_name = f'model_e200', recurrent_unit_size = 100, latent_size = 64, epochs = 4, batch_size = 4, noise_sd = 0, latent_activation = 'sigmoid', verbose = 1)
# print("creating ligands")
# seed = seed
# seed_num_success, seed_success_rate, seed_valid_smiles, seed_valid_mols = generate_around(name = f"{directory_name}/pre_ligands.txt", latent_seed = smile_to_latent(seed, model1),
#                                                                                               sd = 0.01,
#                                                                                               model = model1,
#                                                                                               num_attempts = 4,
#                                                                                               num_attempts_per_latent = 1,
#                                                                                               temp = 1.0)

# tuned_model, history = build_model(model_name = f'tuned_model_e200', recurrent_unit_size = 100, latent_size = 64, epochs = 4, batch_size = 4, noise_sd = 0, latent_activation = 'sigmoid', verbose = 1)
# seed_num_success, seed_success_rate, seed_valid_smiles, seed_valid_mols = generate_around(name = f"{directory_name}/ligands.txt", latent_seed = smile_to_latent(seed, tuned_model),
#                                                                                               sd = 0.01,
#                                                                                               model = tuned_model,
#                                                                                               num_attempts = 4,
#                                                                                               num_attempts_per_latent = 1,
#                                                                                               temp = 1.0)
# print( "Vos modeles et ligands ont été créés avec succés... Le docking vous attend !")

# Valeur d'origine : ------------------------------------------------------------------------------------------------------------

model1, history = build_model(model_name = f'model_e300', recurrent_unit_size = 100, latent_size = 64, epochs = 70, batch_size = 256, noise_sd = 0, latent_activation = 'sigmoid', verbose = 1)
print("creating ligands")
# seed_num_success, seed_success_rate, seed_valid_smiles, seed_valid_mols = generate_around(name = f"{directory_name}/pre_ligands.txt", latent_seed = smile_to_latent(seed, model1),
#                                                                                               sd = 0.01,
#                                                                                               model = model1,
#                                                                                               num_attempts = 1000,
#                                                                                               num_attempts_per_latent = 1,
#                                                                                               temp = 1.0)

# Generate fine-tuned seed ligands

# tuned_model, tuned_history = fine_tune(model_name = f'model_e300', epochs = 60, batch_size = 256, verbose = 1)
# tuned_model, history = build_model(model_name = f'tuned_model_e300', recurrent_unit_size = 128, latent_size = 64, epochs = 65, batch_size = 256, noise_sd = 0.01, latent_activation = 'sigmoid', verbose = 1)
# seed_num_success, seed_success_rate, seed_valid_smiles, seed_valid_mols = generate_around(name = f"{directory_name}/ligands.txt", latent_seed = smile_to_latent(seed, tuned_model),
#                                                                                               sd = 0.01,
#                                                                                               model = tuned_model,
#                                                                                               num_attempts = 1000,
#                                                                                               num_attempts_per_latent = 1,
#                                                                                               temp = 1.0)
print( "Vos modeles et ligands ont été créés avec succés... Le docking vous attend !")