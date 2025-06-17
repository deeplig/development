#!/bin/bash

#SBATCH --job-name=autoencodeur_CRCI2NA_TEAM12        # Name for your job
#SBATCH --comment="Mise à jour du modèle pour l'autoencodeur"  # Comment for your job

#SBATCH --output=%x_%j.out      # Output file
#SBATCH --error=%x_%j.err       # Error file

#SBATCH --time=7-00:00:00       # Time limit
#SBATCH --nodes=1               # How many nodes to run on
#SBATCH --ntasks=1              # How many tasks per node
#SBATCH --cpus-per-task=2    # Number of CPUs per task
#SBATCH --mem-per-cpu=128G       # Memory per CPU
#SBATCH --qos=long           # priority/quality of service

# Command to run
python3 autoencodeur_new.py . seed.txt       
