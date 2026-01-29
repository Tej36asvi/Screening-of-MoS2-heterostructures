# Screening-of-MoS2-heterostructures

https://docs.google.com/document/d/1lvV7QIbScm99bK4lc8aQY0MX3Eyl3xTEAhzkrEUD3pc/edit?usp=sharing

Usage Guidelines
This project is an undergraduate exploratory study designed to demonstrate the application of machine learning in materials science. Please adhere to the following rules when using this code:

Educational Purpose: This code is intended for educational and exploratory purposes only. The results (predicted bandgaps, IP, EA) are based on a surrogate machine learning model (MAE ~0.4 eV) and should be verified with rigorous DFT calculations before being used for experimental fabrication.

Data Dependency: You must have the C2DB database file (c2db.db) in the root directory for the training script to work.

Attribution: If you use this workflow or code for your own project, please credit this repository.

How to Run the Code
The workflow is divided into three sequential stages. You must run them in the exact order listed below.

Prerequisites

Install the required Python libraries before starting:

Bash
pip install pandas numpy xgboost matplotlib joblib pymatgen matminer ase scikit-learn
Step 1: Training the Model

Script: 1_TrainingModel.py

Description: Scans the c2db.db database, extracts electronic properties, converts chemical data into numerical features using matminer, and trains an XGBoost Regressor.

Output: Saves the trained model to model.pkl.

Command:

Bash
python 1.TrainingModel.py
Step 2: Screening and Classification

Script: 2.Screener.py

Description: Loads the trained model.pkl and screens the entire database. It applies physics-based filters (Stability < 0.2 eV/atom, Strain < 15%) and classifies materials into Solar Absorbers (Type-II) or Gate Dielectrics (Type-I).

Output: Generates a shortlist of candidates in final_candidates.csv.

Command:

Bash
python 2.Screener.py
Step 3: Visualization

Script: 3.Visualise.py

Description: Reads final_candidates.csv and generates publication-quality band alignment diagrams for the top 3 candidates in each category.

Output: Saves high-resolution images:

1_solar_stack.png (Includes charge transfer arrows)

2_dielectric_stack.png (Shows band barriers)

Command:

Bash
python 3.Visualise.py
