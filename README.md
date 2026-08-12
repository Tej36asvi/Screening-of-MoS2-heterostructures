# Screening of MoS2 heterostructures

This repository contains an undergraduate exploratory workflow that applies machine learning to screen MoS2 based heterostructures for materials science use cases.

## Project reference

Detailed project notes:
https://docs.google.com/document/d/1lvV7QIbScm99bK4lc8aQY0MX3Eyl3xTEAhzkrEUD3pc/edit?usp=sharing

## Usage guidelines

- Educational use only. Predicted bandgaps, ionization potential, and electron affinity come from a surrogate model with MAE around 0.4 eV and should be verified with DFT before experimental use.
- Data requirement. The `c2db.db` database file must be present in the repository root.
- Attribution. If you use this workflow, please credit this repository.

## Workflow overview

Run the scripts in this exact order.

### Prerequisites

Install dependencies:

```bash
pip install pandas numpy xgboost matplotlib joblib pymatgen matminer ase scikit-learn
```

### Step 1 Training the model

- Script: `1_TrainingModel.py`
- What it does: Scans `c2db.db`, extracts electronic properties, converts chemistry data into numeric features using matminer, and trains an XGBoost regressor.
- Output: `model.pkl`
- Run:

```bash
python 1_TrainingModel.py
```

### Step 2 Screening and classification

- Script: `2_Screener.py`
- What it does: Loads `model.pkl`, screens the database, applies filters (stability less than 0.2 eV per atom and strain less than 15 percent), and classifies candidates as Type II solar absorbers or Type I gate dielectrics.
- Output: `final_candidates.csv`
- Run:

```bash
python 2_Screener.py
```

### Step 3 Visualization

- Script: `3_Visualise.py`
- What it does: Reads `final_candidates.csv` and generates band alignment diagrams for the top 3 candidates in each class.
- Output: `1_solar_stack.png` and `2_dielectric_stack.png`
- Run:

```bash
python 3_Visualise.py
```

## Files

- `c2db.db` Input database file. Not included in this repository.
- `model.pkl` Trained model from Step 1.
- `final_candidates.csv` Screening results from Step 2.
- `*.png` Generated plots from Step 3.
