import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
import os
import warnings
from ase.db import connect
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Settings
DB_FILE = "c2db.db"
MODEL_FILE = "model.pkl"
RANDOM_SEED = 42

# Ignore warnings to keep the output clean
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    print("Starting Stage 1: Training...")

    if not os.path.exists(DB_FILE):
        print(f"Error: {DB_FILE} not found.")
        return

    # --- Step 1: Get Data from Database ---
    print("Step 1: Extracting data from database...")
    
    db = connect(DB_FILE)
    valid_data = []
    
    for row in db.select():
        try:
            # We need these three values to calculate IP and EA
            if not (hasattr(row, 'evac') and hasattr(row, 'vbm') and hasattr(row, 'cbm')):
                continue
            
            # Calculate the target properties (IP and EA)
            ip = row.evac - row.vbm
            ea = row.evac - row.cbm
            
            # Get other useful properties
            # Use 'Unknown' or 1.0 if data is missing
            crystal = getattr(row, "crystal_system", getattr(row, "spgname", "Unknown"))
            mag = getattr(row, "magstate", "NM")
            ehull = getattr(row, "ehull", 1.0)
            
            # Calculate volume per atom
            if hasattr(row, 'volume') and hasattr(row, 'natoms'):
                vpa = row.volume / row.natoms
            else:
                vpa = 15.0

            valid_data.append({
                "formula": row.formula,
                "composition": Composition(row.formula),
                "CrystalSystem": crystal,
                "MagState": mag,
                "VolPerAtom": vpa,
                "EHull": ehull,
                "Target_IP": ip,
                "Target_EA": ea
            })
            
        except:
            continue

    df = pd.DataFrame(valid_data)
    print(f"Found {len(df)} materials with valid data.")

    # --- Step 2: Convert Chemistry to Numbers ---
    print("Step 2: Generating features...")
    
    # Use Matminer to get atomic properties (mass, radius, etc.)
    # n_jobs=1 is used to prevent errors on Mac
    feat_extractor = ElementProperty.from_preset(preset_name="magpie")
    X_chem = feat_extractor.featurize_dataframe(df, col_id="composition", ignore_errors=True, pbar=True,)
    X_chem = X_chem[feat_extractor.feature_labels()]

    # Convert text data (Crystal System) into numbers (0s and 1s)
    X_geo = pd.get_dummies(df[['CrystalSystem', 'MagState']], prefix=['Sys', 'Mag'])
    X_geo['VolPerAtom'] = df['VolPerAtom']
    X_geo['EHull'] = df['EHull']
    
    # Combine all features
    X = pd.concat([X_chem, X_geo], axis=1).fillna(0)

    # --- Step 3: Train the Model ---
    print("Step 3: Training the model...")
    
    y_ip = df['Target_IP']
    y_ea = df['Target_EA']
    
    # Save 10% of data for testing
    X_train, X_test, y_ip_train, y_ip_test, y_ea_train, y_ea_test = train_test_split(
        X, y_ip, y_ea, test_size=0.15, random_state=RANDOM_SEED
    )

    # Train for Ionization Potential (IP)
    model_ip = xgb.XGBRegressor(n_estimators=300, max_depth=6, )
    model_ip.fit(X_train, y_ip_train)
    
    # Train for Electron Affinity (EA)
    model_ea = xgb.XGBRegressor(n_estimators=300, max_depth=6, )
    model_ea.fit(X_train, y_ea_train)

    # --- Step 4: Check Accuracy ---
    print("Step 4: Checking accuracy...")
    
    test_pred_ip = model_ip.predict(X_test)
    error_ip = mean_absolute_error(y_ip_test, test_pred_ip)
    print(f"IP Error: {error_ip:.3f} eV")
    
    test_pred_ea = model_ea.predict(X_test)
    error_ea = mean_absolute_error(y_ea_test, test_pred_ea)
    print(f"EA Error: {error_ea:.3f} eV")

    # --- Step 5: Save ---
    print(f"Step 5: Saving model to {MODEL_FILE}...")
    
    # Save models and column names so Stage 2 can use them
    data_to_save = {
        "model_ip": model_ip,
        "model_ea": model_ea,
        "featurizer": feat_extractor,
        "feature_columns": X_geo.columns
    }
    
    joblib.dump(data_to_save, MODEL_FILE)
    print("Done.")

if __name__ == "__main__":
    main()