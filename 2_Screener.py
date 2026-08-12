import pandas as pd
import joblib
import os
import numpy as np
import warnings
from ase.db import connect
from pymatgen.core import Composition

# Settings
DB_FILE = "c2db.db"
MODEL_FILE = "model.pkl"
RESULTS_FILE = "final_candidates.csv"

# Reference values for MoS2
MOS2_LATTICE = 3.16
MOS2_IP = 5.9
MOS2_EA = 4.1

# Filters
MAX_EHULL = 0.20        
MAX_STRAIN = 0.15       

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    print("Starting Stage 2: Screening (Solar & Dielectric Only)...")

    if not os.path.exists(MODEL_FILE):
        print("Error: Model file not found.")
        return

    # --- Step 1: Load Model ---
    print("Step 1: Loading model...")
    saved_data = joblib.load(MODEL_FILE)
    model_ip = saved_data["model_ip"]
    model_ea = saved_data["model_ea"]
    feat_extractor = saved_data["featurizer"]
    geo_columns = saved_data["feature_columns"]

    # --- Step 2: Scan Database ---
    print("Step 2: Scanning database...")
    db = connect(DB_FILE)
    candidates = []
    
    for row in db.select():
        # Filter 1: Stability
        ehull = getattr(row, "ehull", 1.0)
        if ehull > MAX_EHULL: continue

        # Filter 2: Strain
        if hasattr(row, 'cell'): lat_a = np.linalg.norm(row.cell[0])
        else: lat_a = getattr(row, 'a', 0.0)

        if MOS2_LATTICE > 0: strain = abs(lat_a - MOS2_LATTICE) / MOS2_LATTICE
        else: strain = 1.0

        if strain > MAX_STRAIN: continue

        # Save candidate
        crystal = getattr(row, "crystal_system", getattr(row, "spgname", "Unknown"))
        mag = getattr(row, "magstate", "NM")
        vpa = row.volume / row.natoms if hasattr(row, 'volume') else 15.0

        candidates.append({
            "formula": row.formula,
            "lattice": lat_a,
            "Strain": strain,
            "CrystalSystem": crystal,
            "MagState": mag,
            "VolPerAtom": vpa,
            "EHull": ehull
        })

    if not candidates:
        print("No candidates found.")
        return

    # --- Step 3: Run AI ---
    print("Step 3: Running AI predictions...")
    df = pd.DataFrame(candidates)
    
    df["composition"] = df["formula"].apply(Composition)
    X_chem = feat_extractor.featurize_dataframe(df, col_id="composition", ignore_errors=True, pbar=False)
    X_chem = X_chem[feat_extractor.feature_labels()]
    
    X_geo = pd.get_dummies(df[["CrystalSystem", "MagState"]], prefix=["Sys", "Mag"])
    X_geo["VolPerAtom"] = df["VolPerAtom"]
    X_geo["EHull"] = df["EHull"]
    
    X_geo = X_geo.reindex(columns=geo_columns, fill_value=0)
    X = pd.concat([X_chem, X_geo], axis=1)

    df["Pred_IP"] = model_ip.predict(X)
    df["Pred_EA"] = model_ea.predict(X)
    df["Pred_Gap"] = df["Pred_IP"] - df["Pred_EA"]

    # --- Step 4: Classify (NO METALS) ---
    print("Step 4: Classifying results...")
    results = []
    
    for i, row in df.iterrows():
        role = None
        
        # A. SOLAR (Type-II Staggered)
        if (1.0 < row['Pred_Gap'] < 2.5) and (row['Strain'] < 0.05):
            if (row['Pred_EA'] > MOS2_EA) and (row['Pred_IP'] > MOS2_IP):
                role = "Solar (Type-II Lower)"
            elif (row['Pred_EA'] < MOS2_EA) and (row['Pred_IP'] < MOS2_IP):
                role = "Solar (Type-II Higher)"
        
        # B. DIELECTRIC (Strict Barrier)
        elif row['Pred_Gap'] > 4.0:
            cbo = MOS2_EA - row['Pred_EA']
            vbo = row['Pred_IP'] - MOS2_IP
            # Must block both electrons and holes
            if (cbo > 1.0) and (vbo > 1.0):
                role = "Gate Dielectric"

        # NOTE: Metal check deleted here.

        if role:
            results.append({
                "Material": row['formula'],
                "Role": role,
                "Bandgap": round(row['Pred_Gap'], 2),
                "Strain": f"{row['Strain']:.2%}",
                "Stability": round(row['EHull'], 3),
                "IP": round(row['Pred_IP'], 2),
                "EA": round(row['Pred_EA'], 2)
            })

    # --- Step 5: Save ---
    final_df = pd.DataFrame(results)
    
    if not final_df.empty:
        final_df = final_df.sort_values(by=["Role", "Strain"])
        final_df.to_csv(RESULTS_FILE, index=False)
        print(f"\nSaved {len(final_df)} Solar & Dielectric candidates to {RESULTS_FILE}")
    else:
        print("No candidates found.")

if __name__ == "__main__":
    main()