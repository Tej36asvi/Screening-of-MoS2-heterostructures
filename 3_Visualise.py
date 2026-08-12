import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- CONFIGURATION ---
INPUT_FILE = "final_candidates.csv"
MOS2_DATA = {
    "Material": "MoS2",
    "Role": "Reference",
    "Strain": "0%",
    "IP": 5.9,
    "EA": 4.1
}

def plot_single_category(category_key, title, filename, df, show_arrows=False):
    """
    Generates a single plot for a specific category and saves it as a separate file.
    """
    # 1. Filter Top 3
    if category_key == "Solar":
        subset = df[df['Role'].str.contains("Solar")]
    else:
        subset = df[df['Role'] == category_key]
        
    candidates = subset.sort_values(by="Strain_Val").head(3).to_dict('records')
    stack = [MOS2_DATA] + candidates
    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    spacing = 1.0
    bar_width = 0.5
    
    colors = {
        "Reference": "#FFD700",       # Gold
        "Solar": "#32CD32",           # Green
        "Gate Dielectric": "#D3D3D3", # Grey
    }

    # Vacuum Line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(0, 0.2, "Vacuum Level (0 eV)", fontsize=10, fontstyle='italic')
    
    # Store coords for arrows
    band_coords = []

    for i, mat in enumerate(stack):
        x = i * spacing
        
        # Color Logic
        if i == 0: role_color = "Reference"
        elif "Solar" in mat['Role']: role_color = "Solar"
        else: role_color = mat['Role']
        color = colors.get(role_color, "purple")
        
        cbm = -mat['EA']
        vbm = -mat['IP']
        band_coords.append({"x": x, "cbm": cbm, "vbm": vbm})
        
        # --- DRAW BANDS ---
        # Semiconductor Bands (No Metal Logic Needed)
        rect_cb = patches.Rectangle((x - bar_width/2, cbm), bar_width, 1.5, 
                                  facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect_cb)
        
        rect_vb = patches.Rectangle((x - bar_width/2, vbm - 1.5), bar_width, 1.5, 
                                  facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect_vb)
        
        # Labels
        ax.text(x, cbm + 0.1, f"{cbm:.2f}", ha='center', fontsize=8, fontweight='bold')
        ax.text(x, vbm - 0.3, f"{vbm:.2f}", ha='center', fontsize=8, fontweight='bold')
        gap = mat['IP'] - mat['EA']
        ax.text(x, (cbm+vbm)/2, f"Eg={gap:.2f}", ha='center', fontsize=8)

        # --- TEXT LABELS ---
        ax.text(x, -9.0, mat['Material'], ha='center', fontweight='bold', fontsize=11)
        if i == 0: sub_label = "MoS2\n(Substrate)"
        else: sub_label = f"#{i} Match\n{mat['Strain']}"
        ax.text(x, -10.0, sub_label, ha='center', fontsize=9, color='dimgrey')

    # --- ARROWS (Solar Only) ---
    if show_arrows and len(stack) >= 2:
        ref = band_coords[0]
        cand = band_coords[1]
        
        # Electron Flow (Downhill)
        if ref['cbm'] > cand['cbm']: start, end = ref, cand
        else: start, end = cand, ref
        
        ax.annotate("", xy=(end['x'] - 0.25, end['cbm'] + 0.8), xytext=(start['x'] + 0.25, start['cbm'] + 0.8),
                    arrowprops=dict(arrowstyle="->", color='blue', lw=2, linestyle='--'))
        ax.text((start['x'] + end['x'])/2, max(start['cbm'], end['cbm']) + 1.0, "e⁻ Flow", 
                ha='center', color='blue', fontweight='bold')

        # Hole Flow (Uphill)
        if ref['vbm'] < cand['vbm']: start, end = ref, cand
        else: start, end = cand, ref
        
        ax.annotate("", xy=(end['x'] - 0.25, end['vbm'] - 0.8), xytext=(start['x'] + 0.25, start['vbm'] - 0.8),
                    arrowprops=dict(arrowstyle="->", color='red', lw=2, linestyle='--'))
        ax.text((start['x'] + end['x'])/2, min(start['vbm'], end['vbm']) - 1.2, "h⁺ Flow", 
                ha='center', color='red', fontweight='bold')

    # Final Polish
    ax.set_ylim(-11, 2)
    ax.set_xlim(-0.6, len(stack) * spacing - 0.4)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_ylabel("Energy vs Vacuum (eV)", fontsize=12)
    ax.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close()

def main():
    print("Generating 2 Separate Images (Solar & Dielectric)...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    df['Strain_Val'] = df['Strain'].str.rstrip('%').astype(float)
    
    # 1. SOLAR (With Arrows)
    plot_single_category(
        "Solar", 
        "Solar Absorber Candidates (Type-II Staggered)", 
        "1_solar_stack.png", 
        df, 
        show_arrows=True
    )
    
    # 2. DIELECTRIC (No Arrows)
    plot_single_category(
        "Gate Dielectric", 
        "Gate Dielectric Candidates (Insulating Barrier)", 
        "2_dielectric_stack.png", 
        df, 
        show_arrows=False
    )

if __name__ == "__main__":
    main()