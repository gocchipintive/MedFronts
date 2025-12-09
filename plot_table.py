#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# === SETTINGS ===
VARS = ['P1l', 'P2l', 'P3l', 'P4l', 'Total']  # now includes Total
#VARS = ['N1p','N3n']
INPUT_DIR = "RESULTS"
OUTPUT_DIR = "FIGS"
LABELS = ['Diatoms', 'Nanoflagellates', 'Picoplankton', 'Dinoflagellates']
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === COLOR FUNCTION ===
def value_to_color(val):
    """Map value ranges to a qualitative color scale"""
    if pd.isna(val):
        return (1, 1, 1)
    if val > 100:
        return mcolors.to_rgb("#800000")  # dark red
    elif val > 50:
        return mcolors.to_rgb("#ff6666")  # light red
    elif val > 20:
        return mcolors.to_rgb("#ffb366")  # orange
    elif val > 0:
        return mcolors.to_rgb("#fff2cc")  # light yellow
    elif val < -100:
        return mcolors.to_rgb("#000080")  # dark blue
    elif val < -50:
        return mcolors.to_rgb("#3399ff")  # medium blue
    elif val < -20:
        return mcolors.to_rgb("#99ccff")  # light blue
    elif val < 0:
        return mcolors.to_rgb("#cce5ff")  # pale blue
    else:
        return (1, 1, 1)

# === LOOP OVER VARIABLES ===
for var,lab in zip(VARS,LABELS):
    csv_file = os.path.join(INPUT_DIR, f"{var}_seasonal_percent_change.csv")
    if not os.path.exists(csv_file):
        print(f"⚠️ File not found: {csv_file}")
        continue

    output_png = os.path.join(OUTPUT_DIR, f"{var}_colored_table.png")
    title = f"{lab} – Seasonal Percent Change (Front vs No Front)"

    # === READ DATA ===
    df = pd.read_csv(csv_file, index_col=0).round(1)

    # Expected column order
    weak_vals_cols = df.columns[:4]
    weak_sig_cols  = df.columns[4:8]
    strong_vals_cols = df.columns[8:12]
    strong_sig_cols  = df.columns[12:16]

    # Split into values and significance
    df_values = pd.concat([df[weak_vals_cols], df[strong_vals_cols]], axis=1)
    df_sig    = pd.concat([df[weak_sig_cols],  df[strong_sig_cols]], axis=1)

    # === PLOT TABLE ===
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    table = ax.table(
        cellText=df_values.values,
        rowLabels=df_values.index,
        colLabels=df_values.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.3)

    # Apply colors + hatching for non-significant
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor("#f2f2f2")
        else:
            try:
                val = float(df_values.iloc[row-1, col])
                sig = int(df_sig.iloc[row-1, col])
            except Exception:
                val = None
                sig = 1
            cell.set_facecolor(value_to_color(val))
            if sig == 0:
                cell.set_hatch('///')
                cell.set_edgecolor('black')

    # Title
    ax.set_title(title, fontsize=11, pad=20, weight="bold")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_png}")

print("\n All tables (P1–P4 + Total) plotted successfully!")

