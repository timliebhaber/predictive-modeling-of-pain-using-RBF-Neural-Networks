import os
import glob
import pandas as pd
import numpy as np

bio_parent_dir = 'data/biosignals_filtered'     
temp_parent_dir = 'data/temperature'   
output_dir = 'data/combined'

# Suche alle Dateien mit "_bio.csv"
bio_files = glob.glob(os.path.join(bio_parent_dir, '**', '*_bio.csv'), recursive=True)

for bio_file in bio_files:
    # Bestimme den gemeinsamen Namensanteil
    basename = os.path.basename(bio_file)
    base, ext = os.path.splitext(basename)
    common_name = base.replace('_bio', '')
    
    pattern = os.path.join(temp_parent_dir, '**', f'{common_name}_temp{ext}')
    temp_files = glob.glob(pattern, recursive=True)
    
    if not temp_files:
        print(f"Keine passende Temp-Datei gefunden für {bio_file}")
        continue  # Falls keine Temp-Datei vorhanden ist, überspringe diese Datei
    
    temp_file = temp_files[0]
    
    try:
        bio_df = pd.read_csv(bio_file, sep="\t")
    except Exception as e:
        print(f"Fehler beim Lesen der Bio-Datei {bio_file}: {e}")
        continue
    try:
        temp_df = pd.read_csv(temp_file, sep="\t")
        print("Bio-Datei Spalten:", bio_df.columns.tolist())
        print("Temp-Datei Spalten:", temp_df.columns.tolist())

    except Exception as e:
        print(f"Fehler beim Lesen der Temp-Datei {temp_file}: {e}")
        continue

    # Überprüfe, ob time da ist
    if 'time' not in bio_df.columns:
        print(f"Spalte 'time' fehlt in {bio_file}. Datei wird übersprungen.")
        continue
    if 'time' not in temp_df.columns or 'temperature' not in temp_df.columns:
        print(f"Erforderliche Spalten fehlen in {temp_file}. Datei wird übersprungen.")
        continue
    
    # Extrahiere Zeit und Temperatur
    bio_times = bio_df['time'].values
    temp_times = temp_df['time'].values
    temp_values = temp_df['temperature'].values
    
    # Sortierung prüfen
    if not np.all(np.diff(temp_times) >= 0):
        sort_idx = np.argsort(temp_times)
        temp_times = temp_times[sort_idx]
        temp_values = temp_values[sort_idx]
    
    #Numpy Interpolation
    interpolated_temps = np.interp(bio_times, temp_times, temp_values)
    
    # Erstelle kombinierten Datensatz
    combined_df = bio_df.copy()
    combined_df['temp_adj'] = interpolated_temps
    
    output_file = os.path.join(output_dir, f'{common_name}_combined.csv')
    combined_df.to_csv(output_file, index=False)
    
    print(f"Kombinierte Datei gespeichert: {output_file}")

