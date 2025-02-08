import os
import glob
import pandas as pd
import numpy as np

# Pfade zu den übergeordneten Ordnern der Bio- und Temp-Daten
bio_parent_dir = 'data/biosignals_filtered'     
temp_parent_dir = 'data/temperature'   
output_dir = 'data/combined'

# Sicherstellen, dass der Ausgabeordner existiert
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Suche alle CSV-Dateien im Bio-Ordner (rekursiv) mit dem Suffix "_bio.csv"
bio_files = glob.glob(os.path.join(bio_parent_dir, '**', '*_bio.csv'), recursive=True)

for bio_file in bio_files:
    # Bestimme den gemeinsamen Namensanteil (ohne den Suffix "_bio")
    basename = os.path.basename(bio_file)
    base, ext = os.path.splitext(basename)
    common_name = base.replace('_bio', '')
    
    # Erstelle das Suchmuster für die zugehörige Temp-Datei im Temp-Ordner
    pattern = os.path.join(temp_parent_dir, '**', f'{common_name}_temp{ext}')
    temp_files = glob.glob(pattern, recursive=True)
    
    if not temp_files:
        print(f"Keine passende Temp-Datei gefunden für {bio_file}")
        continue  # Falls keine Temp-Datei vorhanden ist, überspringe diese Datei
    
    # Bei mehreren Treffern wird die erste Temp-Datei gewählt
    temp_file = temp_files[0]
    
    # Lese die Dateien ein. Da in den Quelldateien alle Variablen in einem einzigen Feld stehen
    # und per Tab getrennt sind (sowohl in der Kopfzeile als auch in den Daten), wird sep="\t" genutzt.
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

    # Überprüfe, ob die erforderlichen Spalten vorhanden sind
    if 'time' not in bio_df.columns:
        print(f"Spalte 'time' fehlt in {bio_file}. Datei wird übersprungen.")
        continue
    if 'time' not in temp_df.columns or 'temperature' not in temp_df.columns:
        print(f"Erforderliche Spalten fehlen in {temp_file}. Datei wird übersprungen.")
        continue
    
    # Extrahiere die Zeitstempel und Temperaturwerte
    bio_times = bio_df['time'].values
    temp_times = temp_df['time'].values
    temp_values = temp_df['temperature'].values
    
    # Sicherstellen, dass die Temp-Zeitstempel sortiert sind (für die lineare Interpolation)
    if not np.all(np.diff(temp_times) >= 0):
        sort_idx = np.argsort(temp_times)
        temp_times = temp_times[sort_idx]
        temp_values = temp_values[sort_idx]
    
    # Interpolation: Für jeden Zeitpunkt in bio_times wird ein interpolierter Temperaturwert
    # aus den Temp-Daten errechnet. Es wird von einem linearen Verlauf zwischen den Messwerten ausgegangen.
    interpolated_temps = np.interp(bio_times, temp_times, temp_values)
    
    # Erstelle den kombinierten Datensatz: Alle Spalten der Bio-Daten bleiben unverändert,
    # zusätzlich wird die interpolierte Temperatur in der neuen Spalte 'temp_adj' hinzugefügt.
    combined_df = bio_df.copy()
    combined_df['temp_adj'] = interpolated_temps
    
    # Speichere den kombinierten Datensatz. Durch to_csv werden die Felder in getrennten Spalten gespeichert,
    # sodass die weitere Verarbeitung (z.B. in Excel) vereinfacht wird.
    output_file = os.path.join(output_dir, f'{common_name}_combined.csv')
    combined_df.to_csv(output_file, index=False)
    
    print(f"Kombinierte Datei gespeichert: {output_file}")

