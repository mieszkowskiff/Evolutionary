import os
import subprocess
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Ustawienia
DIRECTORY = "."

def parse_filename(fname):
    # Dopasowuje grupy (np. 'AoS', 'tensor', 'original') i numery przebiegów
    match = re.match(r'([a-zA-Z]+)(\d+)\.sqlite', fname)
    if match:
        return match.group(1), int(match.group(2))
    return "unknown", 0

def collect_data():
    all_runs_data = []
    sqlite_files = [f for f in os.listdir(DIRECTORY) if f.endswith(".sqlite")]

    for file in sqlite_files:
        print(f"Processing {file}...")
        cmd = ["nsys", "stats", "--report", "nvtx_sum", "--format", "csv", file]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            row_data = {"File Name": file}
            
            csv_lines = output.strip().split('\n')
            start_idx = next((i for i, line in enumerate(csv_lines) if "Time (%)" in line), -1)
            
            if start_idx != -1:
                reader = csv.DictReader(csv_lines[start_idx:])
                for row in reader:
                    name = row.get("Range", "").strip().lstrip(':')
                    # POMIJANIE 'Tick'
                    if name == "Tick": continue
                    
                    try:
                        row_data[name] = int(row.get("Total Time (ns)", "0").replace(",", ""))
                    except ValueError:
                        continue
            all_runs_data.append(row_data)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    df = pd.DataFrame(all_runs_data).fillna(0)
    df[['Group', 'Run']] = df['File Name'].apply(lambda x: pd.Series(parse_filename(x)))
    return df

def plot_data(df):
    # Dynamiczne wyciąganie grup i przebiegów z danych
    groups = sorted(df['Group'].unique())
    runs = sorted(df['Run'].unique())
    components = [col for col in df.columns if col not in ['File Name', 'Group', 'Run']]

    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.2
    # Odstęp między grupami słupków
    group_spacing = len(runs) * bar_width + 0.5 
    x_positions = np.arange(len(groups)) * group_spacing
    colors = plt.cm.tab10(np.linspace(0, 1, len(components)))

    for run_idx, run in enumerate(runs):
        # Przesunięcie słupka względem środka grupy
        offset = (run_idx - (len(runs) - 1) / 2) * bar_width
        
        for i, group in enumerate(groups):
            row = df[(df['Group'] == group) & (df['Run'] == run)]
            if not row.empty:
                values = row[components].iloc[0].values
                bottoms = 0
                for comp_idx, val in enumerate(values):
                    ax.bar(x_positions[i] + offset, val, bar_width, bottom=bottoms, 
                           color=colors[comp_idx], edgecolor='white', linewidth=0.5,
                           label=components[comp_idx] if (run_idx == 0 and i == 0) else "")
                    bottoms += val
        
        # Podpisy numerów przebiegów pod grupami
        for i in range(len(groups)):
            ax.text(x_positions[i] + offset, -ax.get_ylim()[1]*0.03, str(run), 
                    ha='center', va='top', fontsize=9, color='gray')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups, fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Time (ns) excluding Tick')
    ax.set_title('Performance Analysis (Dynamic Groups)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('performance_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    data = collect_data()
    if not data.empty:
        plot_data(data)