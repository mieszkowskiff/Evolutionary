#!/usr/bin/env python3
import sys, re, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if len(sys.argv) != 2:
    raise SystemExit("Usage: python plot_seasons.py /path/to/run.log")

LOG_PATH = Path(sys.argv[1])
RUN_DIR = LOG_PATH.parent

SEASON_OFFSET = 1.0
SEASON_AMPLITUDE = 0.5
SEASON_PERIOD = 500.0

# Old format:
# tick global_id population move eat attack reproduce kills= K
# New format:
# tick global_id population move eat attack reproduce drink kills= K
pattern = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    r"(?:\s+(\d+))?\s+kills=\s*(\d+)"
)

rows = []
for line in LOG_PATH.read_text(errors="replace").splitlines():
    m = pattern.match(line)
    if not m:
        continue

    tick, global_id, population, move, eat, attack, reproduce, maybe_drink, kills = m.groups()
    rows.append((
        int(tick),
        int(global_id),
        int(population),
        int(move),
        int(eat),
        int(attack),
        int(reproduce),
        int(maybe_drink) if maybe_drink is not None else 0,
        int(kills),
    ))

if not rows:
    raise SystemExit("No matching tick lines found.")

df = pd.DataFrame(
    rows,
    columns=["tick", "global_id", "population", "move", "eat", "attack", "reproduce", "drink", "kills"],
)

cfg_path = RUN_DIR / "run_config.json"
base_food = 3000
base_water = None
if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text())
    args = cfg.get("runtime_args", {})
    base_food = int(args.get("food_spawn_quantity", base_food))
    base_water = int(args.get("water_spawn_quantity", base_food)) if "water_spawn_quantity" in args else None

if base_water is None:
    base_water = base_food

seasonal_factor = np.maximum(
    0.0,
    SEASON_OFFSET + SEASON_AMPLITUDE * np.sin(2.0 * np.pi * df["tick"] / SEASON_PERIOD),
)

df["food_spawn"] = np.rint(base_food * seasonal_factor).astype(int)
df["water_spawn"] = np.rint(base_water * seasonal_factor).astype(int)

plt.figure(figsize=(20, 10), dpi=140)

series = [
    ("population", 3.0, "-"),
    ("food_spawn", 2.5, "--"),
    ("water_spawn", 2.5, ":"),
    ("move", 1.8, "-"),
    ("eat", 1.8, "-"),
    ("drink", 1.8, "-"),
    ("attack", 1.8, "-"),
    ("reproduce", 1.8, "-"),
    ("kills", 2.3, "-"),
]

for col, width, style in series:
    if col in df.columns and df[col].sum() > 0:
        plt.plot(df["tick"], df[col], linestyle=style, linewidth=width, label=col)

plt.title(f"Simulation dynamics: {RUN_DIR.name}", fontsize=20, pad=14)
plt.xlabel("tick", fontsize=14)
plt.ylabel("count / spawned resource units", fontsize=14)
plt.grid(True, alpha=0.28)
plt.legend(ncol=5, fontsize=11, frameon=True)
plt.tight_layout()

out = RUN_DIR / "run_curves.png"
plt.savefig(out)
print(f"Saved: {out}")
