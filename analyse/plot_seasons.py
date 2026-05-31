#!/usr/bin/env python3
import sys, re, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LOG_PATH = Path(sys.argv[1])
RUN_DIR = LOG_PATH.parent

SEASON_OFFSET = 1.0
SEASON_AMPLITUDE = 0.5
SEASON_PERIOD = 500.0

pattern = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+kills=\s*(\d+)"
)

rows = []
for line in LOG_PATH.read_text(errors="replace").splitlines():
    m = pattern.match(line)
    if m:
        rows.append(tuple(map(int, m.groups())))

if not rows:
    raise SystemExit("No matching tick lines found.")

df = pd.DataFrame(
    rows,
    columns=["tick", "count", "alive", "move", "eat", "attack", "reproduce", "kills"],
)

cfg_path = RUN_DIR / "run_config.json"
if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text())
    base_food = int(cfg["runtime_args"]["food_spawn_quantity"])
else:
    base_food = 3000

df["food_spawn"] = np.rint(
    base_food
    * np.maximum(
        0.0,
        SEASON_OFFSET
        + SEASON_AMPLITUDE * np.sin(2.0 * np.pi * df["tick"] / SEASON_PERIOD),
    )
).astype(int)

plt.figure(figsize=(18, 9), dpi=130)

for col in ["alive", "food_spawn", "move", "eat", "attack", "reproduce", "kills"]:
    plt.plot(df["tick"], df[col], linewidth=2, label=col)

plt.title(f"Simulation dynamics: {RUN_DIR.name}", fontsize=18)
plt.xlabel("tick", fontsize=13)
# plt.ylabel("count / food units", fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(ncol=4, fontsize=11)
plt.tight_layout()

out = RUN_DIR / "run_curves.png"
plt.savefig(out)
print(f"Saved: {out}")