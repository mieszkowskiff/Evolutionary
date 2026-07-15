import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from save_reader import read_run, ACTIONS_N, BUILD_SAVE_DIR


OUT_DIR = Path(BUILD_SAVE_DIR).resolve().parent / "analysis"

# NOTE on "chosen action type" analysis (dropped vs. the old version of this
# script): `chosen_action` indexes into ACTIONS_N (=OUTPUT_NEURONS_N=32)
# output-layer slots. The new save stream's `action_type` field is only
# ACTION_TYPES_N (=5) entries per creature, and represents per-creature
# genetic parameters recorded at birth -- not a slot -> type lookup table.
# There's no data in the stream that maps a chosen slot (0-31) to one of the
# 5 semantic action types, so that heatmap has been removed here. If you have
# that mapping defined elsewhere in the sim code, it can be reintroduced.


def chosen_action_slot_counts(record: dict) -> np.ndarray:
    """Counts of raw chosen_action slot values (0..ACTIONS_N-1) among old
    creatures this tick. chosen_action is only recorded for old creatures
    (the first `first_newborn_index` entries), so newly-born creatures this
    tick are not included -- they have not acted yet.
    """
    counts = np.zeros(ACTIONS_N, dtype=np.int64)

    chosen_action = record["chosen_action"]
    valid = (chosen_action >= 0) & (chosen_action < ACTIONS_N)

    if np.any(valid):
        values, value_counts = np.unique(chosen_action[valid].astype(np.int64), return_counts=True)
        counts[values] = value_counts

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=0, help="0 means no limit.")

    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_run()  # reads from the fixed build/save/ location

    if not records:
        raise SystemExit(f"No save records found under {BUILD_SAVE_DIR}")

    records = records[::max(args.stride, 1)]

    if args.max_ticks > 0:
        records = records[:args.max_ticks]

    ticks = [r["t"] for r in records]

    print(f"Reading from: {BUILD_SAVE_DIR}")
    print(f"Output: {out_dir}")
    print(f"Selected ticks: {len(ticks)}")

    rows = []
    slot_matrix = []

    for record in records:
        slot_counts = chosen_action_slot_counts(record)
        slot_matrix.append(slot_counts)

        count = record["count"]
        if count > 0:
            energy_mean = float(np.mean(record["energy"]))
            energy_min = float(np.min(record["energy"]))
            energy_max = float(np.max(record["energy"]))
            water_mean = float(np.mean(record["water"]))
            water_min = float(np.min(record["water"]))
            water_max = float(np.max(record["water"]))
        else:
            energy_mean = energy_min = energy_max = np.nan
            water_mean = water_min = water_max = np.nan

        rows.append(
            {
                "tick": record["t"],
                "count": count,
                "energy_mean": energy_mean,
                "energy_min": energy_min,
                "energy_max": energy_max,
                "water_mean": water_mean,
                "water_min": water_min,
                "water_max": water_max,
            }
        )

    slot_matrix = np.asarray(slot_matrix, dtype=np.int64)
    tick_array = np.asarray(ticks, dtype=int)

    csv_path = out_dir / "stats.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {csv_path}")

    creature_counts = np.asarray([row["count"] for row in rows], dtype=float)
    mean_energy = np.asarray([row["energy_mean"] for row in rows], dtype=float)
    mean_water = np.asarray([row["water_mean"] for row in rows], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(tick_array, creature_counts)
    plt.xlabel("tick")
    plt.ylabel("creature count")
    plt.title("Creature count over time")
    plt.tight_layout()
    plt.savefig(out_dir / "creature_count.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(tick_array, mean_energy, label="energy")
    plt.plot(tick_array, mean_water, label="water")
    plt.xlabel("tick")
    plt.ylabel("mean value")
    plt.title("Mean creature energy / water over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_energy_water.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.imshow(slot_matrix.T, aspect="auto", origin="lower")
    plt.colorbar(label="number of agents")

    plt.yticks(
        ticks=np.arange(ACTIONS_N),
        labels=[str(i) for i in range(ACTIONS_N)],
    )

    x_positions = np.linspace(0, len(ticks) - 1, min(8, len(ticks)), dtype=int)
    plt.xticks(x_positions, [str(ticks[i]) for i in x_positions], rotation=45)

    plt.xlabel("tick")
    plt.ylabel("chosen action slot")
    plt.title("Chosen action slot heatmap (old creatures only)")
    plt.tight_layout()
    plt.savefig(out_dir / "chosen_action_slot_heatmap.png", dpi=180)
    plt.close()

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()