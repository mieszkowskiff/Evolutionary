import argparse
import csv
import re
import struct
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"

ACTION_NAMES = {
    0: "MOVE",
    1: "EAT",
    2: "ATTACK",
    3: "REPRODUCE",
}


def find_run(exp: str) -> Path:
    exp = str(exp).strip()

    if exp.startswith("exp_"):
        run_dir = RUNS_DIR / exp
    else:
        run_dir = RUNS_DIR / f"exp_{int(exp):04d}"

    if not run_dir.exists():
        available = sorted(p.name for p in RUNS_DIR.glob("exp_*") if p.is_dir())
        raise SystemExit(f"Run not found: {run_dir}. Available runs: {available}")

    return run_dir


def tick_from_creatures_path(path: Path) -> int:
    match = re.fullmatch(r"creatures_(\d{6})\.bin", path.name)
    if not match:
        raise ValueError(f"Bad creatures filename: {path.name}")
    return int(match.group(1))


def available_ticks(run_dir: Path) -> list[int]:
    ticks = []

    for path in run_dir.glob("creatures_*.bin"):
        try:
            ticks.append(tick_from_creatures_path(path))
        except ValueError:
            pass

    return sorted(ticks)


def read_creatures(run_dir: Path, tick: int) -> dict:
    path = run_dir / f"creatures_{tick:06d}.bin"

    with path.open("rb") as file:
        count, sensors_n, actions_n = struct.unpack("iii", file.read(12))

        x = np.frombuffer(file.read(count * 4), dtype=np.uint32).copy()
        y = np.frombuffer(file.read(count * 4), dtype=np.uint32).copy()
        energy = np.frombuffer(file.read(count * 4), dtype=np.float32).copy()
        ids = np.frombuffer(file.read(count * 8), dtype=np.int64).copy()
        chosen_action = np.frombuffer(file.read(count), dtype=np.int8).copy()

        sensor_x = np.frombuffer(
            file.read(count * sensors_n),
            dtype=np.int8,
        ).reshape(count, sensors_n).copy()

        sensor_y = np.frombuffer(
            file.read(count * sensors_n),
            dtype=np.int8,
        ).reshape(count, sensors_n).copy()

        sensor_type = np.frombuffer(
            file.read(count * sensors_n),
            dtype=np.int8,
        ).reshape(count, sensors_n).copy()

        action_x = np.frombuffer(
            file.read(count * actions_n),
            dtype=np.int8,
        ).reshape(count, actions_n).copy()

        action_y = np.frombuffer(
            file.read(count * actions_n),
            dtype=np.int8,
        ).reshape(count, actions_n).copy()

        action_type = np.frombuffer(
            file.read(count * actions_n),
            dtype=np.int8,
        ).reshape(count, actions_n).copy()

    return {
        "count": count,
        "sensors_n": sensors_n,
        "actions_n": actions_n,
        "x": x,
        "y": y,
        "energy": energy,
        "ids": ids,
        "chosen_action": chosen_action,
        "sensor_x": sensor_x,
        "sensor_y": sensor_y,
        "sensor_type": sensor_type,
        "action_x": action_x,
        "action_y": action_y,
        "action_type": action_type,
    }


def chosen_action_type_counts(creatures: dict) -> np.ndarray:
    """
    chosen_action is a selected action slot.
    action_type[agent, chosen_slot] gives semantic action type:
      0 MOVE, 1 EAT, 2 ATTACK, 3 REPRODUCE.
    """
    counts = np.zeros(4, dtype=np.int64)

    chosen_action = creatures["chosen_action"]
    action_type = creatures["action_type"]
    actions_n = creatures["actions_n"]

    valid = (chosen_action >= 0) & (chosen_action < actions_n)

    if not np.any(valid):
        return counts

    agent_indices = np.nonzero(valid)[0]
    chosen_slots = chosen_action[valid].astype(np.int64)

    selected_types = action_type[agent_indices, chosen_slots].astype(np.int64)

    for action_id in range(4):
        counts[action_id] = int(np.sum(selected_types == action_id))

    return counts


def chosen_action_slot_counts(creatures: dict) -> np.ndarray:
    actions_n = creatures["actions_n"]
    counts = np.zeros(actions_n, dtype=np.int64)

    chosen_action = creatures["chosen_action"]
    valid = (chosen_action >= 0) & (chosen_action < actions_n)

    if np.any(valid):
        values, value_counts = np.unique(chosen_action[valid].astype(np.int64), return_counts=True)
        counts[values] = value_counts

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("exp", help="Experiment number, e.g. 1, 2, 15, or exp_0001.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=0, help="0 means no limit.")

    args = parser.parse_args()

    run_dir = find_run(args.exp)
    out_dir = run_dir / "analysis" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    ticks = available_ticks(run_dir)

    if not ticks:
        raise SystemExit(f"No creatures_*.bin files found in {run_dir}")

    ticks = ticks[::max(args.stride, 1)]

    if args.max_ticks > 0:
        ticks = ticks[:args.max_ticks]

    print(f"Run: {run_dir}")
    print(f"Output: {out_dir}")
    print(f"Selected ticks: {len(ticks)}")

    rows = []
    action_type_matrix = []
    action_slot_matrix = []

    max_actions_n = 0

    for tick in ticks:
        creatures = read_creatures(run_dir, tick)

        type_counts = chosen_action_type_counts(creatures)
        slot_counts = chosen_action_slot_counts(creatures)

        max_actions_n = max(max_actions_n, len(slot_counts))

        action_type_matrix.append(type_counts)
        action_slot_matrix.append(slot_counts)

        if creatures["count"] > 0:
            energy_mean = float(np.mean(creatures["energy"]))
            energy_min = float(np.min(creatures["energy"]))
            energy_max = float(np.max(creatures["energy"]))
        else:
            energy_mean = np.nan
            energy_min = np.nan
            energy_max = np.nan

        rows.append(
            {
                "tick": tick,
                "count": creatures["count"],
                "energy_mean": energy_mean,
                "energy_min": energy_min,
                "energy_max": energy_max,
                "chosen_MOVE": int(type_counts[0]),
                "chosen_EAT": int(type_counts[1]),
                "chosen_ATTACK": int(type_counts[2]),
                "chosen_REPRODUCE": int(type_counts[3]),
            }
        )

    action_type_matrix = np.asarray(action_type_matrix, dtype=np.int64)

    padded_slot_matrix = np.zeros((len(action_slot_matrix), max_actions_n), dtype=np.int64)
    for i, row in enumerate(action_slot_matrix):
        padded_slot_matrix[i, : len(row)] = row

    tick_array = np.asarray(ticks, dtype=int)

    csv_path = out_dir / "stats.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {csv_path}")

    creature_counts = np.asarray([row["count"] for row in rows], dtype=float)
    mean_energy = np.asarray([row["energy_mean"] for row in rows], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(tick_array, creature_counts)
    plt.xlabel("tick")
    plt.ylabel("creature count")
    plt.title("Creature count over time")
    plt.tight_layout()
    plt.savefig(out_dir / "creature_count.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(tick_array, mean_energy)
    plt.xlabel("tick")
    plt.ylabel("mean energy")
    plt.title("Mean creature energy over time")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_energy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.imshow(action_type_matrix.T, aspect="auto", origin="lower")
    plt.colorbar(label="number of agents")

    plt.yticks(
        ticks=np.arange(4),
        labels=[ACTION_NAMES[i] for i in range(4)],
    )

    x_positions = np.linspace(0, len(ticks) - 1, min(8, len(ticks)), dtype=int)
    plt.xticks(x_positions, [str(ticks[i]) for i in x_positions], rotation=45)

    plt.xlabel("tick")
    plt.ylabel("chosen action type")
    plt.title("Chosen action type heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "chosen_action_type_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.imshow(padded_slot_matrix.T, aspect="auto", origin="lower")
    plt.colorbar(label="number of agents")

    plt.yticks(
        ticks=np.arange(max_actions_n),
        labels=[str(i) for i in range(max_actions_n)],
    )

    x_positions = np.linspace(0, len(ticks) - 1, min(8, len(ticks)), dtype=int)
    plt.xticks(x_positions, [str(ticks[i]) for i in x_positions], rotation=45)

    plt.xlabel("tick")
    plt.ylabel("chosen action slot")
    plt.title("Chosen action slot heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "chosen_action_slot_heatmap.png", dpi=180)
    plt.close()

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()