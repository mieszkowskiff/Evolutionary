#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SEASON_OFFSET = 1.0
SEASON_AMPLITUDE = 0.5
SEASON_PERIOD = 500.0


SERIES = [
    ("population", 3.0, "-", True),
    ("food_spawn", 2.5, "--", True),
    ("water_spawn", 2.5, ":", True),
    ("move", 1.8, "-", False),
    ("eat", 1.8, "-", False),
    ("drink", 1.8, "-", False),
    ("attack", 1.8, "-", False),
    ("reproduce", 1.8, "-", False),
    ("kills", 2.3, "-", False),
]


FIT_SERIES = [
    "population",
    "food_spawn",
    "water_spawn",
    "move",
    "eat",
    "drink",
    "attack",
    "reproduce",
    "kills",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path, help="Path to run.log")
    parser.add_argument("--window", type=int, default=20000, help="Tick interval width for separate plots and fits.")
    parser.add_argument("--fit-period", type=float, default=SEASON_PERIOD, help="Initial/central period guess for sine fitting.")
    parser.add_argument("--fit-band", type=float, default=0.30, help="Relative period search band around --fit-period.")
    parser.add_argument("--fit-grid", type=int, default=241, help="Number of period candidates in grid search.")
    parser.add_argument("--min-fit-points", type=int, default=20, help="Minimum points required to fit an interval.")
    return parser.parse_args()


def parse_log(log_path: Path) -> pd.DataFrame:
    rows = []

    for line in log_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            continue

        kills = 0
        left = stripped
        if "kills=" in stripped:
            left, right = stripped.split("kills=", 1)
            m = re.search(r"-?\d+", right)
            if m:
                kills = int(m.group(0))

        nums = [int(x) for x in re.findall(r"-?\d+", left)]

        # Supported formats:
        # tick global_id population move eat attack reproduce
        # tick global_id population move eat attack reproduce drink
        # with optional trailing "kills= K" handled above.
        if len(nums) < 7:
            continue

        tick = nums[0]
        global_id = nums[1]
        population = nums[2]
        move = nums[3]
        eat = nums[4]
        attack = nums[5]
        reproduce = nums[6]
        drink = nums[7] if len(nums) >= 8 else 0

        rows.append((tick, global_id, population, move, eat, attack, reproduce, drink, kills))

    if not rows:
        raise SystemExit("No matching tick lines found.")

    return pd.DataFrame(
        rows,
        columns=["tick", "global_id", "population", "move", "eat", "attack", "reproduce", "drink", "kills"],
    )


def add_resource_reference_curves(df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    cfg_path = run_dir / "run_config.json"
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
        SEASON_OFFSET + SEASON_AMPLITUDE * np.sin(2.0 * np.pi * df["tick"].to_numpy(dtype=float) / SEASON_PERIOD),
    )

    df = df.copy()
    df["food_spawn"] = np.rint(base_food * seasonal_factor).astype(int)
    df["water_spawn"] = np.rint(base_water * seasonal_factor).astype(int)
    return df


def active_series(df: pd.DataFrame) -> list[tuple[str, float, str, bool]]:
    result = []
    for col, width, style, always in SERIES:
        if col not in df.columns:
            continue
        if always or df[col].sum() > 0:
            result.append((col, width, style, always))
    return result


def plot_curves(df: pd.DataFrame, run_dir: Path, out: Path, title_suffix: str = "") -> None:
    plt.figure(figsize=(20, 10), dpi=140)

    for col, width, style, _always in active_series(df):
        plt.plot(df["tick"], df[col], linestyle=style, linewidth=width, label=col)

    title = f"Simulation dynamics: {run_dir.name}"
    if title_suffix:
        title += f" — {title_suffix}"

    plt.title(title, fontsize=20, pad=14)
    plt.xlabel("tick", fontsize=14)
    plt.ylabel("count / spawned resource units", fontsize=14)
    plt.grid(True, alpha=0.28)
    plt.legend(ncol=5, fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def iter_windows(df: pd.DataFrame, window: int):
    min_tick = int(df["tick"].min())
    max_tick = int(df["tick"].max())

    first_start = (min_tick // window) * window
    start = first_start

    while start <= max_tick:
        end = start + window
        part = df[(df["tick"] >= start) & (df["tick"] < end)].copy()
        if len(part) > 0:
            yield start, end, part
        start = end


def fit_sine_grid(tick: np.ndarray, y: np.ndarray, period_guess: float, period_band: float, grid_n: int, min_points: int):
    mask = np.isfinite(tick) & np.isfinite(y)
    tick = tick[mask].astype(float)
    y = y[mask].astype(float)

    if len(y) < min_points:
        return None

    y_mean = float(np.mean(y))
    y_var = float(np.sum((y - y_mean) ** 2))
    if y_var <= 1e-12:
        return {
            "A": 0.0,
            "omega": 2.0 * math.pi / period_guess,
            "period": period_guess,
            "phi": 0.0,
            "B": y_mean,
            "rmse": 0.0,
            "r2": 1.0,
            "n_points": int(len(y)),
        }

    x = tick - tick[0]

    low_period = max(2.0, period_guess * (1.0 - period_band))
    high_period = max(low_period + 1e-6, period_guess * (1.0 + period_band))
    periods = np.linspace(low_period, high_period, max(3, grid_n))

    best = None

    for period in periods:
        omega = 2.0 * math.pi / period
        design = np.column_stack([
            np.sin(omega * x),
            np.cos(omega * x),
            np.ones_like(x),
        ])

        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        sse = float(np.sum((y - pred) ** 2))

        if best is None or sse < best["sse"]:
            best = {
                "period": float(period),
                "omega": float(omega),
                "coef": coef,
                "pred": pred,
                "sse": sse,
            }

    assert best is not None

    c_sin, c_cos, b = best["coef"]
    amplitude = float(np.sqrt(c_sin * c_sin + c_cos * c_cos))
    phi = float(np.arctan2(c_cos, c_sin))
    rmse = float(np.sqrt(best["sse"] / len(y)))
    r2 = float(1.0 - best["sse"] / y_var)

    return {
        "A": amplitude,
        "omega": best["omega"],
        "period": best["period"],
        "phi": phi,
        "B": float(b),
        "rmse": rmse,
        "r2": r2,
        "n_points": int(len(y)),
    }


def fit_windows(df: pd.DataFrame, run_dir: Path, fits_dir: Path, window: int, period_guess: float, period_band: float, fit_grid: int, min_points: int) -> pd.DataFrame:
    fits_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for start, end, part in iter_windows(df, window):
        rows = []

        for col in FIT_SERIES:
            if col not in part.columns:
                continue
            if col not in {"food_spawn", "water_spawn"} and part[col].sum() <= 0:
                continue

            fit = fit_sine_grid(
                part["tick"].to_numpy(),
                part[col].to_numpy(dtype=float),
                period_guess=period_guess,
                period_band=period_band,
                grid_n=fit_grid,
                min_points=min_points,
            )

            row = {
                "run": run_dir.name,
                "series": col,
                "interval_start": start,
                "interval_end": end,
                "interval_mid": 0.5 * (start + end),
                "model": "A*sin(omega*(tick-interval_start)+phi)+B",
            }

            if fit is None:
                row.update({
                    "A": np.nan,
                    "omega": np.nan,
                    "period": np.nan,
                    "phi": np.nan,
                    "B": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "n_points": len(part),
                })
            else:
                row.update(fit)

            rows.append(row)
            all_rows.append(row)

        interval_out = fits_dir / f"sine_fit_params_{start:06d}_{end:06d}.csv"
        pd.DataFrame(rows).to_csv(interval_out, index=False)
        print(f"Saved: {interval_out}")

    all_fits = pd.DataFrame(all_rows)
    all_out = fits_dir / "sine_fit_params_all_windows.csv"
    all_fits.to_csv(all_out, index=False)
    print(f"Saved: {all_out}")
    return all_fits


def plot_aggregate_fit_params(all_fits: pd.DataFrame, out: Path) -> None:
    if all_fits.empty:
        return

    params = [
        ("A", "amplitude A"),
        ("B", "baseline B"),
        ("period", "period"),
        ("phi", "phase phi"),
        ("r2", "R²"),
    ]

    fig, axes = plt.subplots(len(params), 1, figsize=(18, 20), dpi=140, sharex=True)

    for ax, (param, label) in zip(axes, params):
        for series in FIT_SERIES:
            part = all_fits[all_fits["series"] == series].sort_values("interval_mid")
            if part.empty:
                continue
            if param not in part.columns:
                continue
            ax.plot(part["interval_mid"], part[param], marker="o", linewidth=1.8, label=series)

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.28)
        ax.legend(ncol=5, fontsize=9, frameon=True)

    axes[-1].set_xlabel("tick / interval midpoint")
    fig.suptitle("Sine fit parameters over time windows", fontsize=20, y=0.995)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    args = parse_args()
    log_path = args.log_path.resolve()
    run_dir = log_path.parent

    df = parse_log(log_path)
    df = add_resource_reference_curves(df, run_dir)

    # Original full-run plot, kept for compatibility.
    full_out = run_dir / "run_curves.png"
    plot_curves(df, run_dir, full_out)
    print(f"Saved: {full_out}")

    # Same plot split by time windows.
    windows_dir = run_dir / f"run_curves_windows_{args.window}"
    windows_dir.mkdir(parents=True, exist_ok=True)

    for start, end, part in iter_windows(df, args.window):
        out = windows_dir / f"run_curves_{start:06d}_{end:06d}.png"
        plot_curves(part, run_dir, out, title_suffix=f"ticks {start}-{end}")
        print(f"Saved: {out}")

    fits_dir = run_dir / f"sine_fit_windows_{args.window}"
    all_fits = fit_windows(
        df=df,
        run_dir=run_dir,
        fits_dir=fits_dir,
        window=args.window,
        period_guess=args.fit_period,
        period_band=args.fit_band,
        fit_grid=args.fit_grid,
        min_points=args.min_fit_points,
    )

    aggregate_out = fits_dir / "sine_fit_parameters_aggregate.png"
    plot_aggregate_fit_params(all_fits, aggregate_out)


if __name__ == "__main__":
    main()