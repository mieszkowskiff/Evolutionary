#!/usr/bin/env python3

import argparse
import re
import struct
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import imageio
    import imageio.v3 as iio
except Exception:
    imageio = None
    iio = None


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"


LAYER_SETTINGS = {
    "danger": {
        "title": "Danger",
        "cmap": ["#ffffff", "#ffd6d6", "#ff6666", "#b30000"],
        "gamma": 0.55,
    },
    "food": {
        "title": "Food",
        "cmap": ["#ffffff", "#d8ffd8", "#4ee44e", "#00a000"],
        "gamma": 0.30,
    },
    "creature": {
        "title": "Creatures",
        "cmap": ["#ffffff", "#d7e1ff", "#4f73dd", "#001f6b"],
        "gamma": 0.45,
    },
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


def tick_from_map_path(path: Path) -> int:
    match = re.fullmatch(r"map_(\d{6})\.bin", path.name)
    if not match:
        raise ValueError(f"Bad map filename: {path.name}")
    return int(match.group(1))


def available_ticks(run_dir: Path) -> list[int]:
    ticks = []

    for path in run_dir.glob("map_*.bin"):
        try:
            ticks.append(tick_from_map_path(path))
        except ValueError:
            pass

    return sorted(ticks)

def map_path(run_dir: Path, tick: int, after_damage: bool) -> Path:
    if after_damage:
        return run_dir / f"map_{tick:06d}_after_damage.bin"

    return run_dir / f"map_{tick:06d}.bin"

def read_map(run_dir: Path, tick: int, after_damage: bool = False) -> dict:
    path = map_path(run_dir, tick, after_damage)

    with path.open("rb") as file:
        width, height = struct.unpack("ii", file.read(8))
        n = width * height

        food = np.frombuffer(file.read(n * 4), dtype=np.float32).reshape(height, width).copy()
        danger = np.frombuffer(file.read(n * 4), dtype=np.float32).reshape(height, width).copy()
        creature = np.frombuffer(file.read(n * 4), dtype=np.float32).reshape(height, width).copy()

    return {
        "width": width,
        "height": height,
        "food": food,
        "danger": danger,
        "creature": creature,
    }


def map_to_rgb(map_data: dict) -> np.ndarray:
    """
    RGB convention:
      red   = danger
      green = food
      blue  = creatures

    Black background. Overlaps become mixed colors.
    """
    food = np.clip(np.nan_to_num(map_data["food"], nan=0.0), 0.0, 1.0)
    danger = np.clip(np.nan_to_num(map_data["danger"], nan=0.0), 0.0, 1.0)
    creature = np.clip(np.nan_to_num(map_data["creature"], nan=0.0), 0.0, 1.0)

    rgb = np.zeros((map_data["height"], map_data["width"], 3), dtype=np.uint8)

    rgb[..., 0] = (danger * 255).astype(np.uint8)
    rgb[..., 1] = (food * 255).astype(np.uint8)
    rgb[..., 2] = (creature * 255).astype(np.uint8)

    return rgb


def prepare_layer(layer: np.ndarray, use_log: bool) -> np.ndarray:
    layer = np.nan_to_num(layer, nan=0.0, posinf=0.0, neginf=0.0)
    layer = np.maximum(layer, 0.0)

    if use_log:
        layer = np.log1p(layer)

    return layer


def get_percentile_for_layer(name: str, split_percentile: float, food_percentile: float) -> float:
    if name == "food":
        return food_percentile
    return split_percentile


def calculate_split_vmaxes(
    run_dir: Path,
    ticks: list[int],
    use_log: bool,
    split_percentile: float,
    food_percentile: float,
    after_damage: bool = False,
) -> dict:
    values = {
        "danger": [],
        "food": [],
        "creature": [],
    }

    for tick in ticks:
        map_data = read_map(run_dir, tick, after_damage)

        for name in values:
            layer = prepare_layer(map_data[name], use_log)
            positive = layer[layer > 0]

            if positive.size > 0:
                values[name].append(positive.ravel())

    vmaxes = {}

    for name, arrays in values.items():
        if not arrays:
            vmaxes[name] = 1.0
            continue

        merged = np.concatenate(arrays)
        percentile = get_percentile_for_layer(name, split_percentile, food_percentile)
        vmax = float(np.percentile(merged, percentile))

        if vmax <= 0.0:
            vmax = float(merged.max()) if merged.size > 0 else 1.0

        if vmax <= 0.0:
            vmax = 1.0

        vmaxes[name] = vmax

    return vmaxes


def get_split_cmap(name: str):
    return mcolors.LinearSegmentedColormap.from_list(
        f"{name}_white_scale",
        LAYER_SETTINGS[name]["cmap"],
    )


def get_split_norm(name: str, vmax: float):
    return mcolors.PowerNorm(
        gamma=LAYER_SETTINGS[name]["gamma"],
        vmin=0.0,
        vmax=vmax,
        clip=True,
    )


def save_split_frame(
    map_data: dict,
    tick: int,
    frame_path: Path,
    vmaxes: dict,
    use_log: bool,
) -> None:
    layers = ["danger", "food", "creature"]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=120)
    fig.patch.set_facecolor("white")

    for ax, name in zip(axes, layers):
        layer = prepare_layer(map_data[name], use_log)
        cmap = get_split_cmap(name)
        norm = get_split_norm(name, vmaxes[name])

        ax.imshow(
            layer,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )

        ax.set_title(LAYER_SETTINGS[name]["title"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(f"tick {tick}")
    fig.tight_layout()
    fig.savefig(frame_path, facecolor="white")
    plt.close(fig)


def merged_frame_path(frames_dir: Path, tick: int) -> Path:
    return frames_dir / f"frame_{tick:06d}.png"


def split_frame_path(split_frames_dir: Path, tick: int) -> Path:
    return split_frames_dir / f"split_frame_{tick:06d}.png"


def ensure_merged_frames(
    run_dir: Path,
    ticks: list[int],
    frames_dir: Path,
    recycle: bool,
    after_damage: bool = False,
) -> list[Path]:
    frame_paths = []

    for i, tick in enumerate(ticks):
        frame_path = merged_frame_path(frames_dir, tick)

        if recycle and frame_path.exists():
            frame_paths.append(frame_path)
            continue

        map_data = read_map(run_dir, tick, after_damage)
        rgb = map_to_rgb(map_data)
        plt.imsave(frame_path, rgb)
        frame_paths.append(frame_path)

        if (i + 1) % 25 == 0:
            print(f"Prepared {i + 1}/{len(ticks)} merged frames")

    return frame_paths


def ensure_split_frames(
    run_dir: Path,
    ticks: list[int],
    split_frames_dir: Path,
    recycle: bool,
    use_log: bool,
    split_percentile: float,
    food_percentile: float,
    after_damage: bool = False,
) -> list[Path]:
    frame_paths = []

    need_generate = []
    for tick in ticks:
        frame_path = split_frame_path(split_frames_dir, tick)
        if recycle and frame_path.exists():
            frame_paths.append(frame_path)
        else:
            need_generate.append(tick)

    if need_generate:
        vmaxes = calculate_split_vmaxes(
            run_dir,
            ticks,
            use_log,
            split_percentile,
            food_percentile,
            after_damage,
        )

        existing = {tick_from_generated_split_path(path): path for path in frame_paths if path.exists()}
        frame_paths = []

        for i, tick in enumerate(ticks):
            frame_path = split_frame_path(split_frames_dir, tick)

            if recycle and frame_path.exists():
                frame_paths.append(frame_path)
                continue

            map_data = read_map(run_dir, tick, after_damage)
            save_split_frame(
                map_data=map_data,
                tick=tick,
                frame_path=frame_path,
                vmaxes=vmaxes,
                use_log=use_log,
            )
            frame_paths.append(frame_path)

            if (i + 1) % 25 == 0:
                print(f"Prepared {i + 1}/{len(ticks)} split frames")
    else:
        frame_paths = [split_frame_path(split_frames_dir, tick) for tick in ticks]

    return frame_paths


def tick_from_generated_split_path(path: Path) -> int:
    match = re.fullmatch(r"split_frame_(\d{6})\.png", path.name)
    if not match:
        raise ValueError(f"Bad split frame filename: {path.name}")
    return int(match.group(1))


def write_mp4(frame_paths: list[Path], out_path: Path, fps: int) -> None:
    if imageio is None or iio is None:
        raise SystemExit("Missing imageio. Install with: uv pip install imageio imageio-ffmpeg")

    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as writer:
        for path in frame_paths:
            writer.append_data(iio.imread(path))


def write_gif(frame_paths: list[Path], out_path: Path, fps: int) -> None:
    if imageio is None or iio is None:
        raise SystemExit("Missing imageio. Install with: uv pip install imageio imageio-ffmpeg")

    with imageio.get_writer(out_path, mode="I", fps=fps) as writer:
        for path in frame_paths:
            writer.append_data(iio.imread(path))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("exp", help="Experiment number, e.g. 1, 2, 15, or exp_0001.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--frames", action="store_true")
    parser.add_argument("--mp4", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--fps", type=int, default=20)

    parser.add_argument("--split-frames", action="store_true")
    parser.add_argument("--split-mp4", action="store_true")
    parser.add_argument("--split-gif", action="store_true")
    parser.add_argument("--split-log", action="store_true")
    parser.add_argument("--split-percentile", type=float, default=99.5)
    parser.add_argument("--food-percentile", type=float, default=97.0)

    parser.add_argument("--recycle", action="store_true", help="Reuse existing frame PNG files when possible.")
    parser.add_argument("--after-damage", action="store_true", help="Use map_XXXXXX_after_damage.bin files instead of regular map_XXXXXX.bin files.")

    args = parser.parse_args()

    run_dir = find_run(args.exp)
    out_dir = run_dir / "analysis" / "visualization"
    if args.after_damage:
        frames_dir = out_dir / "frames_after_damage"
        split_frames_dir = out_dir / "split_frames_after_damage"
    else:
        frames_dir = out_dir / "frames"
        split_frames_dir = out_dir / "split_frames"

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    split_frames_dir.mkdir(parents=True, exist_ok=True)

    ticks = available_ticks(run_dir)
    if args.after_damage:
        ticks = [tick for tick in ticks if map_path(run_dir, tick, True).exists()]

    if not ticks:
        if args.after_damage:
            raise SystemExit(f"No map_XXXXXX_after_damage.bin files found in {run_dir}")
        raise SystemExit(f"No map_*.bin files found in {run_dir}")

    ticks = ticks[::max(args.stride, 1)]

    if args.max_frames > 0:
        ticks = ticks[:args.max_frames]

    print(f"Run: {run_dir}")
    print(f"Output: {out_dir}")
    print(f"Selected frames: {len(ticks)}")
    if args.recycle:
        print("Recycle mode: ON")
    if args.after_damage:
        print("After-damage mode: ON")

    need_merged = args.frames or args.mp4 or args.gif
    need_split = args.split_frames or args.split_mp4 or args.split_gif

    frame_paths = []
    if need_merged:
        frame_paths = ensure_merged_frames(
            run_dir,
            ticks,
            frames_dir,
            args.recycle,
            args.after_damage,
        )
        print(f"Merged frames ready in: {frames_dir}")

    split_frame_paths = []
    if need_split:
        split_frame_paths = ensure_split_frames(
            run_dir,
            ticks,
            split_frames_dir,
            args.recycle,
            args.split_log,
            args.split_percentile,
            args.food_percentile,
            args.after_damage,
        )
        print(f"Split frames ready in: {split_frames_dir}")

    if args.mp4:
        mp4_path = out_dir / ("simulation_after_damage.mp4" if args.after_damage else "simulation.mp4")
        write_mp4(frame_paths, mp4_path, args.fps)
        print(f"Saved MP4: {mp4_path}")

    if args.gif:
        gif_path = out_dir / ("simulation_after_damage.gif" if args.after_damage else "simulation.gif")
        write_gif(frame_paths, gif_path, args.fps)
        print(f"Saved GIF: {gif_path}")

    if args.split_mp4:
        split_mp4_path = out_dir / ("split_layers_after_damage.mp4" if args.after_damage else "split_layers.mp4")
        write_mp4(split_frame_paths, split_mp4_path, args.fps)
        print(f"Saved split MP4: {split_mp4_path}")

    if args.split_gif:
        split_gif_path = out_dir / ("split_layers_after_damage.gif" if args.after_damage else "split_layers.gif")
        write_gif(split_frame_paths, split_gif_path, args.fps)
        print(f"Saved split GIF: {split_gif_path}")


if __name__ == "__main__":
    main()
