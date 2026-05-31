import argparse
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
BINARY = ROOT / "build" / "Evolutionary"


def next_experiment_dir() -> tuple[int, Path]:
    RUNS_DIR.mkdir(exist_ok=True)

    max_n = 0

    for path in RUNS_DIR.iterdir():
        if not path.is_dir():
            continue

        match = re.fullmatch(r"exp_(\d+)", path.name)
        if match:
            max_n = max(max_n, int(match.group(1)))

    exp_n = max_n + 1
    run_dir = RUNS_DIR / f"exp_{exp_n:04d}"
    return exp_n, run_dir


def parse_duration_seconds(value: str | None) -> float | None:
    if value is None:
        return None

    value = value.strip().lower()

    if value in {"none", "inf", "infinite", "unlimited"}:
        return None

    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", value)
    if not match:
        raise ValueError(f"Bad duration: {value}. Use examples like 10s, 2m, 1h, none.")

    number = float(match.group(1))
    unit = match.group(2) or "s"

    if unit == "s":
        return number
    if unit == "m":
        return number * 60
    if unit == "h":
        return number * 3600

    raise ValueError(f"Unsupported duration unit: {unit}")


def read_compile_time_constants() -> dict:
    constants_path = ROOT / "constants.h"

    if not constants_path.exists():
        return {}

    text = constants_path.read_text(errors="replace")

    keys = [
        "WIDTH",
        "HEIGHT",
        "MAX_CREATURE_N",
        "SENSORS_N",
        "SEASON_SENSORS_N",
        "TOTAL_SENSORS_N",
        "HIDDEN_N",
        "ACTIONS_N",
        "ACTION_TYPES_N",
        "FOOD_SPAWN_QUANTITY",
        "WATER_SPAWN_QUANTITY",
        "SEASON_PERIOD",
    ]

    result = {}

    for key in keys:
        match = re.search(
            rf"^\s*#define\s+{key}\s+(.+?)\s*$",
            text,
            flags=re.MULTILINE,
        )
        if match:
            result[key] = match.group(1)

    return result


def command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return f"<failed: {exc}>"


def build_binary_command(args) -> list[str]:
    command = [
        str(BINARY),
        "--seed",
        str(args.seed),
        "--initial-creatures",
        str(args.initial_creatures),
        "--food-spawn-quantity",
        str(args.food_spawn_quantity),
        "--initial-food-multiplier",
        str(args.initial_food_multiplier),
        "--save-every",
        str(args.save_every),
        "--max-ticks",
        str(args.max_ticks),
        "--contract-every",
        str(args.contract_every),
        "--save-creatures",
        str(args.save_creatures),
    ]

    return command


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="", help="Optional human-readable experiment label.")
    parser.add_argument("--duration", default="10s", help="Wall-clock limit, e.g. 10s, 2m, 1h, none.")

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--initial-creatures", type=int, default=4096)
    parser.add_argument("--food-spawn-quantity", type=int, default=1024)
    parser.add_argument("--initial-food-multiplier", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-ticks", type=int, default=1000)
    parser.add_argument("--contract-every", type=int, default=10)
    parser.add_argument("--save-creatures", type=int, default=1)

    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not BINARY.exists():
        raise SystemExit(
            f"Binary not found: {BINARY}\n"
            f"Build first with: cmake --build build -j\"$(nproc)\""
        )

    if not os.access(BINARY, os.X_OK):
        raise SystemExit(f"Binary is not executable: {BINARY}")

    exp_n, run_dir = next_experiment_dir()
    run_dir.mkdir(parents=True, exist_ok=False)

    command = build_binary_command(args)

    config = {
        "exp_num": exp_n,
        "exp_dir": run_dir.name,
        "name": args.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration": args.duration,
        "command": command,
        "runtime_args": {
            "seed": args.seed,
            "initial_creatures": args.initial_creatures,
            "food_spawn_quantity": args.food_spawn_quantity,
            "initial_food_multiplier": args.initial_food_multiplier,
            "save_every": args.save_every,
            "max_ticks": args.max_ticks,
            "contract_every": args.contract_every,
            "save_creatures": args.save_creatures,
        },
        "compile_time_constants": read_compile_time_constants(),
    }

    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    metadata = {
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "nvcc": command_output(["bash", "-lc", "which nvcc && nvcc --version"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "cmake": command_output(
            [
                "bash",
                "-lc",
                "grep -n 'CMAKE_CUDA_ARCHITECTURES\\|target_compile_options' CMakeLists.txt || true",
            ]
        ),
    }

    (run_dir / "metadata.txt").write_text(
        "\n\n".join(f"[{key}]\n{value}" for key, value in metadata.items())
    )

    print(f"Created run folder: {run_dir}")
    print("Command:")
    print(" ".join(command))

    if args.dry_run:
        print("Dry run only. Simulation not executed.")
        return

    timeout_seconds = parse_duration_seconds(args.duration)
    log_path = run_dir / "run.log"
    started = time.time()

    with log_path.open("w", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        try:
            while True:
                line = process.stdout.readline()

                if line:
                    print(line, end="")
                    log.write(line)

                if process.poll() is not None:
                    break

                if timeout_seconds is not None and time.time() - started >= timeout_seconds:
                    message = "\nTimeout reached. Sending SIGINT...\n"
                    print(message, end="")
                    log.write(message)

                    process.send_signal(signal.SIGINT)

                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        message = "Process did not stop after SIGINT. Killing.\n"
                        print(message, end="")
                        log.write(message)
                        process.kill()

                    break

            for line in process.stdout:
                print(line, end="")
                log.write(line)

        finally:
            return_code = process.wait()

    summary = {
        "return_code": return_code,
        "elapsed_seconds": time.time() - started,
        "map_files": len(list(run_dir.glob("map_*.bin"))),
        "creature_files": len(list(run_dir.glob("creatures_*.bin"))),
        "disk_usage": command_output(["du", "-sh", str(run_dir)]),
    }

    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()