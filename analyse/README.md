# Analyze scripts

Tools for analyzing the simulation's saved creature data.

## Layout

```
analyze/
  save_reader.py   # shared reader for the creature save-stream format
  plots.py         # population / energy / action-slot plots
  plot_seasons.py  # parses a run.log for population & resource curves
```

## Where the data comes from

The simulation always writes creature data to a single **fixed** location,
as two alternating ("ping-pong") binary streams:

```
build/save/stream1.bin
build/save/stream2.bin
```

There's no per-experiment or per-run directory — just these two files. Only
one experiment's data exists at a time, so analyze it before starting the
next run, or it will be overwritten.

If your build actually places these files somewhere else, update
`BUILD_SAVE_DIR` in `save_reader.py` — that's the only place the path is
hardcoded (it defaults to `<project_root>/build/save`).

Map-file saving (`map_*.bin`) is not covered by these scripts.

## 1. Reading the raw save data — `save_reader.py`

You normally won't call this directly — `plots.py` uses it — but it's a
reusable module if you want to write your own analysis:

```python
from save_reader import read_run

records = read_run()   # reads build/save/stream1.bin + stream2.bin, sorted by tick

for r in records:
    print(r["t"], r["count"], r["energy"].mean())
```

Each record dict always has: `t`, `count`, `first_newborn_index`,
`new_creatures_count`, the three `save_*` flags, `chosen_action` (old
creatures only), and `x`/`y`/`energy`/`water`/`ids` (all creatures).

Depending on which `save_*` flags were true that tick, it may also have:
- `input_layer_values`, `hidden_layer_values`, `output_layer_values` (old creatures only)
- `sensor_x`/`sensor_y`/`sensor_type`, `action_x`/`action_y`/`action_type` (new creatures only)
- `first_matrix`, `second_matrix`, `bias` (new creatures only)

Note: `fp8_e4m3_to_float` is used internally to decode all `__nv_fp8_e4m3`
fields into regular `float32` arrays, so you don't need to handle fp8 bytes
yourself.

## 2. Plotting — `plots.py`

```bash
python plots.py                    # all ticks
python plots.py --stride 5         # use every 5th saved tick
python plots.py --max-ticks 200    # only look at the first 200 selected ticks
```

Reads every record from the fixed `build/save/` streams and writes to
`build/analysis/`:
- `stats.csv` — tick, count, energy mean/min/max, water mean/min/max
- `creature_count.png`
- `mean_energy_water.png`
- `chosen_action_slot_heatmap.png` — counts of raw `chosen_action` slot values
  (0–31) over time, for old creatures only (newly-born creatures haven't
  acted yet in their birth tick)

**Known limitation:** the old script also produced a "chosen action *type*"
breakdown (MOVE/EAT/ATTACK/REPRODUCE). That's been removed — `chosen_action`
indexes into the 32 output-layer slots, but `action_type` in the new format
is only 5 per-creature genetic values recorded at birth, not a slot→type
lookup table. There's nothing in the save stream that maps one to the other,
so the type breakdown isn't reconstructible unless that mapping is added to
the save format or is fixed/known elsewhere in the sim code.

## 3. Plotting seasonal dynamics — `plot_seasons.py`

This one is independent of the save-stream format — it only parses a text
log file, so it works unchanged and just needs a path to that log.

```bash
python plot_seasons.py path/to/run.log
python plot_seasons.py path/to/run.log --window 5000
```

Expects log lines shaped like:
```
tick global_id population move eat attack reproduce [drink] [kills=K]
```

Produces (written next to the log file):
- `run_curves.png` — full-run population/action/resource curves
- `run_curves_windows_<window>/` — the same curves split into tick windows
- `sine_fit_windows_<window>/` — per-window sine fits (period, amplitude,
  phase, R²) for each series, trying to recover the seasonal cycle, plus an
  aggregate plot of how those fit parameters drift over the run

## Typical workflow

```bash
python plots.py
python plot_seasons.py path/to/run.log
```