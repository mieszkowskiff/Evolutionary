### Invokes subprocess with previously built CUDA code and passes the runtime parameters
python analyse/run.py   --name run_name   --duration none   --seed 1   --initial-creatures 100000   --food-spawn-quantity 500  --initial_food_multiplier 5  --save-every 50   --contract-every 1 --max-ticks 10000  --save_creatures 0

### Reads map.bin files to create visualizations
python analyse/visual.py 55 --gif --split-gif --fps 10 --max-frames 3000  --stride 1 --split-lo
g --split-percentile 99.5

### Reads log file to plot action_curves
python analyse/plot_seasons.py runs/exp_0055/run.log