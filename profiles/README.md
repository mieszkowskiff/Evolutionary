1) contracting modes: 0, 1, 2, 3
2) runtime argument --contract-mode 0|1|2|3
3) Results: modes 0 and 1 - no difference in performance, modes 2 and 3 - better than modes 0 and 1, modes 2 and 3 - no differnce
4) contract.cu has sections with all versions on contracting functions with comments on what the current version focuses on
5) Commands:

To start a run and gather data: 
nsys profile   --trace=nvtx   --sample=none   --force-overwrite=true   -o ../profiles/evo_contract_NEW_RUN_nvtx   ./Evolutionary     --seed 1     --initial-creatures 100096     --food-spawn-quantity 5064     --initial-food-multiplier 50     --save-every 0     --save-creatures 0     --contract-every 1     --max-ticks 2500     --nvtx 1     --contract-mode 3

To analyze the gathered data from the run:
nsys stats ../profiles/evo_contract_NEW_RUN_nvtx.nsys-rep

6) CMakeLists.txt got some changes so build first before trying 5) 