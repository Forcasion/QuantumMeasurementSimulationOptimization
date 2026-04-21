"""
Script that gives parallel tasks to run ascending measurements solver for multiple entanglements
and merge the output for each entanglement in summary/ascending_ent{ENTANGLEMENT}_merged.csv

complex: str(Z1)+str(Z2) >= 1/2
"""


import subprocess
import numpy as np
import os
import time

os.makedirs("logs", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("summary", exist_ok=True)

NUM_WORKERS = 39
TOTAL_STATES = 20000
STATES_PER_WORKER = TOTAL_STATES // NUM_WORKERS +1

# Common args
N_MODES = 1
MAX_ATTEMPTS = 1

entanglement_list = np.linspace(2.0, 5.0, num=4)

for ENTANGLEMENT in entanglement_list:
    start_time = time.time()

    processes = []
    for i in range(NUM_WORKERS):
        log_file = open(f"logs/worker_{i}.log", "w")
        p = subprocess.Popen(
            [
                "/home/andrei/.conda/envs/QuantumMeasurementSimulationOptimization/bin/python", "bin/ascending_measurements_solver_complex.py",
                "--worker_id", str(i),
                "--total_states", str(STATES_PER_WORKER),
                "--n_modes", str(N_MODES),
                "--entanglement", str(ENTANGLEMENT),
                "--max_attempts", str(MAX_ATTEMPTS),
            ],
            stdout=log_file,
            stderr=log_file,
        )
        processes.append((p, log_file))
        print(f"Started worker {i} (PID {p.pid})")

    for p, log_file in processes:
        p.wait()
        log_file.close()

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"All workers done, entanglement {ENTANGLEMENT}. Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    # Merge output files
    merged = f"summary/ascending_ent{ENTANGLEMENT}_merged.csv"
    with open(merged, "w") as out:
        for i in range(NUM_WORKERS):
            worker_file = f"output/ascending_ent{ENTANGLEMENT}_worker{i}.csv"
            with open(worker_file) as f:
                lines = f.readlines()
                if i == 0:
                    out.writelines(lines)  # include header
                else:
                    out.writelines(lines[2:])  # skip header

    print(f"Merged output written to {merged}")

print("All done.")


