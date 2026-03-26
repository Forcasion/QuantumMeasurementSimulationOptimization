import subprocess
import os
import time


NUM_WORKERS = 2
TOTAL_STATES = 20
STATES_PER_WORKER = TOTAL_STATES // NUM_WORKERS

# Common args
N_MODES = 1
ENTANGLEMENT = 1
NUM_OPS = 10
MAX_ATTEMPTS = 1

os.makedirs("logs", exist_ok=True)
os.makedirs("output", exist_ok=True)

start_time = time.time()

processes = []
for i in range(NUM_WORKERS):
    log_file = open(f"logs/worker_{i}.log", "w")
    p = subprocess.Popen(
        [
            "python", "batch_running_optimizer.py",
            "--worker_id", str(i),
            "--total_states", str(STATES_PER_WORKER),
            "--n_modes", str(N_MODES),
            "--entanglement", str(ENTANGLEMENT),
            "--num_ops", str(NUM_OPS),
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
print(f"All workers done. Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

print("All workers done.")

# Merge output files
merged = f"output/multiple_states_ent{ENTANGLEMENT}_merged.csv"
with open(merged, "w") as out:
    for i in range(NUM_WORKERS):
        worker_file = f"output/multiple_states_ent{ENTANGLEMENT}_worker{i}.csv"
        with open(worker_file) as f:
            lines = f.readlines()
            if i == 0:
                out.writelines(lines)  # include header + ideal
            else:
                out.writelines(lines[2:])  # skip header + ideal lines

print(f"Merged output written to {merged}")