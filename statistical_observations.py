import subprocess
import sys

for ent in range(1, 6):
    for num_ops in range(9, 1, -1):
        print(f"\n{'='*50}")
        print(f"Entanglement: {ent} | Num ops: {num_ops}")
        print(f"{'='*50}")
        subprocess.run(
            [sys.executable, "steering_detection.py", "-e", str(ent), "-no", str(num_ops), "-ms", "20", "-fa", "-ma", "10"],
        )