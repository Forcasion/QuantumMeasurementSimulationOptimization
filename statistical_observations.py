import subprocess
import sys

for ent in range(1, 6):
    for num_ops in range(10, 0, -1):
        print(f"\n{'='*50}")
        print(f"Entanglement: {ent} | Num ops: {num_ops}")
        print(f"{'='*50}")
        subprocess.run(
            [sys.executable, "steering_detection.py", "-e", str(ent), "-no", str(num_ops), "-ms", "100", "-fa"],
        )