import subprocess
import sys

for num_ops in range(10, 0, -1):
    print(f"\n{'='*50}")
    print(f"Trying num_ops = {num_ops}")
    print('='*50)
    result = subprocess.run(
        [sys.executable, "steering_detection.py", "-no", str(num_ops)],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"Failed at num_ops = {num_ops}")
        break