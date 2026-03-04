import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Find minimum number of operators for steering detection")
    parser.add_argument("-nm", "--n_modes", type=int, default=1, help="Number of modes (default: 2)")
    parser.add_argument("-s", "--step", type=int, default=1, help="Step size for decreasing operators (default: 1)")
    parser.add_argument("--start", type=int, default=None, help="Starting number of operators (default: 2*n*(4*n+1))")
    args = parser.parse_args()

    n_modes = args.n_modes
    default_ops = 2 * n_modes * (4 * n_modes + 1)
    start_ops = args.start if args.start is not None else default_ops
    
    print(f"Starting search for minimum operators for n_modes={n_modes}")
    print(f"Starting from {start_ops} operators, decreasing by {args.step}")
    
    last_success = None
    
    for num_ops in range(start_ops, 0, -args.step):
        print(f"\n{'='*50}")
        print(f"Testing with num_ops = {num_ops}")
        print(f"{'='*50}")
        cmd = [sys.executable, "steering_detection.py", "-nm", str(n_modes), "-no", str(num_ops)]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        success = False
        line_buffer = ""
        # Read output character by character to handle \r properly
        if process.stdout:
            while True:
                char_b = process.stdout.read(1)
                if not char_b:
                    break
                char = char_b.decode('utf-8', errors='replace') if isinstance(char_b, bytes) else char_b
                sys.stdout.write(char)
                sys.stdout.flush()
                
                if char == '\n' or char == '\r':
                    if "SUCCESS! Steering detected" in line_buffer:
                        success = True
                    line_buffer = ""
                else:
                    line_buffer += char
                    
            process.stdout.close()
        process.wait()
        
        if success:
            print(f"\n[+] Success with {num_ops} operators.")
            last_success = num_ops
        else:
            print(f"\n[-] Failed to detect steering with {num_ops} operators after all attempts.")
            break
            
    if last_success is not None:
        failed_val = last_success - args.step
        print(f"\n{'*'*50}")
        print(f"SEARCH COMPLETE")
        print(f"Minimum successful operators: {last_success}")
        if failed_val > 0:
            print(f"First failing operators count: {failed_val}")
        print(f"{'*'*50}")
    else:
        print(f"\nFailed even at starting operators ({start_ops}).")

if __name__ == "__main__":
    main()
