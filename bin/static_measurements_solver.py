import numpy as np
import time
import os
from steering_detection import steering_detection
from state_generation import randCM
from measurement_generation import measurement_random

from numpy import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Steering Detection")
    parser.add_argument("-nm", "--n_modes", type=int, default=1, help="Number of modes per block (default: 1)")
    parser.add_argument("-e", "--entanglement", type=int, default=1, help="Target entanglement level (default: 1)")
    parser.add_argument("-no", "--num_ops", type=int, default=10, help="Number of measurement operators")
    parser.add_argument("-ma", "--max_attempts", type=int, default=1, help="Max optimization attempts per state (default: 1)")
    parser.add_argument("-ts", "--total_states", type=int, default=100, help="Total number of states to check(default: 100)")
    parser.add_argument("--worker_id", type=int, default=0, help="Worker ID for parallel runs")

    args = parser.parse_args()
    n_modes = args.n_modes
    entanglement_target = args.entanglement
    num_ops = args.num_ops
    max_attempts = args.max_attempts
    total_states = args.total_states

    fileobject = open(os.path.join(os.pardir, f"output/multiple_states_ent{entanglement_target}_worker{args.worker_id}.csv"), "w")
    with fileobject as f:
        for state in range(total_states):
            start_time_total = time.time()
            # Generate new state
            state_g = None
            print(f"\nGenerating state {state}.")
            start_time_state = time.time()
            while state_g is None:
                # state_g = randCM_fixed(entanglement_target, n_modes, seed = state)
                state_g = randCM(entanglement_target, n_modes)
            end_time_state = time.time()
            print(f"\nState {state} generated. time {end_time_state - start_time_state}s")


            # Generate measurements for this state
            M_list = measurement_random(n_modes, num_ops)
            m_list = [np.real(np.trace(M @ state_g)) for M in M_list]
            num_ops = len(M_list)

            for attempt in range(max_attempts):
                print(f"  Attempt {attempt+1}/{max_attempts} ({num_ops} operators)...")
                min_val, w_opt = steering_detection(M_list, m_list, num_ops, n_modes)

                if min_val is not inf:
                    print(f"\nSteering detected for {num_ops} measurements using entanglement level {entanglement_target}.")
                else:
                    print(f"\nOptimization failed for {num_ops} measurements using entanglement level {entanglement_target}.")
                f.write(f"{state},{min_val}\n")
                if state % 10 == 9:
                    f.flush()
            end_time_total = time.time()
            print(f"\nTotal time state {state}: {end_time_total - start_time_total}s")




