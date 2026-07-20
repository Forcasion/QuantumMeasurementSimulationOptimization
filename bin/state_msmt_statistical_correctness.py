# Find parameters for a random state; see from a sample of other states how many can be detected with these parameters

import numpy as np
import time
import os
from entanglement_detection import entanglement_detection, check_constraints
from state_generation import randCM
from measurement_generation import measurement_random

from numpy import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Steering Detection")
    parser.add_argument("-nm", "--n_modes", type=int, default=1, help="Number of modes per block (matrix size) (default: 1)")
    parser.add_argument("-e", "--entanglement", type=float, default=2, help="Target entanglement level (default: 1.0)")
    parser.add_argument("-ma", "--max_attempts", type=int, default=1, help="Max optimization attempts per state (default: 1)")
    parser.add_argument("-ts", "--total_states", type=int, default=1, help="Total number of states to check(default: 100)")
    parser.add_argument("--worker_id", type=int, default=0, help="Worker ID for parallel runs")
    parser.add_argument("--alternate_states", type=int, default=20000, help="Number of states checked against")

    args = parser.parse_args()
    n_modes = args.n_modes
    entanglement_target = args.entanglement
    max_attempts = args.max_attempts
    total_states = args.total_states
    total_alternate_states = args.alternate_states

    for state in range(total_states):
        start_time_total = time.time()
        # Generate new state
        state_g = None
        print(f"\nGenerating state {state}.")
        start_time_state = time.time()
        while state_g is None:
            state_g = randCM(entanglement_target, n_modes)
        end_time_state = time.time()
        print(f"\nState {state} generated. time {end_time_state - start_time_state}s")

        detected_steering = False

        for num_ops in range(1, 11):
            # Generate measurements for this state
            M_list = measurement_random(n_modes, num_ops)
            m_list = [np.real(np.trace(M @ state_g)) for M in M_list]
            num_ops = len(M_list)

            for attempt in range(max_attempts):
                print(f"  Attempt {attempt+1}/{max_attempts} ({num_ops} operators)...")
                min_val, w_opt = entanglement_detection(M_list, m_list, num_ops, n_modes)

                if min_val is not inf:
                    print(f"\nSteering detected for entanglement level {entanglement_target} using {num_ops} measurements.")
                    detected_steering = True
                    break
                else:
                    print(f"\nOptimization failed for entanglement level {entanglement_target} using {num_ops} measurements.")
            if detected_steering:
                break
        count = 0
        if detected_steering:
            for state in range(total_alternate_states):
                state_g_temp = None
                print(f"\nChecking alternate state {state+1}/{total_alternate_states}.")
                while state_g_temp is None:
                    state_g_temp = randCM(entanglement_target, n_modes)
                m_list = [np.real(np.trace(M @ state_g_temp)) for M in M_list] #this M_list is the one that stopped in the num_ops loop
                num_ops = len(M_list)
                obj = np.dot(w_opt, m_list) #w_opt is the one from the same loop
                res = check_constraints(w_opt, M_list, obj, num_ops, n_modes, verbose=False)
                if res['all_constraints_ok']:
                    print()
                    count = count +1
            print(f"Solution is also applicable for {count}/{total_alternate_states} states.")
        else:
            print("Initial optimization failed")

        end_time_total = time.time()
        print(f"\nTotal time state {state}: {end_time_total - start_time_total}s")




