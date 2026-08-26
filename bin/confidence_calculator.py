import numpy as np
import time
import matplotlib.pyplot as plt

from steering_detection import steering_detection
from state_generation import randCM_fixed
from measurement_generation import measurement_random

from numpy import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence calculator")
    parser.add_argument("-nm", "--n_modes", type=int, default=1, help="Number of modes per block (default: 1)")
    parser.add_argument("-e", "--entanglement", type=int, default=5, help="Target entanglement level (default: 1)")
    parser.add_argument("--worker_id", type=int, default=0, help="Worker ID for parallel runs")

    start_time_total = time.time()

    seed_state = 8
    seed_measurements = 2

    min_ops = 6
    max_ops = 11

    args = parser.parse_args()
    n_modes = args.n_modes
    entanglement_target = args.entanglement


    # Generate new state
    state_g = None
    print(f"\nGenerating state.")
    start_time_state = time.time()

    # Generate state
    while state_g is None:
        state_g = randCM_fixed(entanglement_target, n_modes, seed = seed_state)
    end_time_state = time.time()
    print(f"\nState generated. time {end_time_state - start_time_state}s")


    # Generate measurements
    M_list_global = measurement_random(n_modes, max_ops, seed=seed_measurements)
    m_list_global = [np.real(np.trace(M @ state_g)) for M in M_list_global]

    # Calculate ideal
    Z_min = 2**(-entanglement_target/5)

    x_axis = np.linspace(2, 100000, 5000)
    fig, ax = plt.subplots()
    ax.set(ylim=(0, 2))

    Zs= []
    zn= []

    for num_ops in range(min_ops, max_ops, 1):
        # Generate measurements for this state
        M_list =M_list_global[:num_ops]
        m_list = m_list_global[:num_ops]
        num_ops = len(M_list)

        print(f"  Attempting with {num_ops} measurements...")
        min_val, w_opt = steering_detection(M_list, m_list, num_ops, n_modes)

        if min_val is not inf:
            print(f"\nSteering detected for entanglement level {entanglement_target} using {num_ops} measurements.")
            square_sum = 0
            for i in range(len(m_list)):
                square_sum += m_list[i]*m_list[i]+w_opt[i]*w_opt[i]
            deltaZ = np.sqrt(2/(x_axis-1)*square_sum)
            Zs.append(np.sqrt(square_sum))
            zn.append(min_val)
            ax.plot(x_axis, Z_min+3*deltaZ, label=f'{num_ops}')
        else:
            print(f"\nOptimization failed for entanglement level {entanglement_target} using {num_ops} measurements.")
            break




    print(Zs)
    print(zn)
    print(Z_min)
    end_time_total = time.time()
    print(f"\nTotal time: {end_time_total - start_time_total}s")
    plt.legend()
    plt.show()