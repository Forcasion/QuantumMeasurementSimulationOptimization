def evaluate_2d_slice(w0, d1, d2, M_list, m_list, num_ops, n_modes,
                      grid_size=80, span=1.0):
    size = 2 * n_modes
    alphas = np.linspace(-span, span, grid_size)
    betas  = np.linspace(-span, span, grid_size)

    F = np.zeros((grid_size, grid_size))
    PSD = np.zeros_like(F)
    STR = np.zeros_like(F)
    STEER = np.zeros_like(F)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            w = w0 + a*d1 + b*d2

            # Objective
            f = np.dot(w, m_list)
            F[i, j] = f

            # Build W
            W = np.sum([w[k] * M_list[k] for k in range(num_ops)], axis=0)
            W = (W + W.T) / 2.0

            # PSD constraint
            min_eig = np.min(np.linalg.eigvalsh(W))
            PSD[i, j] = min_eig

            # Symplectic trace
            Z1 = W[0:size, 0:size]
            Z2 = W[size:2*size, size:2*size]
            sTr1, _ = sTr(Z1, n_modes)
            sTr2, _ = sTr(Z2, n_modes)

            if sTr1 is None or sTr2 is None:
                STR[i, j] = -1.0
            else:
                STR[i, j] = sTr1 + sTr2 - 0.5

            # Steering
            STEER[i, j] = 0.999 - f

    return alphas, betas, F, PSD, STR, STEER


def plot_2d_slice(alphas, betas, F, PSD, STR, STEER):
    A, B = np.meshgrid(alphas, betas, indexing='ij')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ============================================================
    # LEFT: OBJECTIVE LANDSCAPE
    # ============================================================
    ax = axes[0]

    obj = ax.contourf(A, B, F, levels=50)
    ax.scatter(0, 0, s=80, marker='x')
    fig.colorbar(obj, ax=ax, label="Objective f(w)")

    ax.set_title("Objective Landscape")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.grid(True, alpha=0.3)

    # ============================================================
    # RIGHT: CONSTRAINT GEOMETRY
    # ============================================================
    ax = axes[1]
    ax.scatter(0, 0, s=80, marker='x')

    # Feasible regions (binary masks)
    psd_region = PSD >= 0
    str_region = STR >= 0
    steer_region = STEER >= 0

    # Plot shaded regions separately (different layers)
    ax.contourf(A, B, psd_region, levels=[0.5, 1], alpha=0.2)
    ax.contourf(A, B, str_region, levels=[0.5, 1], alpha=0.2)
    ax.contourf(A, B, steer_region, levels=[0.5, 1], alpha=0.2)

    # Boundaries
    c_psd = ax.contour(A, B, PSD, levels=[0], linewidths=2)
    c_str = ax.contour(A, B, STR, levels=[0], linestyles='--', linewidths=2)
    c_steer = ax.contour(A, B, STEER, levels=[0], linestyles=':', linewidths=2)

    # Labels
    ax.clabel(c_psd, fmt={0: 'PSD'}, inline=True)
    ax.clabel(c_str, fmt={0: 'sTr'}, inline=True)
    ax.clabel(c_steer, fmt={0: 'Steering'}, inline=True)

    ax.set_title("Constraint Feasible Regions")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualiser_wrapper(opt, w_res):
    result_code = opt.last_optimize_result()
    if w_res is not None:
        print("\nVisualizing local geometry")

        d1, d2 = random_orthonormal_directions(num_ops)

        alphas, betas, F, PSD, STR, STEER = evaluate_2d_slice(
            w_res, d1, d2,
            M_list, m_list, num_ops, n_modes,
            grid_size=160,
            span=1
        )

        plot_2d_slice(alphas, betas, F, PSD, STR, STEER)

def state_diagnostics(state_g, m_list, num_ops):
    print(f"\nCM:\n{np.round(state_g, 4)}")
    print(f"CM min eigval: {np.min(np.linalg.eigvalsh(state_g)):.6f}")

    # Check m_list values
    print(f"\nm_list: min={min(m_list):.6f}, max={max(m_list):.6f}, sum={sum(m_list):.6f}")
    print(f"m_list values: {[round(m, 4) for m in m_list]}")

    # Check if a trivial feasible point exists at all
    w_test = np.ones(num_ops) / num_ops
    W_test = np.sum([w_test[k] * M_list[k] for k in range(num_ops)], axis=0)
    Z1_test = W_test[0:2, 0:2]
    Z2_test = W_test[2:4, 2:4]
    sTr1_test, _ = sTr(Z1_test, n_modes)
    sTr2_test, _ = sTr(Z2_test, n_modes)
    print(f"\nAt w=1/num_ops:")
    print(f"  min_eig(W): {np.min(np.linalg.eigvalsh(W_test)):.6f}")
    print(f"  sTr1+sTr2: {sTr1_test + sTr2_test if sTr1_test and sTr2_test else 'None'}")
    print(f"  f = w·m: {np.dot(w_test, m_list):.6f}")

    # Check best seed found by find_good_seeds
    seeds = find_good_seeds(M_list, m_list, num_ops, n_modes)
    for i, s in enumerate(seeds):
        W_s = np.sum([s[k] * M_list[k] for k in range(num_ops)], axis=0)
        Z1_s = W_s[0:2, 0:2]
        Z2_s = W_s[2:4, 2:4]
        sTr1_s, _ = sTr(Z1_s, n_modes)
        sTr2_s, _ = sTr(Z2_s, n_modes)
        print(f"\nSeed {i}: f={np.dot(s, m_list):.6f}, "
              f"min_eig={np.min(np.linalg.eigvalsh(W_s)):.6f}, "
              f"sTr={sTr1_s + sTr2_s if sTr1_s and sTr2_s else 'None':.6f}")

    # Build the "Gram matrix" of measurements weighted by m_list
    G = np.array([[np.trace(M_list[i] @ M_list[j]) for j in range(num_ops)]
                  for i in range(num_ops)])
    print(f"Condition number of G: {np.linalg.cond(G):.2e}")

    # Also check how well m_list can be represented
    A = np.array([[np.trace(M_list[i] @ M_list[j]) for j in range(num_ops)]
                  for i in range(num_ops)])
    print(f"Rank of measurement matrix: {np.linalg.matrix_rank(A)}")

    # Project CM onto measurement basis
    coeffs = np.linalg.lstsq(np.array([[np.trace(M @ N) for N in M_list]
                                       for M in M_list]), m_list, rcond=None)
    print(f"Reconstruction residual: {coeffs[1]}")