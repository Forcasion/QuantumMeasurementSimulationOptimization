import warnings
import numpy as np
import cvxpy as cp


def _omega(n_modes):
    """Symplectic form for n_modes modes, ordering (x1, p1, x2, p2, ...)."""
    return np.kron(np.eye(n_modes), np.array([[0.0, 1.0], [-1.0, 0.0]]))


def _symplectic_trace(Z, n_modes):
    """Sum of symplectic eigenvalues of a PSD matrix Z (Williamson)."""
    Om = _omega(n_modes)
    ev = np.linalg.eigvals(1j * Om @ Z)
    # eigenvalues come in +/- pairs of the symplectic eigenvalues
    return float(np.sum(np.abs(ev.real)) / 2.0)


def _psd_herm(Z, A):
    """Real embedding of the Hermitian constraint Z + iA >= 0 (Z sym, A antisym)."""
    return cp.bmat([[Z, -A], [A, Z]]) >> 0


def entanglement_detection_sdp(M_list, m_list, num_ops=14, n_modes=1,
                               c_bound=None, solver=None, verbose=False, cert_tol=1e-4):
    """
    Optimal witness from a fixed set of measurement settings, as an SDP.

    Bipartite split with n_modes modes per party (W is 4*n_modes square, party 1
    first). Z = sum_j c_j M_j.

        minimize    c . m
        subject to  Z >= 0
                    Z_j + i A_j >= 0,  A_j real antisymmetric,  j = 1, 2
                    sum_j Tr[Omega_j^T A_j] = 1

    The last two lines are an exact linear (SDP) reformulation of
    sTr(Z1) + sTr(Z2) >= 1/2 (Hyllus & Eisert 2006, Thm 1 / dual Eq. 40).

    Returns (best_obj, best_w), same as the nlopt version.
    best_obj = inf (and best_w None) if no valid witness can be certified from
    these settings. best_obj < 1 means entanglement is detected.
    """
    size = 2 * n_modes
    M = [np.real(np.asarray(Mk)) for Mk in M_list[:num_ops]]
    M = [(Mk + Mk.T) / 2.0 for Mk in M]
    m = np.real(np.asarray(m_list, dtype=float))[:num_ops]
    Om = _omega(n_modes)

    c = cp.Variable(num_ops)
    Z = sum(c[k] * M[k] for k in range(num_ops))
    Z = (Z + Z.T) / 2  # keeps CVXPY aware of symmetry
    Z1 = Z[0:size, 0:size]
    Z2 = Z[size:2 * size, size:2 * size]

    # real antisymmetric A_j built from its strictly upper-triangular entries
    def antisym(name):
        n_free = size * (size - 1) // 2
        v = cp.Variable(n_free, name=name)
        iu = np.triu_indices(size, k=1)
        U = cp.vstack([
            cp.hstack([v[np.where((iu[0] == r) & (iu[1] == s))[0][0]]
                       if r < s else 0 for s in range(size)])
            for r in range(size)
        ])
        return U - U.T

    A1 = antisym("A1")
    A2 = antisym("A2")

    constraints = [
        Z >> 0,
        _psd_herm(Z1, A1),
        _psd_herm(Z2, A2),
        cp.trace(Om.T @ A1) + cp.trace(Om.T @ A2) == 1,
    ]
    if c_bound is not None:
        constraints += [c <= c_bound, c >= -c_bound]

    prob = cp.Problem(cp.Minimize(m @ c), constraints)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # we certify the result ourselves below
        try:
            prob.solve(solver=solver, verbose=verbose)
        except cp.error.SolverError:
            prob.solve(solver=cp.SCS, verbose=verbose, eps=1e-8)

    if prob.status not in ("optimal", "optimal_inaccurate") or c.value is None:
        return np.inf, None

    w = np.array(c.value, dtype=float)

    # --- A posteriori certification -------------------------------------
    # Never trust the solver status alone: with too few settings the problem is
    # infeasible and "inaccurate" solvers can return huge, meaningless c.
    W = sum(w[k] * M[k] for k in range(num_ops))
    W = (W + W.T) / 2.0
    scale = max(np.abs(W).max(), 1e-12)
    if np.linalg.eigvalsh(W).min() < -cert_tol * scale:
        return np.inf, None
    def _psd_proj(B):
        e, V = np.linalg.eigh((B + B.T) / 2.0)
        return (V * np.clip(e, 0, None)) @ V.T
    s_tot = (_symplectic_trace(_psd_proj(W[0:size, 0:size]), n_modes)
             + _symplectic_trace(_psd_proj(W[size:, size:]), n_modes))
    if not np.isfinite(s_tot) or s_tot < 0.5 * (1 - 1e-3):
        return np.inf, None
    if s_tot < 0.5:
        w = w * (0.5 / s_tot)  # absorb solver tolerance; keeps sTr >= 1/2 exactly

    return float(np.dot(w, m)), w