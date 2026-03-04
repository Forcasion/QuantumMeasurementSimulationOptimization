import traceback
import numpy as np
import nlopt
import math
import random
from scipy.linalg import block_diag, expm
from numpy import *

def get_S(Z):
    """
    Find the symplectic transformation S such that S Z S^T = D,
    where D = diag(d1, d1, d2, d2, ..., dn, dn).
    """
    size = len(Z)
    n = size // 2

    # 1. Standard interleaved Omega for [x1, y1, x2, y2, ...]
    J = np.array([[0, 1], [-1, 0]])
    Omega = block_diag(*([J] * n))

    # 2. Ensure Z is symmetric and regularized (robust to singular matrices)
    Z = (Z + Z.T) / 2 + 1e-12 * np.eye(size)
    
    # 3. Williamson decomposition:
    # Symplectic eigenvalues are eigenvalues of A = i * Omega @ Z
    A = 1j * Omega @ Z
    vals, vecs = np.linalg.eig(A)
    
    # Positive eigenvalues (d_k) and their eigenvectors
    # Sort by real parts descending. For PSD Z, they should be +/- d_k.
    # We take the n largest (one from each pair).
    sort_idx = np.argsort(vals.real)[::-1]
    d_ks = vals[sort_idx[:n]].real
    vecs_pos = vecs[:, sort_idx[:n]]
    
    # 4. Construct S such that S Z S^T = D
    # Normalize v_k such that v_k^H (i Omega) v_k = 2
    # Then the rows of S are interleaved: Im(v_k)^T and Re(v_k)^T
    rows = []
    for i in range(n):
        v = vecs_pos[:, i]
        # Normalize: v^H (i Omega) v = 2
        norm = np.real(v.conj().T @ (1j * Omega) @ v)
        # Safety for zero norm due to precision
        norm = np.maximum(1e-16, np.abs(norm))
        v = v * np.sqrt(2.0 / norm)
        
        # Phase fixing for numerical stability: force first non-zero element to be real-positive
        # This keeps the gradient from jumping around due to eigenvector phase choice.
        first_nonzero = v[np.argmax(np.abs(v) > 1e-10)]
        if np.abs(first_nonzero) > 0:
            v = v * (np.conj(first_nonzero) / np.abs(first_nonzero))
        
        # S rows for interleaved basis [x1, y1, x2, y2, ...]
        rows.append(np.imag(v).T)
        rows.append(np.real(v).T)
        
    S = np.vstack(rows)
    
    return S

def symplectic_values(Z):
    """
    Return the sum of the symplectic eigenvalues for Z.
    This is calculated exactly using the Williamson decomposition from get_S(Z).
    """
    S = get_S(Z)
    result = S @ Z @ S.T
    # For interleaved diagonal D = diag(d1, d1, d2, d2, ...), 
    # we sum d_k once for each pair.
    values = np.diag(result)[::2]
    return np.sum(values)

def measurement(theta, psi, phi):
    """Return the measurement for specific angles

    Parameters:
        theta: theta angle in [0,pi]
        psi: psi angle in [0,pi]
        z: phi angle in [0,2pi]
    Returns:
        A 4x4 numpy array
    """
    u=np.matrix([math.cos(theta)*math.cos(psi-phi), math.cos(theta)*math.sin(psi-phi), math.sin(theta)*math.cos(psi), math.sin(theta)*math.sin(psi)])
    v=np.matrix([math.cos(theta)*math.cos(psi-phi),math.cos(theta)*math.sin(psi-phi),math.sin(theta)*math.cos(psi),math.sin(theta)*math.sin(psi)])
    return np.outer(u,v.T)

def qmult_unit(n):
    """Generates a Haar-random unitary matrix of size n"""
    F = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = np.linalg.qr(F)
    return Q @ np.diag(np.exp(1j * np.random.rand(n) * 2 * np.pi))

def symp_orth(n):
    """Generates a symplectic orthogonal matrix of size n"""
    B = qmult_unit(n)
    Re = np.real(B)
    Im = np.imag(B)
    return np.block([[Re, -Im],
                    [Im,  Re]])

def rand_rsymp(n,c):
    """Generate random symplectic matrix with specified symplectic eigenvalues.
    Parameters:
        n: dimension (2nx2n)
        c: symplectic eigenvalue(s) - either a scalar or n-vector
    Returns:
        A symplectic matrix
    """
    assert (len(c)==1 or len(c)==n),  (f"Expected vector component length for "
                                       f"symplectic matrix generation either 1 or {n}, got: {len(c)}")
    for scalar in c:
        assert scalar >= 1, f"Expected vector component for symplectic matrix generation >=, got: {scalar}"

    s = np.zeros(2*n)
    if(len(c)==1):
        s[0]=np.sqrt(c[0])
        s[1:n] = np.sort(1 + (s[0] - 1) * np.random.rand(n-1))[::-1]
        s[n:] = 1.0 / s[:n]
    else:
        c = np.sort(c)
        s[0:n] = c[::-1]
        s[n:2*n] = 1.0 / s[0:n]

    U=symp_orth(n)
    V=symp_orth(n)

    return U @ np.diag(s) @ V

def randCM(entg = 1):
    """Searches for a random state with a specific entanglement level
    Parameters:
        entg: entanglement level
    Returns:
        A 4x4 matrix
    """
    int_val = 5
    rot = 10
    num = 300000

    P = np.matrix([[1,0,0,0],
                  [0,0,1,0],
                  [0,1,0,0],
                  [0,0,0,1]])

    x = 1
    y = 1.1

    a=1
    b=10

    for i in range(num):
        # generate thermal state CM
        g1 = np.matrix([[random.uniform(x,y),0],
                        [0,random.uniform(x,y)]])
        g2 = block_diag(g1, g1)

        # generate random symplectic transformations with symplectic eigenvalues between 1 and 7.5
        S=rand_rsymp(2,(random.uniform(a,b),random.uniform(a,b)))
        # apply symplectic transformations to obtain a general CM

        g3 = S.transpose()@g2@S
        cm = P.transpose()@g3@P

        # calculate the logarithmic negativity
        det_11 = np.linalg.det(cm[0:2, 0:2])
        det_22 = np.linalg.det(cm[2:4, 2:4])
        det_12 = np.linalg.det(cm[0:2, 2:4])
        det_cm = np.linalg.det(cm)

        f = (0.5 * (det_11 + det_22) - det_12
             - np.sqrt((0.5 * (det_11 + det_22) - det_12)**2 - det_cm))

        EN = -0.5 * np.log2(f)
        EN1 = np.round(EN * rot) / rot

        if EN1 == entg/int_val:
            return cm
    return None

def check_constraints(w_opt, M_list, min_val, num_ops=14, verbose=False):
    """
    Verify all constraints are satisfied.
    """
    W = np.sum([w_opt[k] * M_list[k] for k in range(num_ops)], axis=0)

    # Check 1: W ≥ 0 (PSD)
    eigvals = np.linalg.eigvalsh(W)
    min_eigval = np.min(np.real(eigvals))
    W_psd = min_eigval >= -1e-9

    # Check 2: sTr(Z11) + sTr(Z22) ≥ 0.5
    Z11 = W[0:2, 0:2]
    Z22 = W[2:4, 2:4]

    sTr_Z11 = symplectic_values(Z11)
    sTr_Z22 = symplectic_values(Z22)

    sTr_sum = sTr_Z11 + sTr_Z22
    sTr_satisfied = sTr_sum >= 0.5 - 1e-7

    steering_ok = min_val < 1.0 - 1e-7

    results = {
        'min_eigval_W': min_eigval,
        'W_is_PSD': W_psd,
        'sTr_sum': sTr_sum,
        'sTr_satisfied': sTr_satisfied,
        'steering_satisfied': steering_ok,
        'objective_value': min_val,
        'all_constraints_ok': W_psd and sTr_satisfied and steering_ok
    }

    if verbose:
        print("\n" + "="*50)
        print(f"{'Condition':30} | {'Value':12} | {'Result'}")
        print("-" * 50)
        print(f"{'PSD (Min eigval >= 0)':30} | {min_eigval:12.8f} | {'[OK]' if W_psd else '[FAIL]'}")
        if sTr_sum is not None:
            print(f"{'sTr (Sum traces >= 0.5)':30} | {sTr_sum:12.8f} | {'[OK]' if sTr_satisfied else '[FAIL]'}")
        else:
            print(f"{'sTr Condition':30} | {'ERROR':12} | [FAIL]")
        print(f"{'Steering (w·m < 1)':30} | {min_val:12.8f} | {'[OK]' if steering_ok else '[FAIL]'}")
        print("-" * 50)
        print(f"OVERALL RESULT: {'FEASIBLE [OK]' if results['all_constraints_ok'] else 'UNFEASIBLE [FAIL]'}")
        print("="*50 + "\n")

    return results

def steering_detection(M_list, m_list, num_ops=14):
    """
    Detect steering using NLopt with specific constraints

    Constraints:
    - W ≥ 0 (PSD)
    - sTr(Z11) + sTr(Z22) ≥ 0.5
    - w·m < 1 (Steering condition added as constraint)
    """
    n_vars = num_ops
    
    # Tracking variables for progress in a mutable dict
    stats = {
        'eval': 0, 
        'best_obj': np.inf, 
        'best_v': np.inf, 
        'v1': 0, 'v2': 0, 'v3': 0,
        'best_feasible_w': None,
        'best_feasible_obj': np.inf
    }

    # Directly solve the non-linear optimization using smooth determinants for 2x2 blocks.
    def objective(w, grad):
        obj = np.dot(w, m_list)
        if grad.size > 0:
            grad[:] = m_list
        stats['eval'] += 1
        if stats['eval'] % 500 == 0:
            print(f"\r    Eval: {stats['eval']:7d} | Obj: {obj:10.6f}", end="", flush=True)
        return float(obj)

    def constraint_W_psd(w, grad):
        W = np.sum([w[k] * M_list[k] for k in range(num_ops)], axis=0)
        eigvals, eigvecs = np.linalg.eigh(W)
        min_idx = np.argmin(eigvals)
        min_eigval = eigvals[min_idx]
        v_min = eigvecs[:, min_idx]
        if grad.size > 0:
            for k in range(num_ops):
                grad[k] = -float(np.real(v_min.conj().T @ M_list[k] @ v_min))
        # Require min_eigval >= 1e-8 to avoid -0.0000 displaying
        return float(1e-10 - min_eigval)

    def constraint_symplectic_trace(w, grad):
        W = np.sum([w[k] * M_list[k] for k in range(num_ops)], axis=0)
        Z1 = W[0:2, 0:2]
        Z2 = W[2:4, 2:4]
        
        # For 2x2 blocks, sTr(Z) = sqrt(det(Z))
        # Ensure Z is regularized for safe determinant
        # W constraint ensures Z is PSD, but we add epsilon for numerical stability
        det1 = np.linalg.det(Z1 + 1e-12 * np.eye(2))
        det2 = np.linalg.det(Z2 + 1e-12 * np.eye(2))
        sTr1 = np.sqrt(np.maximum(1e-18, det1))
        sTr2 = np.sqrt(np.maximum(1e-18, det2))
        
        val = 0.5 - (sTr1 + sTr2)
        
        if grad.size > 0:
            # Gradient of sqrt(det(Z)) is 0.5 * sqrt(det(Z)) * Z^-1
            inv1 = np.linalg.inv(Z1 + 1e-12 * np.eye(2))
            inv2 = np.linalg.inv(Z2 + 1e-12 * np.eye(2))
            g_Z1 = 0.5 * sTr1 * inv1
            g_Z2 = 0.5 * sTr2 * inv2
            
            for k in range(num_ops):
                # Chain rule: Tr(grad_Z^T * M_k)
                # Since grad_Z is symmetric, it's Tr(grad_Z * M_k)
                dk1 = np.trace(g_Z1 @ M_list[k][0:2, 0:2])
                dk2 = np.trace(g_Z2 @ M_list[k][2:4, 2:4])
                grad[k] = -float(dk1 + dk2)
        return float(val)

    def constraint_steering(w, grad):
        val = np.dot(w, m_list) - 0.999
        if grad.size > 0:
            grad[:] = m_list
        return float(val)

    # Initial Guess
    w0 = np.random.randn(num_ops)
    
    # Configure Optimizer
    opt = nlopt.opt(nlopt.LD_SLSQP, num_ops)
    opt.set_min_objective(objective)
    opt.set_stopval(0.999)
    opt.add_inequality_constraint(constraint_W_psd, 1e-10)
    opt.add_inequality_constraint(constraint_symplectic_trace, 1e-10)
    opt.add_inequality_constraint(constraint_steering, 1e-6)
    opt.set_lower_bounds(-50 * np.ones(num_ops))
    opt.set_upper_bounds(50 * np.ones(num_ops))
    opt.set_xtol_rel(1e-7)
    opt.set_maxeval(100000)

    try:
        w_opt = opt.optimize(w0)
        min_value = opt.last_optimum_value()
        if stats['eval'] >= 500: print()
        return min_value, w_opt
    except Exception as e:
        if stats['eval'] >= 500: print()
        print(f"  Optimization failed: {e}")
        return np.inf, None

if __name__ == "__main__":
    # Main detection loop
    num_ops = 14
    entanglement_target = 1  # Target entanglement level

    print(f"Searching for steering witness for state with entanglement level {entanglement_target}...")

    # Generate target state once
    while True:
        state_g = randCM(entanglement_target)
        if state_g is not None:
            break
        print("  ...generating new state")

    attempt = 0
    while True:
        attempt += 1
        # Generate random measurements
        theta = np.pi * np.random.rand(num_ops)
        psi = np.pi * np.random.rand(num_ops)
        phi = 2 * np.pi * np.random.rand(num_ops)

        M_list = []
        m_list = []

        for idx in range(num_ops):
            M = measurement(theta[idx], psi[idx], phi[idx])
            M_list.append(M)
            m_list.append(np.real(np.trace(M @ state_g)))

        # Run optimization
        print(f"Attempt {attempt}: Running optimization with new random measurements...")
        min_val, w_opt = steering_detection(M_list, m_list, num_ops)
        
        if w_opt is not None:
            # Check constraints carefully
            # Set verbose=True to see why near-misses are occurring
            results = check_constraints(w_opt, M_list, min_val, num_ops, True)
            
            # Check if steering detected
            if results['all_constraints_ok']:
                print(f"\nSUCCESS! Steering detected on attempt {attempt}!")
                print(f"Final value: {min_val:.6f}")
                
                # Calculate and print diagonalized matrices for verification
                W_opt = np.sum([w_opt[k] * M_list[k] for k in range(num_ops)], axis=0)
                Z11 = W_opt[0:2, 0:2]
                Z22 = W_opt[2:4, 2:4]
                S1 = get_S(Z11)
                S2 = get_S(Z22)
                D1 = S1 @ Z11 @ S1.T
                D2 = S2 @ Z22 @ S2.T
                
                print("\nVerification of Williamson Decomposition for optimal W:")
                print("Diagonalized Z11 (S1 Z11 S1^T):")
                print(np.array2string(D1, precision=4, suppress_small=True))
                print("\nDiagonalized Z22 (S2 Z22 S2^T):")
                print(np.array2string(D2, precision=4, suppress_small=True))
                
                # Export weights to CSV
                filename_w = f"steering_weights_ent{entanglement_target}.csv"
                np.savetxt(filename_w, w_opt, delimiter=",", header="weight", comments="")
                print(f"\nFinal weights exported to: {filename_w}")
                
                # Export S matrices to CSV
                filename_s = f"symplectic_matrices_ent{entanglement_target}.csv"
                # Stack them for easy export
                S_stacked = np.vstack([S1, S2])
                np.savetxt(filename_s, S_stacked, delimiter=",", header="S1 (top 2 rows), S2 (bottom 2 rows)", comments="")
                print(f"Symplectic matrices exported to: {filename_s}")
                break
        
        print(f"  -> Current Result: {min_val:.6f} (No valid witness found yet)")
