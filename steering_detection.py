import traceback
import numpy as np
import nlopt
import math
import random
from scipy.linalg import block_diag, expm
from numpy import *

def symplectic_values(Z, A=None):
    """
    Return the symplectic eigenvalues trace for Z using a random symplectic transformation.
    The transformation S = exp(Omega @ A_symmetric) is independent of Z.
    """
    size = len(Z)
    n = size // 2
    
    # Symplectic form Omega
    Omega = np.zeros((size, size))
    Omega[0:n, n:2*n] = np.eye(n)
    Omega[n:2*n, 0:n] = -np.eye(n)
    
    if A is None:
        A = np.random.randn(size, size)
    
    A_symmetric = (A + A.T) / 2
    # Generator of symplectic group: Omega @ Symmetric
    H = Omega @ A_symmetric
    S = expm(H)

    result = S @ Z @ S.T
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

def check_constraints(w_opt, M_list, min_val, num_ops=14, verbose=False, A_list=None):
    """
    Verify all constraints are satisfied
    A_list: optional list of [A1, A2] random matrices for symplectic_values
    """
    W = np.sum([w_opt[k] * M_list[k] for k in range(num_ops)], axis=0)

    # Check 1: W ≥ 0 (PSD)
    eigvals = np.linalg.eigvalsh(W)
    min_eigval = np.min(np.real(eigvals))
    W_psd = min_eigval >= -1e-9

    # Check 2: sTr(Z11) + sTr(Z22) ≥ 0.5
    Z11 = W[0:2, 0:2]
    Z22 = W[2:4, 2:4]

    A1 = A_list[0] if A_list else None
    A2 = A_list[1] if A_list else None

    sTr_Z11 = symplectic_values(Z11, A1)
    sTr_Z22 = symplectic_values(Z22, A2)

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

    # Objective function: sum(w*m)
    def objective(w, grad):
        # Base objective
        obj = np.dot(w, m_list)
        
        # Calculate violations for progress reporting
        # W PSD violation
        W = np.sum([w[k] * M_list[k] for k in range(num_ops)], axis=0)
        min_eigval = np.min(np.real(np.linalg.eigvalsh(W)))
        v1 = float(np.max([0.0, -min_eigval]))
        
        # sTr violation
        s_val = symplectic_values(W[0:2, 0:2], A1) + symplectic_values(W[2:4, 2:4], A2)
        v2 = float(np.max([0.0, 0.5 - s_val]))
        
        # Steering violation
        v3 = float(np.max([0.0, obj - 0.999]))
        
        max_v = float(np.max([v1, v2, v3]))
        
        # Store individual violations for diagnostic printing on failure
        stats['v1'], stats['v2'], stats['v3'] = v1, v2, v3
        
        stats['eval'] += 1
        
        # Track best violation (how close we are to feasibility)
        if max_v < stats['best_v']:
            stats['best_v'] = max_v
            
        # Only track "best obj" if it's feasible-ish
        # 1e-9 matches the threshold in check_constraints
        if max_v < 1e-9:
            if stats['best_feasible_w'] is None or obj < stats['best_feasible_obj']:
                stats['best_feasible_w'] = w.copy()
                stats['best_feasible_obj'] = obj
            
            if obj < stats['best_obj']:
                stats['best_obj'] = obj
            
        if stats['eval'] % 500 == 0:
            print(f"\r    Eval: {stats['eval']:7d} | Curr obj: {obj:10.6f} | Best obj: {stats['best_obj']:10.6f} | Curr Viol: {max_v:10.6f}", end="", flush=True)

        if grad.size > 0:
            grad[:] = m_list
        return float(obj)

    # Pre-generate random matrices and fixed transformation components
    A1 = np.random.randn(2, 2)
    A2 = np.random.randn(2, 2)
    
    # Pre-calculate linear coefficients for sTr gradient
    # Since sTr(S Z S.T) is linear in Z, sTr(sum w_k M_k) = sum w_k sTr(M_k)
    sTr_coeffs = []
    for k in range(num_ops):
        val1 = symplectic_values(M_list[k][0:2, 0:2], A1)
        val2 = symplectic_values(M_list[k][2:4, 2:4], A2)
        sTr_coeffs.append(val1 + val2)
    sTr_coeffs = np.array(sTr_coeffs)

    # Constraint 1: W ≥ 0 (PSD) -> -min_eigval <= 0
    def constraint_W_psd(w, grad):
        W = np.sum([w[k] * M_list[k] for k in range(num_ops)], axis=0)
        eigvals, eigvecs = np.linalg.eigh(W)
        min_idx = np.argmin(eigvals)
        min_eigval = eigvals[min_idx]
        v_min = eigvecs[:, min_idx]

        if grad.size > 0:
            # Derivative of lambda_min is v^T @ (dW/dw) @ v
            # dW/dw_k is M_list[k]
            # Constraint is -lambda_min, so grad is -v^T @ M_k @ v
            for k in range(num_ops):
                grad[k] = -float(np.real(v_min.conj().T @ M_list[k] @ v_min))
        return float(-min_eigval)

    # Constraint 2: sTr(Z11) + sTr(Z22) ≥ 0.5 -> 0.5 - sTr_sum <= 0
    def constraint_symplectic_trace(w, grad):
        sTr_sum = np.dot(w, sTr_coeffs)
        val = 0.5 - sTr_sum

        if grad.size > 0:
            # Gradient is -sTr_coeffs
            grad[:] = -sTr_coeffs
        return float(val)

    # Constraint 3: w·m < 1 (Steering condition) -> obj - 0.999 <= 0
    def constraint_steering(w, grad):
        obj = np.dot(w, m_list)
        val = obj - 0.999
        if grad.size > 0:
            grad[:] = m_list
        return float(val)

    # Set up optimizer
    # LD_SLSQP is a robust gradient-based solver for nonlinear constraints.
    # Alternatively LD_AUGLAG with LD_LBFGS.
    opt = nlopt.opt(nlopt.LD_SLSQP, n_vars)

    opt.set_min_objective(objective)
    # Stop as soon as we find ANY steering witness
    opt.set_stopval(0.999)

    # Add all constraints with tighter tolerances for physical constraints
    opt.add_inequality_constraint(constraint_W_psd, 1e-10)
    opt.add_inequality_constraint(constraint_symplectic_trace, 1e-10)
    opt.add_inequality_constraint(constraint_steering, 1e-6)

    # Set bounds
    opt.set_lower_bounds(-20 * np.ones(n_vars))
    opt.set_upper_bounds(20 * np.ones(n_vars))

    # Set tolerances
    opt.set_xtol_rel(1e-6)
    opt.set_maxeval(1000000)

    # Initial guess
    w0 = np.random.randn(n_vars)

    # Optimize
    try:
        w_opt = opt.optimize(w0)
        min_value = opt.last_optimum_value()
        
        # Double check if we found something better during search
        if stats['best_feasible_w'] is not None and stats['best_feasible_obj'] < min_value:
             return stats['best_feasible_obj'], stats['best_feasible_w'], [A1, A2]
             
        if stats['eval'] >= 100:
            print() # New line after the progress print
        return min_value, w_opt, [A1, A2]
    except Exception as e:
        if stats['eval'] >= 100:
            print() # New line after the progress print
        
        # RECOVERY: If we found a feasible solution during search, return it!
        if stats['best_feasible_w'] is not None:
            print(f"  Optimization ended with error ({e}), but RECOVERED a feasible solution found during search.")
            return stats['best_feasible_obj'], stats['best_feasible_w'], [A1, A2]

        print(f"  Optimization failed: {e}")
        # Print a sample of the violation values on failure
        print(f"    Sample violations at failure -> PSD: {stats['v1']:.6f}, sTr: {stats['v2']:.6f}, Steering: {stats['v3']:.6f}")
        return np.inf, None, None

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
        min_val, w_opt, A_list = steering_detection(M_list, m_list, num_ops)
        
        if w_opt is not None:
            # Check constraints carefully
            # Set verbose=True to see why near-misses are occurring
            results = check_constraints(w_opt, M_list, min_val, num_ops, True, A_list)
            
            # Check if steering detected
            if results['all_constraints_ok']:
                print(f"\nSUCCESS! Steering detected on attempt {attempt}!")
                print(f"Final value: {min_val:.6f}")
                # Detailed check is already printed because of verbose=True above
                
                # Export weights to CSV
                filename = f"steering_weights_ent{entanglement_target}.csv"
                np.savetxt(filename, w_opt, delimiter=",", header="weight", comments="")
                print(f"Final weights exported to: {filename}")
                break
        
        print(f"  -> Current Result: {min_val:.6f} (No valid witness found yet)")
