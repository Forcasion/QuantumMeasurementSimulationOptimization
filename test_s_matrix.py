import numpy as np
from steering_detection import get_S, symp_orth
from scipy.linalg import block_diag

# Set print options for better matrix visibility
np.set_printoptions(precision=4, suppress=True, linewidth=200)

def is_symplectic(S, rtol=1e-5, atol=1e-8):
    size = len(S)
    n = size // 2
    # Interleaved Omega for [x1, y1, x2, y2, ...]
    J = np.array([[0, 1], [-1, 0]])
    Omega = block_diag(*([J] * n))
    
    # S Omega S^T = Omega
    res = S @ Omega @ S.T
    return np.allclose(res, Omega, rtol=rtol, atol=atol)

def test_williamson_general(n_modes):
    size = 2 * n_modes
    print(f"\n{'='*20} Testing {size}x{size} Williamson decomposition ({n_modes} modes) {'='*20}")
    
    # Generate a random positive definite matrix
    A = np.random.rand(size, size)
    Z = A @ A.T + np.eye(size) * 0.5
    Z = (Z + Z.T) / 2
    
    print(f"Original Covariance Matrix Z ({size}x{size}):")
    print(Z)
    
    S = get_S(Z)
    
    is_symp = is_symplectic(S)
    print(f"\nS is symplectic: {is_symp}")
    if not is_symp:
        J_block = np.array([[0, 1], [-1, 0]])
        Omega = block_diag(*([J_block] * n_modes))
        print("S Omega S^T - Omega (Violation):")
        print(S @ Omega @ S.T - Omega)
    
    D = S @ Z @ S.T
    print(f"\nTransformed Matrix S Z S^T (Full {size}x{size}):")
    print(D)
    
    is_diag = np.allclose(D, np.diag(np.diag(D)), atol=1e-7)
    print(f"\nIs diagonal: {is_diag}")
    
    diag_vals = np.diag(D)
    # Check if diagonal elements have the structure [d1, d1, d2, d2, ...]
    diag_correct = np.allclose(diag_vals[::2], diag_vals[1::2], atol=1e-7)
    print(f"Diagonal structure [d1, d1, d2, d2, ...]: {diag_correct}")
    print(f"Symplectic eigenvalues (d_k): {diag_vals[::2]}")
    
    assert is_symp, f"S matrix is not symplectic for {size}x{size}"
    assert is_diag, f"S Z S^T is not diagonal for {size}x{size}"
    assert diag_correct, f"Diagonal does not have [d1, d1, d2, d2, ...] structure for {size}x{size}"
    print(f"--- {size}x{size} Test Passed! ---")

if __name__ == "__main__":
    try:
        # User specified tests
        test_williamson_general(1) # 2x2
        test_williamson_general(2) # 4x4
        test_williamson_general(4) # 8x8
        print("\n" + "#"*30)
        print("   ALL TESTS PASSED!   ")
        print("#"*30)
    except Exception as e:
        print(f"\n!!! TEST FAILED: {e} !!!")
        import traceback
        traceback.print_exc()
