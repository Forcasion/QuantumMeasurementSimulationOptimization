import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steering_detection import randCM, symplectic_values

# ── helpers ────────────────────────────────────────────────────────────────────

def is_symmetric(M, tol=1e-8):
    return np.allclose(M, M.T, atol=tol)

def is_positive_definite(M, tol=1e-9):
    eigvals = np.linalg.eigvalsh(M)
    return np.all(eigvals > -tol)

def is_physical_CM(M, tol=1e-9):
    """A valid covariance matrix must be symmetric and positive definite."""
    return is_symmetric(M) and is_positive_definite(M)

def log_negativity(cm):
    """Compute log negativity for a 2-mode (4x4) CM."""
    det_11 = np.linalg.det(cm[0:2, 0:2])
    det_22 = np.linalg.det(cm[2:4, 2:4])
    det_12 = np.linalg.det(cm[0:2, 2:4])
    det_cm = np.linalg.det(cm)

    f = (0.5 * (det_11 + det_22) - det_12
         - np.sqrt(np.maximum(0, (0.5 * (det_11 + det_22) - det_12)**2 - det_cm)))

    return -0.5 * np.log2(np.maximum(1e-10, f))

def symplectic_eigenvalues_williamson(cm):
    """Extract symplectic eigenvalues via Williamson decomposition."""
    from scipy.linalg import block_diag
    size = len(cm)
    n = size // 2
    J = np.array([[0, 1], [-1, 0]])
    Omega = block_diag(*([J] * n))
    A = 1j * Omega @ cm
    vals = np.linalg.eigvals(A)
    pos_vals = np.sort(vals[vals.real > 0].real)[::-1]
    return pos_vals

# ── tests ──────────────────────────────────────────────────────────────────────

class TestRandCM:

    # --- Basic validity ---

    def test_returns_matrix_not_none(self):
        """randCM should find a valid state within its search budget."""
        cm = randCM(entg=1, n_modes=1)
        assert cm is not None, "randCM returned None — search budget exhausted"

    def test_output_shape_1mode(self):
        cm = randCM(entg=1, n_modes=1)
        assert cm is not None
        assert cm.shape == (4, 4), f"Expected (4,4), got {cm.shape}"

    def test_output_shape_2mode(self):
        cm = randCM(entg=1, n_modes=2)
        assert cm is not None
        assert cm.shape == (8, 8), f"Expected (8,8), got {cm.shape}"

    def test_is_symmetric(self):
        cm = randCM(entg=1, n_modes=1)
        assert cm is not None
        assert is_symmetric(cm), "CM is not symmetric"

    def test_is_positive_definite(self):
        cm = randCM(entg=1, n_modes=1)
        assert cm is not None
        assert is_positive_definite(cm), \
            f"CM is not positive definite, min eigval = {np.min(np.linalg.eigvalsh(cm)):.6f}"

    def test_is_physical_CM(self):
        cm = randCM(entg=1, n_modes=1)
        assert cm is not None
        assert is_physical_CM(cm), "CM fails physicality check (not symmetric PD)"

    # --- Symplectic eigenvalues (uncertainty principle) ---

    def test_symplectic_eigenvalues_geq_1(self):
        """All symplectic eigenvalues must be >= 1 for a physical state."""
        cm = randCM(entg=1, n_modes=1)
        assert cm is not None
        symp_eigs = symplectic_eigenvalues_williamson(cm)
        assert np.all(symp_eigs >= 1.0 - 1e-6), \
            f"Symplectic eigenvalues below 1: {symp_eigs}"

    # --- Entanglement level ---

    def test_entanglement_level_1(self):
        """Returned CM should match the requested entanglement level."""
        entg = 1
        cm = randCM(entg=entg, n_modes=1)
        assert cm is not None
        EN = log_negativity(cm)
        EN_rounded = np.round(EN * 10) / 10
        assert EN_rounded == entg / 5.0, \
            f"Expected EN={entg/5.0}, got EN_rounded={EN_rounded} (raw={EN:.4f})"

    def test_entanglement_level_2(self):
        entg = 2
        cm = randCM(entg=entg, n_modes=1)
        assert cm is not None
        EN = log_negativity(cm)
        EN_rounded = np.round(EN * 10) / 10
        assert EN_rounded == entg / 5.0, \
            f"Expected EN={entg/5.0}, got EN_rounded={EN_rounded} (raw={EN:.4f})"

    def test_entanglement_level_3(self):
        entg = 3
        cm = randCM(entg=entg, n_modes=1)
        assert cm is not None
        EN = log_negativity(cm)
        EN_rounded = np.round(EN * 10) / 10
        assert EN_rounded == entg / 5.0, \
            f"Expected EN={entg/5.0}, got EN_rounded={EN_rounded} (raw={EN:.4f})"

    # --- Mode ordering convention ---

    def test_interleaved_convention_1mode(self):
        """For n_modes=1, check that block structure is [x_A,p_A | x_B,p_B].
        Alice's block is cm[0:2,0:2], Bob's is cm[2:4,2:4].
        Both should be positive definite (non-degenerate subsystems).
        """
        cm = randCM(entg=1, n_modes=1)
        assert cm is not None
        alice = cm[0:2, 0:2]
        bob   = cm[2:4, 2:4]
        assert is_positive_definite(alice), "Alice's reduced CM is not PD"
        assert is_positive_definite(bob),   "Bob's reduced CM is not PD"

    def test_interleaved_convention_2mode(self):
        """For n_modes=2, each party's reduced 4x4 CM should be PD."""
        cm = randCM(entg=1, n_modes=2)
        assert cm is not None
        alice = cm[0:4, 0:4]
        bob   = cm[4:8, 4:8]
        assert is_positive_definite(alice), "Alice's 2-mode reduced CM is not PD"
        assert is_positive_definite(bob),   "Bob's 2-mode reduced CM is not PD"

    # --- Symplectic trace (used by steering constraint) ---

    def test_symplectic_trace_positive(self):
        """symplectic_values should return a positive value for a valid CM block."""
        cm = randCM(entg=1, n_modes=1)
        assert cm is not None
        sTr = symplectic_values(cm[0:2, 0:2])
        assert sTr > 0, f"Symplectic trace is non-positive: {sTr}"

    # --- Multiple samples consistency ---

    def test_multiple_samples_all_physical(self):
        """Generate several CMs and verify all are physical."""
        for i in range(5):
            cm = randCM(entg=1, n_modes=1)
            assert cm is not None, f"Sample {i} returned None"
            assert is_physical_CM(cm), f"Sample {i} is not a physical CM"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])