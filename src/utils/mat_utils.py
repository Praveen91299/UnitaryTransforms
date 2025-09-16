import numpy as np
import scipy.sparse as spr

def is_close_to_identity(A, tol=1e-6):
    if not spr.issparse(A):
        raise ValueError("Input matrix must be sparse.")

    identity = spr.eye(A.shape[0], format=A.format)  # Create sparse identity matrix
    diff = A - identity  # Compute difference
    max_diff = np.abs(diff).max()  # Maximum absolute entry

    return max_diff < tol  # Check if within tolerance

def is_hermitian(A, tol=1e-10):
    """
    Checks if a sparse matrix A is Hermitian (A = A.H).
    
    Parameters:
        A (scipy.sparse matrix): Input sparse matrix.
        tol (float): Tolerance for numerical errors.
        
    Returns:
        bool: True if A is Hermitian, False otherwise.
    """
    if not spr.issparse(A):
        #print("Hermiticity check: Matrix not sparse, converting to sparse to check hermiticity.")
        A = spr.csc_matrix(A, dtype=complex)
    
    # Check if square
    if A.shape[0] != A.shape[1]:
        print("Hermiticity check: Matrix not square.")
        return False  # Non-square matrices cannot be Hermitian
    
    # Compute difference between A and its conjugate transpose
    diff = A - A.getH()  # A.getH() is equivalent to A.conj().T for sparse matrices
    
    # Check if the maximum absolute entry in diff is within tolerance
    return np.abs(diff).max() < tol

def is_antihermitian(A, tol=1e-10):
    if not spr.issparse(A):
        #print("Anti-Hermiticity check: Matrix not sparse, converting to sparse to check anti-hermiticity.")
        A = spr.csc_matrix(A, dtype=complex)
    
    # Check if square
    if A.shape[0] != A.shape[1]:
        print("Anti hermiticity check: Matrix not square.")
        return False  # Non-square matrices cannot be Hermitian

    return is_hermitian(1.j*A)

def is_unitary(U, tol=1e-10):
    """
    Checks if a sparse matrix is unitary (U @ U.getH() = I = U.getH() @ U)

    """
    if not spr.issparse(U):
        #print("Unitary check: Matrix is not sparse, converting to sparse to check unitarity.")
        U = spr.csc_matrix(U)
    
    if U.shape[0] != U.shape[1]:
        print("Unitary check: Matrix not square.")
        return False
    
    U_dag = U.getH()
    return is_close_to_identity(U @ U_dag, tol) and is_close_to_identity(U_dag @ U, tol)