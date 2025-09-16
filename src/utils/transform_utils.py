### utility functions for exact and approximate closed form BCH expansions
import numpy as np
from openfermion import FermionOperator, commutator, normal_ordered, jordan_wigner, QubitOperator, count_qubits, get_sparse_operator
from itertools import combinations
from src.utils.mat_utils import is_hermitian

def get_poly_coeff(roots):
    """
    Get binomial coeffs of polynomial with given roots
    
    """

    d = len(roots)
    coeffs = np.zeros(d+1, complex)
    
    for i in range(d+1):
        root_combs = combinations(roots, i)

        coeffs[i] = ((-1)**i) *  np.sum([np.prod(c, dtype=complex) for c in root_combs])
    return coeffs[::-1]

is_proj = lambda p, tol=1e-5: np.isclose(np.linalg.norm(p @ p - p, 'fro'), 0, atol=tol)

def get_vander(S, split=False, truncate_at_deg = -1):
    """
    Return Vandermonde matrix of with rows defined by S

    split (bool): If True, splits and returns odd and even columns as separate matrices
    
    """
    W = np.vander(S, increasing=True)

    if truncate_at_deg >= 0:
        W = W.T[:truncate_at_deg+1].T

    if split:
        W0, W1 = W.T[::2].T, W.T[1::2].T
        return W0, W1
    else:
        return W

def get_unique_diff(l, return_combs = False):
    S = []
    comb_dict = {}
    for a in l:
        for b in l:
            diff = a - b
            S.append(diff)

            if diff not in comb_dict:
                comb_dict[diff] = [(a, b)]
            else:
                comb_dict[diff].append((a, b))
    
    Sd = np.array(list(set(S)))
    if return_combs:
        return Sd, comb_dict
    else:
        return Sd

def get_commutators(G, h, deg = 0):
    """
    Get upto deg commutators

    """
    commutators = [h]
    for i in range(deg):
        commutators.append(commutator(G, commutators[-1]))
    
    return commutators

def proj(v, tol=1e-5):
    """
    Construct projector with given vector v

    """
    p  = (np.array([v], complex).T @ np.array([v], complex).conjugate()) / (np.linalg.norm(v, 2))
    assert is_proj(p, tol), f"Not a projector!"

    return p

def construct_projectors(G, n_qubits, tol = 1e-10):
    """
    Construct projectors to the distinct eigenspaces of hermitian G
    
    """

    tol_dec = - int(np.log10(tol))

    assert is_hermitian(get_sparse_operator(G, n_qubits)), "Operator not hermitian! Hermiticity required for constructing orthogonal projectors."

    eigenvalues, eigenvectors = np.linalg.eigh(get_sparse_operator(G, n_qubits).toarray())

    projectors = []
    eig_unique = np.sort(list(set(np.around(eigenvalues, tol_dec))))
    e_vecs = []

    for i, e_val in enumerate(eig_unique):
        e_vec = []
        projector = np.zeros((1<<n_qubits, 1<<n_qubits), complex)

        for j, u in enumerate(eigenvalues):
            if np.isclose(e_val, u, atol=tol):
                p = proj(eigenvectors[:, j], tol)
                projector += p
                e_vec.append(eigenvectors[:, j])
        
        assert is_proj(projector, tol), f"Not a projector: e_val = {e_val}"
        e_vecs.append(e_vec)
        projectors.append(projector)
    
    ### verifying completeness of projectors
    assert np.isclose(np.linalg.norm(sum(projectors) - np.identity(1<<n_qubits), 'fro'), 0, atol=tol), f"{np.linalg.norm(sum(projectors) - np.identity(1<<n_qubits))}"
    
    for i, p in enumerate(projectors):
        assert (np.isclose(np.linalg.norm(get_sparse_operator(G, n_qubits) @ p - eig_unique[i]*p, 'fro'), 0, atol=tol)), "Projector yielding incorrect magnitude of eigen value."

    return projectors, eig_unique, e_vecs

def get_block_norms(H, projectors, silent = True, tol=1e-10):
    """
    Prints block frobenius norms and returns index pairs of non-zero blocks

    """
    tol_dec = -int(np.log10(tol))

    n_blocks = len(projectors)
    blocks = []
    block_norms = np.zeros((n_blocks, n_blocks))

    for i, pi in enumerate(projectors):
        for j, pj in enumerate(projectors):
            #print("e1: {}, e2: {}, diff {}".format(e_vals[i], e_vals[j], e_vals[i] - e_vals[j]))

            b = pi @ H @ pj
            blocks.append(b)

            fro_norm = np.round(np.linalg.norm(b, 'fro'), tol_dec)
            block_norms[i, j] = fro_norm

            if not silent: print(f"Block ({i}, {j}): {fro_norm}")
    
    return blocks, block_norms

def get_S(G, H, verbose=True, tol=1e-5):
    """
    Returns set of eigenvalues of G and set S of eigenvalue differences over non-vanishing projector blocks

    G, H : FermionOperators/QubitOperators

    """
    if verbose: print("G: {}\nH: {}".format(G, H))
    n_qubits = max(count_qubits(G), count_qubits(H))
    projectors, e_vals, e_vecs = construct_projectors(G, n_qubits, tol=tol)

    Hs = get_sparse_operator(H, n_qubits).toarray()
    bl, bl_norms = get_block_norms(Hs, projectors)

    if verbose: print("Block norms:\n", bl_norms)
    eval_diff = np.zeros(np.shape(bl_norms), complex)

    all_differences = []
    non_zero_differences = []
    for i, v1 in enumerate(e_vals):
        for j, v2 in enumerate(e_vals):
            eval_diff[i, j] = v1 - v2
            all_differences.append(v1 - v2)

            if bl_norms[i, j] > tol:
                non_zero_differences.append(v1 - v2)
    non_zero_differences = np.array(non_zero_differences)
    all_differences = np.array(all_differences)

    tol_dec = int(np.abs(np.log10(tol)))
    if verbose: print("Eigen values:\n", np.around(e_vals, 5))
    if verbose: print("Eigen value differences:\n", np.around(eval_diff, tol_dec))

    S = list(set(non_zero_differences))
    max_eig_diff_count = len(set(np.around(all_differences, tol_dec)))
    if verbose: print("Found {} unique eigenvalue differences over non-vanishing blocks:\n(maximum possible differences: {}) \n{}".format(len(S), max_eig_diff_count, S))

    return e_vals, S