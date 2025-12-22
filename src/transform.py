### simplified, scalable code
from src.utils.transform_utils import get_differences, get_commutators, get_poly_coeff, op_norm, get_vander
import scipy.sparse as sp
import numpy as np
from copy import deepcopy
from openfermion import FermionOperator, jordan_wigner, get_sparse_operator, count_qubits

def transform_op(op, G, theta, eigvals, reduce=False, verify=False, tol=1e-5, silent=True):
    """
    Transform operator op by exp(i G \theta), where G is an HERMITIAN operator with eigenvalues eigvals

    op: FermionOperator/QubitOperator - operator to be transformed
    G: Fermion/QubitOperator - generator of the unitary transform
    theta: np.float - amplitude of rotation
    eigvals: list[np.float] - real eigenvalues of G
    reduce: bool - Finds shortest exact transform by determining vanishing blocks
    verify: bool - verifies transformation using sparse matrices - (not scalable!)

    """
    def return_qubitop(op):
        if isinstance(op, FermionOperator): 
            return jordan_wigner(op) 
        else: 
            return op

    Delta = get_differences(eigvals)

    if reduce:
        ### determining relevant differences
        
        op_pauli = return_qubitop(op)
        G_pauli = return_qubitop(G)
        commutators = get_commutators(G_pauli, op_pauli, len(Delta))
        
        S = []
        Delta_current = deepcopy(Delta)
        for d in Delta:
            Delta_d = deepcopy(Delta_current)
            Delta_d.remove(d)
            coeffs = get_poly_coeff(Delta_d)

            fh = np.sum([c*t for c, t in zip(coeffs, commutators)])

            if op_norm(fh) > tol:
                S.append(d)
            else:
                Delta_current.remove(d)
        assert all([e in Delta_current for e in S]), "S does not match with Delta current!"
        if not silent: print("Found relevant eigenvalue differences: {}\nProceeding to transform".format(S))
    else:
        S = Delta
    
    ### constructing transformed operator
    commutators = get_commutators(G, op, len(S)) #to retain supplied operator form
    e_vec = np.exp(1.j * theta * np.array(S))
    W = get_vander(S)
    a = np.linalg.inv(W) @ e_vec
    transformed_op = sum([ai*comm for ai, comm in zip(a, commutators)])

    if not silent: print("Transformed operator in terms of commutators with coefficients:\n", a)

    if verify:
        # full operator for verification
        n_qubits = max([count_qubits(G), count_qubits(op)])
        U = sp.linalg.expm(1.j * get_sparse_operator(G, n_qubits) * theta)
        UHUd = U @ get_sparse_operator(op, n_qubits) @ U.getH()

        tol =1e-5
        assert np.sum(np.abs(get_sparse_operator(transformed_op, n_qubits) - UHUd)) < tol, "Error in transformation!"
    
    return transformed_op