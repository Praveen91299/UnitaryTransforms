from openfermion import FermionOperator, QubitOperator
import numpy as np
from itertools import combinations

Kia     = lambda i, a: FermionOperator(f'{a}^ {i}', 1.0) - FermionOperator(f'{i}^ {a}', 1.0)
Tijab = lambda i, j, a, b: FermionOperator(f'{a}^ {b}^ {i} {j}', 1.0) - FermionOperator(f'{j}^ {i}^ {b} {a}', 1.0)
Kjaib   = lambda j, a, i, b: FermionOperator(f'{a}^ {b}^ {i} {j}', 1.0) - FermionOperator(f'{j}^ {i}^ {b} {a}', 1.0)

Tia     = lambda i, a: Kia(2*i, 2*a) + Kia(2*i + 1, 2*a + 1)
Tiiaa   = lambda i, a: Kjaib(2*i, 2*a, 2*i+1, 2*a+1)
Tiiab   = lambda i, a, b: Kjaib(2*i, 2*a, 2*i+1, 2*b+1) + Kjaib(2*i, 2*a + 1, 2*i+1, 2*b)
Tijaa   = lambda i, j, a: Kjaib(2*i, 2*a, 2*j+1, 2*a+1) + Kjaib(2*i+1, 2*a, 2*j, 2*a+1)

def op_norm(op):
    return sum(np.abs(list(op.terms.values())))


def has_even_y_real_coeffs(operator: QubitOperator, tol=1e-10):
    """
    Check every Pauli term has an even number of Y operators and a real coefficient.
    """
    for term, coeff in operator.terms.items():
        n_y = sum(pauli == 'Y' for _, pauli in term)
        if n_y % 2 != 0:
            return False
        if not np.isclose(np.imag(coeff), 0.0, atol=tol):
            return False
    return True


def two_body_ucc_generators(n_qubits):
    """
    Return all generalized two-body UCC generators on n_qubits spin-orbitals.

    Each generator has the Hermitian form
        -1j * (a_p^ a_q^ a_s a_r - a_r^ a_s^ a_q a_p)
    for distinct unordered spin-orbital pairs (p, q) and (r, s).
    """
    orbital_pairs = list(combinations(range(n_qubits), 2))
    generators = []

    for source_pair, target_pair in combinations(orbital_pairs, 2):
        r, s = source_pair
        p, q = target_pair
        excitation = FermionOperator(f'{p}^ {q}^ {s} {r}', 1.0)
        deexcitation = FermionOperator(f'{r}^ {s}^ {q} {p}', 1.0)
        generators.append(-1.j * (excitation - deexcitation))

    return generators

def find_commuting_paulis(H, sym_ops, verbose=False):
    """
    Finds Pauli products in H that commute with all sym_ops
    """
    def is_commuting(op1, op2, tol):
        comm = commutator(op1, op2)
        comm.compress()
        return np.isclose(np.sum(np.abs(list(comm.terms.values()))), 0, rtol=tol)
    
    HQ = deepcopy(H)
    c = HQ.constant
    HQ = HQ - c
    HQ.compress()
    n_total_pauli =  len(H.terms.keys())

    commuting_terms = []
    for term, coeff in H.terms.items():
        Pauli =  QubitOperator(term, coeff)

        if all([is_commuting(sym_op, Pauli, 1e-5) for sym_op in sym_ops]):
            commuting_terms.append(Pauli)
    
    if verbose: print("{}/{} Terms in H found to commute with all symmetries.".format(len(commuting_terms), n_total_pauli))

    return commuting_terms
