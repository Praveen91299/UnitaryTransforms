from src.utils.op_utils import two_body_ucc_generators
from openfermion import expectation, commutator, get_sparse_operator, count_qubits, jordan_wigner

#make hamiltonian
from math import sin, cos, radians
from openfermion import MolecularData, get_fermion_operator
from openfermionpyscf import run_pyscf
import numpy as np
#import pandas as pd
from src.transform import transform_op
from copy import deepcopy
import matplotlib.pyplot as plt

#construct Hamiltonian
BASIS_NAME = 'sto-3g'
MULTIPLICITY = 1
CHARGE = 0

def build_geometry(R, n_H=4):
    """
    Build molecular geometry for linear H4 chain.
    
    Parameters
    ----------
    R : float
        Bond distance in Angstroms.
        
    Returns
    -------
    list of tuple
        List of (atom_symbol, (x, y, z)) tuples.
    """
    assert n_H % 2 == 0, "Odd number {} of Hydrogens specified.".format(n_H)
    
    return [("H", (0.0, 0.0, i*R)) for i in range(n_H)]

def build_H_chain_for_R(R, n_H=4):
    """
    Build Hamiltonian for H4 at distance R.
    
    Parameters
    ----------
    R : float
        Bond distance in Angstroms.
        
    Returns
    -------
    H : FermionOperator
        Hamiltonian.
    mol : MolecularData
        OpenFermion molecule object with computed properties.
    """
    geom = build_geometry(R, n_H)
    mol = MolecularData(geom, BASIS_NAME, MULTIPLICITY, CHARGE)
    mol = run_pyscf(mol, run_scf=1, run_fci=1)
    H_mol = mol.get_molecular_hamiltonian()
    H_ferm = get_fermion_operator(H_mol)
    return H_ferm, mol


r =  1
H, mol = build_H_chain_for_R(r, 4) # build_lih(r)#
HQ = jordan_wigner(H)
n_qubits = count_qubits(H)

print("Initial Hamiltonian terms: {}".format(len(list(HQ.terms.keys()))))

gens = two_body_ucc_generators(n_qubits)


n_gen = 10

np.random.seed(0)
rand_idx = np.random.random_integers(0, len(gens), n_gen)

HQ_new = deepcopy(HQ)
n_terms = [len(HQ_new.terms)]
for i, r in enumerate(rand_idx):
    gen = gens[r]

    HQ_new = transform_op(HQ_new, jordan_wigner(gen), 2*np.pi*np.random.rand(), [-1, 0, 1])
    n_terms.append(len(HQ_new.terms))
    print(n_terms[-1])

n_terms_non_const = np.array(n_terms) - 1
print(n_terms)

#plt.plot(n_terms)
plt.yscale("log")

x = np.array(list(range(len(n_terms))))
plt.xticks(x, x)

plt.xlabel("Number of transformations")
plt.ylabel("Number of Pauli product terms")

#add line for 20x
#add fit line for upto 4 transforms

# Requires x > 0

y = n_terms_non_const
a, b = np.polyfit(x[:5], np.log(y[:5]), 1)

x_fit = np.linspace(x.min(), x.max(), 200)
y_fit = (np.e ** b) * (np.e**a) ** x_fit # b

plt.scatter(x, y, label="Term count")
plt.plot(x_fit, y_fit, '--', label=f"Fit", color='black')

x_fit = np.linspace(x.min(), x.max(), 200)
y_fit = (y[0]) * (11) ** x_fit # b

plt.plot(x_fit, y_fit, '-', label=f"Predicted", color='red')


plt.ylim(top=1e5, bottom=1e2)
plt.legend()
plt.savefig('./term_growth_fit_new.pdf', dpi=1000)
plt.show()