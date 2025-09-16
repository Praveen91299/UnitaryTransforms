from openfermion import FermionOperator, commutator, normal_ordered, jordan_wigner, QubitOperator, count_qubits, get_sparse_operator

Kia     = lambda i, a: FermionOperator(f'{a}^ {i}', 1.0) - FermionOperator(f'{i}^ {a}', 1.0)
Tijab = lambda i, j, a, b: FermionOperator(f'{a}^ {b}^ {i} {j}', 1.0) - FermionOperator(f'{j}^ {i}^ {b} {a}', 1.0)
Kjaib   = lambda j, a, i, b: FermionOperator(f'{a}^ {b}^ {i} {j}', 1.0) - FermionOperator(f'{j}^ {i}^ {b} {a}', 1.0)

Tia     = lambda i, a: Kia(2*i, 2*a) + Kia(2*i + 1, 2*a + 1)
Tiiaa   = lambda i, a: Kjaib(2*i, 2*a, 2*i+1, 2*a+1)
Tiiab   = lambda i, a, b: Kjaib(2*i, 2*a, 2*i+1, 2*b+1) + Kjaib(2*i, 2*a + 1, 2*i+1, 2*b)
Tijaa   = lambda i, j, a: Kjaib(2*i, 2*a, 2*j+1, 2*a+1) + Kjaib(2*i+1, 2*a, 2*j, 2*a+1)