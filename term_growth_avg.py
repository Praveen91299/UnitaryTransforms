import csv
import json
import platform
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openfermion import (
    MolecularData,
    count_qubits,
    get_fermion_operator,
    jordan_wigner,
)
from openfermionpyscf import run_pyscf

from src.transform import transform_op
from src.utils.op_utils import two_body_ucc_generators


# -------------------------------------------------------------------------
# Molecular Hamiltonian
# -------------------------------------------------------------------------

BASIS_NAME = "sto-3g"
MULTIPLICITY = 1
CHARGE = 0


def build_geometry(R, n_H=4):
    """Construct a linear H_n chain with spacing R in Angstrom."""
    assert n_H % 2 == 0, f"Odd number {n_H} of hydrogen atoms specified."
    return [("H", (0.0, 0.0, i * R)) for i in range(n_H)]


def build_H_chain_for_R(R, n_H=4, filename=None):
    """Construct the electronic Hamiltonian for a linear H_n chain."""
    geometry = build_geometry(R, n_H)

    molecule = MolecularData(
        geometry,
        BASIS_NAME,
        MULTIPLICITY,
        CHARGE,
        filename=filename,
    )
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)

    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    fermionic_hamiltonian = get_fermion_operator(molecular_hamiltonian)

    return fermionic_hamiltonian, molecule


def count_nonidentity_terms(operator):
    """Count all terms except the identity term, represented by ()."""
    return sum(term != () for term in operator.terms)


def count_pauli_types(term):
    n_x = sum(pauli == "X" for _, pauli in term)
    n_y = sum(pauli == "Y" for _, pauli in term)
    return n_x, n_y


# -------------------------------------------------------------------------
# Experiment settings
# -------------------------------------------------------------------------

SEED = 0
N_TRIALS = 10
N_TRANSFORMATIONS = 10

# Fit points x = 0, 1, 2, 3, 4.
FIT_LAST_TRANSFORMATION = 4

# Analytically predicted multiplicative factor.
PREDICTED_FACTOR = 11.0

# Used for confidence intervals.
N_BOOTSTRAP = 5000

# Set this to True if the original analysis allowed the same generator
# to be selected more than once within a run.
SAMPLE_WITH_REPLACEMENT = True

# All generated artifacts are collected here. Re-running the script replaces
# the previous artifacts with a complete, mutually consistent result set.
RESULTS_DIR = Path("saved/results")
GENERATORS_PATH = RESULTS_DIR / "generators.json"
TERM_COUNTS_PATH = RESULTS_DIR / "term_counts.csv"
SETTINGS_PATH = RESULTS_DIR / "experiment_settings.txt"
PLOT_PATH = RESULTS_DIR / "term_growth_fit.pdf"
MOLECULE_DATA_PATH = RESULTS_DIR / "molecule"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Construct Hamiltonian and generators
# -------------------------------------------------------------------------

R = 1.0
N_HYDROGEN = 4
H, molecule = build_H_chain_for_R(
    R,
    n_H=N_HYDROGEN,
    filename=str(MOLECULE_DATA_PATH),
)

HQ = jordan_wigner(H)
n_qubits = count_qubits(H)
generators = list(two_body_ucc_generators(n_qubits))

initial_term_count = count_nonidentity_terms(HQ)

print(f"Number of qubits: {n_qubits}")
print(f"Number of available generators: {len(generators)}")
print(f"Initial non-identity terms: {initial_term_count}")

if not SAMPLE_WITH_REPLACEMENT:
    if N_TRANSFORMATIONS > len(generators):
        raise ValueError(
            "N_TRANSFORMATIONS exceeds the number of available generators "
            "when sampling without replacement."
        )


# -------------------------------------------------------------------------
# Run independent transformation sequences
# -------------------------------------------------------------------------

rng = np.random.default_rng(SEED)
all_term_counts = []
all_generator_indices = []
all_angles = []
check_violations = True

for trial in range(N_TRIALS):
    generator_indices = rng.choice(
        len(generators),
        size=N_TRANSFORMATIONS,
        replace=SAMPLE_WITH_REPLACEMENT,
    )

    angles = rng.uniform(
        low=0.0,
        high=2.0 * np.pi,
        size=N_TRANSFORMATIONS,
    )

    transformed_hamiltonian = deepcopy(HQ)
    term_counts = [count_nonidentity_terms(transformed_hamiltonian)]

    for generator_index, angle in zip(generator_indices, angles):
        qubit_generator = jordan_wigner(generators[generator_index])

        transformed_hamiltonian = transform_op(
            transformed_hamiltonian,
            qubit_generator,
            angle,
            [-1, 0, 1],
        )

        term_counts.append(
            count_nonidentity_terms(transformed_hamiltonian)
        )

    all_term_counts.append(term_counts)
    all_generator_indices.append(generator_indices)
    all_angles.append(angles)

    if check_violations:
        parity_violations = []
        reality_violations = []

        for term, coefficient in transformed_hamiltonian.terms.items():
            n_x, n_y = count_pauli_types(term)

            # Fermion parity requires an even number of X and Y factors together.
            if (n_x + n_y) % 2 != 0:
                parity_violations.append(term)

            # Reality requires an even number of Y factors.
            if n_y % 2 != 0:
                reality_violations.append(term)

        print("Parity violations:", len(parity_violations))
        print("Reality violations:", len(reality_violations))
        print(
            "Symmetry-allowed non-identity maximum:",
            (4**n_qubits + 2**(n_qubits + 1)) // 4 - 1,
        )

        check_violations = False  # Check only the first trial.

all_term_counts = np.asarray(all_term_counts, dtype=int)
all_generator_indices = np.asarray(all_generator_indices, dtype=int)
all_angles = np.asarray(all_angles, dtype=float)

x = np.arange(N_TRANSFORMATIONS + 1)


# -------------------------------------------------------------------------
# Fit each run separately
#
# log(N_r(x) / N_r(0)) = beta_r x,
# where the growth factor for run r is b_r = exp(beta_r).
# -------------------------------------------------------------------------

x_fit_data = x[: FIT_LAST_TRANSFORMATION + 1]

log_term_ratios = np.log(
    all_term_counts[:, : FIT_LAST_TRANSFORMATION + 1]
    / all_term_counts[:, [0]]
)

# Least-squares slope with the intercept constrained to zero.
log_slopes = (
    log_term_ratios @ x_fit_data
    / np.dot(x_fit_data, x_fit_data)
)

growth_factors = np.exp(log_slopes)

# Averaging log slopes gives the geometric-mean growth factor.
mean_log_slope = np.mean(log_slopes)
mean_growth_factor = np.exp(mean_log_slope)

print(f"Number of trials: {N_TRIALS}")
print(
    "Geometric-mean fitted growth factor: "
    f"{mean_growth_factor:.3f}"
)
print(
    "Median fitted growth factor: "
    f"{np.median(growth_factors):.3f}"
)
print(
    "Growth-factor interquartile range: "
    f"[{np.percentile(growth_factors, 25):.3f}, "
    f"{np.percentile(growth_factors, 75):.3f}]"
)


# -------------------------------------------------------------------------
# Pointwise geometric means and bootstrap confidence intervals
# -------------------------------------------------------------------------

log_counts = np.log(all_term_counts)
geometric_mean_counts = np.exp(np.mean(log_counts, axis=0))

bootstrap_rng = np.random.default_rng(SEED + 1)
bootstrap_indices = bootstrap_rng.integers(
    low=0,
    high=N_TRIALS,
    size=(N_BOOTSTRAP, N_TRIALS),
)

bootstrap_geometric_means = np.exp(
    np.mean(log_counts[bootstrap_indices], axis=1)
)

lower_counts, upper_counts = np.percentile(
    bootstrap_geometric_means,
    [2.5, 97.5],
    axis=0,
)

# Bootstrap confidence interval for the mean logarithmic slope.
bootstrap_mean_slopes = np.mean(
    log_slopes[bootstrap_indices],
    axis=1,
)
growth_factor_ci = np.exp(
    np.percentile(bootstrap_mean_slopes, [2.5, 97.5])
)

print(
    "95% bootstrap confidence interval: "
    f"[{growth_factor_ci[0]:.3f}, {growth_factor_ci[1]:.3f}]"
)


# -------------------------------------------------------------------------
# Save reproducibility data
# -------------------------------------------------------------------------

# Store the FermionOperator terms without relying on its display format. Each
# ladder operator is represented by its orbital index and action (1=create,
# 0=annihilate); complex coefficients are split into JSON-native components.
serialized_generators = []
for generator_index, generator in enumerate(generators):
    serialized_terms = []
    for term, coefficient in generator.terms.items():
        serialized_terms.append(
            {
                "operators": [
                    {"orbital": int(orbital), "action": int(action)}
                    for orbital, action in term
                ],
                "coefficient": {
                    "real": float(np.real(coefficient)),
                    "imag": float(np.imag(coefficient)),
                },
            }
        )
    serialized_generators.append(
        {"generator_index": generator_index, "terms": serialized_terms}
    )

with GENERATORS_PATH.open("w", encoding="utf-8") as output_file:
    json.dump(
        {
            "format": "OpenFermion FermionOperator terms",
            "action_convention": {"1": "creation", "0": "annihilation"},
            "n_qubits": n_qubits,
            "generators": serialized_generators,
        },
        output_file,
        indent=2,
    )
    output_file.write("\n")

# Long-form output gives one row for every iteration of every run. The
# generator and angle on a row are those applied to reach that iteration;
# consequently, they are empty for iteration zero.
with TERM_COUNTS_PATH.open("w", encoding="utf-8", newline="") as output_file:
    writer = csv.DictWriter(
        output_file,
        fieldnames=[
            "run",
            "iteration",
            "term_count",
            "generator_index",
            "angle_radians",
        ],
    )
    writer.writeheader()
    for trial in range(N_TRIALS):
        for iteration in range(N_TRANSFORMATIONS + 1):
            row = {
                "run": trial + 1,
                "iteration": iteration,
                "term_count": int(all_term_counts[trial, iteration]),
                "generator_index": "",
                "angle_radians": "",
            }
            if iteration > 0:
                row["generator_index"] = int(
                    all_generator_indices[trial, iteration - 1]
                )
                row["angle_radians"] = format(
                    all_angles[trial, iteration - 1], ".17g"
                )
            writer.writerow(row)


def installed_version(distribution_name):
    """Return a package version without making result saving fragile."""
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "not available"


maximum_term_count = (
    4**n_qubits + 2**(n_qubits + 1)
) // 4 - 1

settings_lines = [
    "MOLECULE",
    f"hydrogen_chain_spacing_angstrom = {R}",
    f"number_of_hydrogen_atoms = {N_HYDROGEN}",
    f"basis_name = {BASIS_NAME}",
    f"multiplicity = {MULTIPLICITY}",
    f"charge = {CHARGE}",
    f"number_of_qubits = {n_qubits}",
    "",
    "EXPERIMENT",
    f"random_seed = {SEED}",
    f"number_of_trials = {N_TRIALS}",
    f"transformations_per_trial = {N_TRANSFORMATIONS}",
    f"sample_with_replacement = {SAMPLE_WITH_REPLACEMENT}",
    "angle_distribution = uniform [0, 2*pi)",
    "transform_op_eigenvalues = [-1, 0, 1]",
    f"fit_iterations = 0 through {FIT_LAST_TRANSFORMATION}",
    "fit_intercept_constrained_to_zero = True",
    f"predicted_growth_factor = {PREDICTED_FACTOR}",
    f"bootstrap_samples = {N_BOOTSTRAP}",
    f"bootstrap_seed = {SEED + 1}",
    "confidence_level_percent = 95",
    "term_count_excludes_identity = True",
    "generator_indices_are_zero_based = True",
    "",
    "RESULT SUMMARY",
    f"number_of_available_generators = {len(generators)}",
    f"initial_nonidentity_term_count = {initial_term_count}",
    f"symmetry_allowed_nonidentity_maximum = {maximum_term_count}",
    f"geometric_mean_growth_factor = {mean_growth_factor:.17g}",
    f"median_growth_factor = {np.median(growth_factors):.17g}",
    f"growth_factor_ci_lower = {growth_factor_ci[0]:.17g}",
    f"growth_factor_ci_upper = {growth_factor_ci[1]:.17g}",
    "",
    "SOFTWARE",
    f"python = {platform.python_version()}",
    f"numpy = {installed_version('numpy')}",
    f"matplotlib = {installed_version('matplotlib')}",
    f"scipy = {installed_version('scipy')}",
    f"openfermion = {installed_version('openfermion')}",
    f"openfermionpyscf = {installed_version('openfermionpyscf')}",
    f"pyscf = {installed_version('pyscf')}",
    "",
    "FILES",
    f"generators = {GENERATORS_PATH.name}",
    f"term_counts_and_transformations = {TERM_COUNTS_PATH.name}",
    f"plot = {PLOT_PATH.name}",
    f"molecular_data = {MOLECULE_DATA_PATH.name}.hdf5",
]

with SETTINGS_PATH.open("w", encoding="utf-8") as output_file:
    output_file.write("\n".join(settings_lines) + "\n")


# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.2, 4.2))

# Individual runs.
for trial_counts in all_term_counts:
    ax.plot(
        x,
        trial_counts,
        color="tab:blue",
        alpha=0.12,
        linewidth=0.8,
        marker=".",
        markersize=2.5,
    )

# Geometric-mean points and bootstrap uncertainty.
ax.errorbar(
    x,
    geometric_mean_counts,
    yerr=[
        geometric_mean_counts - lower_counts,
        upper_counts - geometric_mean_counts,
    ],
    fmt="o",
    color="tab:blue",
    markersize=4.5,
    capsize=2.5,
    linewidth=1.0,
    label="Geometric mean (95% CI)",
    zorder=3,
)

# Empirical exponential fit. Plot only over the fitted range.
x_fit = np.linspace(0, FIT_LAST_TRANSFORMATION, 200)
empirical_fit = initial_term_count * mean_growth_factor**x_fit

ax.plot(
    x_fit,
    empirical_fit,
    linestyle="--",
    color="black",
    linewidth=1.5,
    label=(
        rf"Empirical fit: "
        rf"$N_0({mean_growth_factor:.2f})^x$"
    ),
)

# Theoretical prediction, including finite Pauli-space saturation.
x_prediction = np.linspace(0, N_TRANSFORMATIONS, 400)
#maximum_term_count = 4**n_qubits - 1

predicted_growth = (
    initial_term_count * PREDICTED_FACTOR**x_prediction
)
predicted_growth = np.minimum(
    predicted_growth,
    maximum_term_count,
)

ax.plot(
    x_prediction,
    predicted_growth,
    linestyle="-",
    color="tab:red",
    linewidth=1.5,
    label=(
        rf"Predicted: $N_0({PREDICTED_FACTOR:.0f})^x$ "
        rf"(symmetry-capped)"
    ),
)

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xlim(0, N_TRANSFORMATIONS)
ax.set_xlabel("Number of transformations")
ax.set_ylabel("Number of non-identity Pauli-product terms")
ax.legend(frameon=True, fontsize=8)
ax.grid(axis="y", which="both", alpha=0.2)

fig.tight_layout()
fig.savefig(PLOT_PATH, bbox_inches="tight")
print(f"Saved reproducibility data and plot to {RESULTS_DIR.resolve()}")
plt.show()
