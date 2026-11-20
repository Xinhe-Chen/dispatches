#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################
"""
Submit the PT2-style wind+PEM price-taker sweep WITH PEM stack replacement and
hydrogen liquid-path delivery costs.

Sweep (identical shape to the PT2 / PCM enumeration, Table
tab:parameter_sweep_space):

    Omega^pem / Omega^gen : [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    PEM bid B [$/MWh]     : [15, 20, 25, 30, 35, 40, 45]
    -> 56 jobs

WHY THE HYDROGEN PRICES ARE NOT 0.75 ... 2.25 ANY MORE
------------------------------------------------------
Without costs the bid was PROPORTIONAL to the hydrogen price, B = psi*rho, so
the PCM bid grid 15..45 $/MWh mapped onto rho = B/20 = 0.75..2.25 $/kg.

With the stack replacement reserve and the delivery charge the relation becomes
AFFINE:

    B = psi^eff * (1 - eps^del) * rho^H2
        - [ psi^eff * (1 - eps^del) * chi^del + 1000 * phi^rep ]

so to hit the SAME bids the delivered hydrogen price must be obtained by
inverting it:

    rho^H2 = (B + 1000 * phi^rep) / (psi^eff * (1 - eps^del)) + chi^del

That is what h2_price_for_bid() below does. Sweeping the bid rather than the
price is the right choice here: the saved PCM enumeration is indexed by
(Omega^pem, B), so matching bids keeps every price-taker case directly
comparable to its PCM counterpart with no new Prescient runs.

MARKET: the driver takes `market` from load_parameters.py. For a PT2-style
real-time sweep set `market = "RT"` there before submitting. If it is left at
"DA-RT" the driver needs a day-ahead stage already run with the SAME delivery
and replacement arguments, and will raise FileNotFoundError otherwise.
"""
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))

# ===========================================================================
# Scenario: PEM stack replacement (DOE current status) + complete liquid
# delivery path. Edit here; the hydrogen prices follow automatically.
# ===========================================================================
DELIVERY_COST = 3.2        # chi^del   [$/kg delivered]  complete liquid path
DELIVERY_LOSS = 0.0        # eps^del   [-]               0 for the base liquid case
DELIVERY_FIXED = 0.0       # kappa^del [$/yr]
COND_ENERGY = 0.0          # theta^cond [kWh/kg]         0 = off-taker service
STACK_LIFE = 40000.0       # Lambda^stack [operating h]  DOE 2022 status
STACK_REPL_FRAC = 0.11     # sigma^stack [-]             DOE Program Record 24005

# Behind-the-meter liquefaction variant (rotates the line as well as shifting
# it). Uncomment to run it instead:
# DELIVERY_LOSS = 0.02
# COND_ENERGY = 11.0

# --- constants that MUST match the model -----------------------------------
# RE_flowsheet.py : m.fs.pem.electricity_to_mol.fix(0.00275984)
# load_parameters.py : h2_mols_per_kg = 500, pem_cap_cost = 1200
PEM_ELECTRICITY_TO_MOL = 0.00275984
H2_MOLS_PER_KG = 500.0
PEM_CAP_COST = 1200.0
PSI_PEM = PEM_ELECTRICITY_TO_MOL * 3600 / H2_MOLS_PER_KG * 1000   # ~19.871 kg/MWh

DRY_RUN = False            # True -> print the qsub commands without submitting

# ===========================================================================
# Derived coefficients of the affine bid relation
# ===========================================================================
PSI_EFF = PSI_PEM / (1.0 + COND_ENERGY * PSI_PEM / 1000.0)
PHI_REP = (STACK_REPL_FRAC * PEM_CAP_COST / STACK_LIFE) if STACK_LIFE > 0 else 0.0
SLOPE = PSI_EFF * (1.0 - DELIVERY_LOSS)                    # $/MWh per $/kg
INTERCEPT = -(SLOPE * DELIVERY_COST + 1000.0 * PHI_REP)    # $/MWh


def h2_price_for_bid(bid):
    """Delivered hydrogen price [$/kg] that produces the PEM bid `bid` [$/MWh]."""
    return (bid + 1000.0 * PHI_REP) / SLOPE + DELIVERY_COST


def bid_for_h2_price(price):
    """Inverse check: PEM bid [$/MWh] implied by a delivered price [$/kg]."""
    return SLOPE * price + INTERCEPT


# submit price-taker jobs for wind PEM case study.

def submit_job(
    sim_id,
    PEM_ratio=0.1,
    H2_price=0.8,
    market='RT'
):

    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)

    file_name = os.path.join(
        job_scripts_dir, f"wind_PEM_price_taker_h2cost_{market}_sim_{sim_id}.sh"
    )
    with open(file_name, "w") as f:
        f.write(
            "#!/bin/bash\n"
            + "#$ -M xchen24@nd.edu\n"
            + "#$ -m a\n"
            + "#$ -q long\n"
            + f"#$ -N wind_PEM_pt_h2cost_sim_{sim_id}\n"
            + "conda activate ../../../../../../../afs/crc.nd.edu/user/x/xchen24/.conda/envs/regen\n"
            + "export LD_LIBRARY_PATH=~/.conda/envs/regen/lib:$LD_LIBRARY_PATH \n"
            + "module load gurobi\n"
            + "module load ipopt/3.14.2 \n"
            + "python ./run_pricetaker_wind_PEM_with_h2_costs.py"
            + f" --PEM_ratio {PEM_ratio}"
            + f" --H2_price {H2_price}"
            + f" --delivery_cost {DELIVERY_COST}"
            + f" --delivery_loss {DELIVERY_LOSS}"
            + f" --delivery_fixed {DELIVERY_FIXED}"
            + f" --cond_energy {COND_ENERGY}"
            + f" --stack_life {STACK_LIFE}"
            + f" --stack_repl_frac {STACK_REPL_FRAC}"
        )

    if DRY_RUN:
        print(f"  [dry run] qsub {file_name}")
    else:
        os.system(f"qsub {file_name}")


if __name__ == "__main__":
    pem_ratio_range = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    pem_bid_range = [15, 20, 25, 30, 35, 40, 45]        # $/MWh, the PCM grid

    # delivered hydrogen price for each PCM bid, rounded for tidy filenames
    h2_price_range = [round(h2_price_for_bid(b), 4) for b in pem_bid_range]

    print("=" * 78)
    print("Affine bid relation  B = slope * rho^H2 + intercept")
    print(f"  psi^pem   = {PSI_PEM:9.4f} kg/MWh")
    print(f"  psi^eff   = {PSI_EFF:9.4f} kg/MWh  (theta^cond = {COND_ENERGY:g} kWh/kg)")
    print(f"  phi^rep   = {PHI_REP:9.6f} $/kWh = {PHI_REP*1000:.2f} $/MWh "
          f"(sigma={STACK_REPL_FRAC:g}, Lambda={STACK_LIFE:g} h)")
    print(f"  chi^del   = {DELIVERY_COST:9.4f} $/kg,  eps^del = {DELIVERY_LOSS:g}")
    print(f"  slope     = {SLOPE:9.4f} ($/MWh)/($/kg)")
    print(f"  intercept = {INTERCEPT:9.4f} $/MWh")
    print(f"  rho_min   = {-INTERCEPT/SLOPE:9.4f} $/kg  (below this the PEM never runs)")
    print("-" * 78)
    print(f"{'PEM bid B':>10} {'rho^H2 delivered':>18} {'netback':>10} {'B realized':>12}")
    print("-" * 78)
    for b, p in zip(pem_bid_range, h2_price_range):
        netback = (1.0 - DELIVERY_LOSS) * (p - DELIVERY_COST)
        print(f"{b:10.1f} {p:18.4f} {netback:10.4f} {bid_for_h2_price(p):12.4f}")
    print("=" * 78)
    print(f"Submitting {len(pem_ratio_range) * len(pem_bid_range)} jobs"
          + (" (DRY RUN)" if DRY_RUN else "") + "\n")

    sim_id = 0
    for i in h2_price_range:
        for j in pem_ratio_range:
            sim_id += 1
            PEM_ratio = j
            H2_price = i
            submit_job(sim_id, PEM_ratio, H2_price)
