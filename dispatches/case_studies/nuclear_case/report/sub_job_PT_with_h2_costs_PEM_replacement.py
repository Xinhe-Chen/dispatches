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
Submit the nuclear (NE) price-taker sweep WITH PEM stack replacement and hydrogen
liquid-path delivery costs.

Sweep (identical shape to the original NE enumeration):

    Omega^pem / Omega^gen : [0.05, 0.1, ..., 0.5]      (10 ratios, set in the driver)
    PEM bid B [$/MWh]     : [15, 20, 25, 30, 35, 40]   (6 bids)
    -> 60 cases per market variant

USAGE
-----
    python sub_job_PT_with_h2_costs_PEM_replacement.py --markets RT
    python sub_job_PT_with_h2_costs_PEM_replacement.py --markets RT --dry-run
    python sub_job_PT_with_h2_costs_PEM_replacement.py --markets DA DA-RT
    python sub_job_PT_with_h2_costs_PEM_replacement.py --help

With no arguments it submits DA and then DA-RT, which is the default two-settlement
study. --dry-run writes the job scripts and prints the qsub commands without queueing
anything, which is the quickest way to check the hydrogen prices and output paths.

ONE JOB PER MARKET VARIANT, NOT ONE PER CASE
--------------------------------------------
The wind driver takes a single (ratio, price) pair per job, so its submit script
fans out 56 jobs. The NE driver instead enumerates the whole grid inside
`run_exhaustive_enumeration` and writes one results json per market variant, with
the case keys "<price index><capacity index>". Splitting a market across jobs would
restart those indices at 0 in every job and the fragments could not be merged, so
each market variant is submitted as one job of 60 LPs.

WHY THE HYDROGEN PRICES ARE NOT 0.75 ... 2 ANY MORE
---------------------------------------------------
Without costs the bid was PROPORTIONAL to the hydrogen price, B = psi * rho, so the
original NE grid 15..40 $/MWh mapped onto rho = B/20 = 0.75..2 $/kg.

With the stack replacement reserve and the delivery charge the relation becomes
AFFINE:

    B = psi^eff * (1 - eps^del) * rho^H2
        - [ psi^eff * (1 - eps^del) * chi^del + phi^rep ]

so to hit the SAME bids the delivered hydrogen price must be obtained by inverting
it,

    rho^H2 = (B + phi^rep) / (psi^eff * (1 - eps^del)) + chi^del,

which gives 1.6279 ... 3.1687 $/kg, the range reported in the manuscript's parameter
table for the NE case. Sweeping the bid rather than the price keeps every NE case
directly comparable to its PCM and surrogate counterpart, and keeps the surrogate
inputs inside the range the networks were trained on, [15, 40] $/MWh.

The driver itself is indexed by the bid, so this script only has to pass --bids; it
computes the prices here purely to print the table below.

DA-RT DEPENDS ON DA
-------------------
The two-step variant reads the day-ahead dispatch produced by the "DA" sweep. When
both are submitted, the DA-RT job is held until the DA job finishes (-hold_jid).
Submitting DA-RT on its own assumes the DA sweep has already been run with the same
cost arguments; the driver raises FileNotFoundError otherwise.
"""
import os
import subprocess

this_file_path = os.path.dirname(os.path.realpath(__file__))

# ===========================================================================
# Scenario: PEM stack replacement (DOE current status) + complete liquid
# delivery path. The hydrogen prices follow automatically.
#
# chi^del, eps^del and kappa^del are passed to the driver on the command line.
# theta^cond, sigma^stack and Lambda^stack are constants of
# price_taker_analysis_with_h2_costs_PEM_replacement, because the surrogate
# module imports them from there; the values below are cross-checked against it
# at the bottom of this file, so a mismatch is caught before anything is queued.
# ===========================================================================
DELIVERY_COST = 0.50       # chi^del   [$/kg delivered]  terminal 0.21 + truck 0.29
DELIVERY_LOSS = 0.005      # eps^del   [-]               99.5% liquefaction mass eff
DELIVERY_FIXED = 0.0       # kappa^del [$/yr]
COND_ENERGY = 11.0         # theta^cond [kWh/kg]         liquefaction behind the meter
STACK_LIFE = 40000.0       # Lambda^stack [operating h]  DOE 2022 status
STACK_REPL_FRAC = 0.11     # sigma^stack [-]             DOE Program Record 24005

# The plant owns and runs the liquefier, so it pays in electricity (COND_ENERGY)
# rather than in a service fee, and DELIVERY_COST covers only the stages it still
# buys. Argonne IN025 slide 18 splits the liquid path as liquefaction 2.69 +
# terminal 0.21 + truck 0.29 = 3.19 $/kg, so DELIVERY_COST = 0.21 + 0.29 = 0.50 and
# COND_ENERGY = 11 kWh/kg.
#
# EXTERNAL-SERVICE variant (off-taker owns the liquefier). Its all-in fee already
# includes the liquefier and its power, so COND_ENERGY must be 0 or liquefaction is
# double-counted:
# DELIVERY_COST = 3.2
# COND_ENERGY = 0.0

# --- constants that MUST match the model -----------------------------------
# price_taker_analysis_with_h2_costs_PEM_replacement.py :
#     PSI_PEM = 0.00275984 * 3600 / 500 * 1000,  NPP_CAPACITY = 400
PEM_ELECTRICITY_TO_MOL = 0.00275984
H2_MOLS_PER_KG = 500.0
PEM_CAP_COST = 1200.0      # gamma^pem [$/kW]
PSI_PEM = PEM_ELECTRICITY_TO_MOL * 3600 / H2_MOLS_PER_KG * 1000   # ~19.871 kg/MWh

DRIVER = "price_taker_analysis_with_h2_costs_PEM_replacement.py"
CONDA_ENV = "regen"
EMAIL = "xchen24@nd.edu"
QUEUE = "long"

DRY_RUN = False            # True -> print the qsub commands without submitting

# ===========================================================================
# Derived coefficients of the affine bid relation
# ===========================================================================
PSI_EFF = PSI_PEM / (1.0 + COND_ENERGY * PSI_PEM / 1000.0)
PHI_REP = (STACK_REPL_FRAC * PEM_CAP_COST * 1000.0 / STACK_LIFE) if STACK_LIFE > 0 else 0.0
SLOPE = PSI_EFF * (1.0 - DELIVERY_LOSS)                 # $/MWh per $/kg
INTERCEPT = -(SLOPE * DELIVERY_COST + PHI_REP)          # $/MWh


def h2_price_for_bid(bid):
    """Delivered hydrogen price [$/kg] that produces the PEM bid `bid` [$/MWh]."""
    return (bid + PHI_REP) / SLOPE + DELIVERY_COST


def bid_for_h2_price(price):
    """Inverse check: PEM bid [$/MWh] implied by a delivered price [$/kg]."""
    return SLOPE * price + INTERCEPT


def fmt(x):
    """Formats a float without a trailing .0, as the driver's file names do."""
    return str(int(x)) if float(x).is_integer() else str(x)


def cost_tag():
    """Folder suffix the driver builds for these cost assumptions."""
    return ("cd" + fmt(DELIVERY_COST)
            + "_loss" + fmt(DELIVERY_LOSS)
            + "_cond" + fmt(COND_ENERGY)
            + "_kfix" + fmt(DELIVERY_FIXED)
            + "_rep" + fmt(round(PHI_REP, 2)))


def check_driver_has_cli():
    """
    Refuses to submit if the driver next to this script predates its command line.

    An older driver has no argument parser, so it would silently ignore --market and
    run whatever variant its own __main__ hard-codes. This reads the file as text, so
    it works on a submit node without pyomo or idaes.
    """
    driver_path = os.path.join(this_file_path, DRIVER)
    if not os.path.isfile(driver_path):
        raise SystemExit("Driver not found: " + driver_path)

    with open(driver_path) as f:
        text = f.read()

    required = ["def _parse_args", '"--market"', '"--bids"', '"--delivery_cost"']
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(
            "The driver at\n  " + driver_path + "\nis an older version without the "
            "command-line options this script passes (missing: "
            + ", ".join(missing) + ").\nIt would ignore --market and run the "
            "variant hard-coded in its __main__. Copy the current "
            + DRIVER + " next to this script and re-run. Nothing was queued."
        )


def check_against_driver():
    """
    Verifies that the constants above agree with the model.

    Importing the driver needs pyomo and idaes, which the submit node may not have,
    so a failed import is reported and tolerated rather than raised.
    """
    import sys
    if this_file_path not in sys.path:
        sys.path.insert(0, this_file_path)

    try:
        from price_taker_analysis_with_h2_costs_PEM_replacement import (
            PSI_PEM as M_PSI_PEM,
            PSI_EFF as M_PSI_EFF,
            THETA_COND as M_THETA,
            CHI_DEL as M_CHI,
            EPS_DEL as M_EPS,
            KAPPA_DEL as M_KAPPA,
            SIGMA_STACK as M_SIGMA,
            LAMBDA_STACK as M_LAMBDA,
            replacement_reserve,
        )
    except Exception as exc:                       # pragma: no cover
        print("  [note] could not import the driver to cross-check "
              "({0}: {1}); relying on the literals above".format(
                  type(exc).__name__, exc))
        return

    mismatches = []
    for name, here, there in [
        ("psi^pem", PSI_PEM, M_PSI_PEM),
        ("psi^eff", PSI_EFF, M_PSI_EFF),
        ("theta^cond", COND_ENERGY, M_THETA),
        ("sigma^stack", STACK_REPL_FRAC, M_SIGMA),
        ("Lambda^stack", STACK_LIFE, M_LAMBDA),
        ("phi^rep", PHI_REP, replacement_reserve(PEM_CAP_COST)),
    ]:
        if abs(here - there) > 1e-9 * max(1.0, abs(there)):
            mismatches.append("{0}: submit script {1!r} vs model {2!r}".format(
                name, here, there))

    # chi^del, eps^del and kappa^del are passed on the command line, so they are
    # allowed to differ from the module defaults; say so rather than failing.
    for name, here, there in [("chi^del", DELIVERY_COST, M_CHI),
                              ("eps^del", DELIVERY_LOSS, M_EPS),
                              ("kappa^del", DELIVERY_FIXED, M_KAPPA)]:
        if abs(here - there) > 1e-12:
            print("  [note] {0} = {1} overrides the module default {2}".format(
                name, here, there))

    if mismatches:
        raise SystemExit(
            "Submit script and model disagree; nothing was queued.\n  "
            + "\n  ".join(mismatches)
            + "\nEdit the constants in " + DRIVER + " and re-run."
        )
    print("  constants cross-checked against " + DRIVER + ": OK")


def submit_job(market, pem_bids, hold_jid=None):
    """
    Writes and submits one job script for a single price-taker variant.

    Args:
        market: "DA", "RT", "Max-DA-RT" or "DA-RT"
        pem_bids: PEM bids to sweep [$/MWh]
        hold_jid: name of a job this one must wait for, or None

    Returns:
        The job name.
    """
    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)

    job_name = "ne_pt_h2cost_" + market.replace("-", "_")
    file_name = os.path.join(job_scripts_dir, job_name + ".sh")

    bids = " ".join(fmt(b) for b in pem_bids)

    lines = [
        "#!/bin/bash",
        "#$ -M " + EMAIL,
        "#$ -m a",
        "#$ -q " + QUEUE,
        "#$ -N " + job_name,
        "#$ -cwd",
    ]
    if hold_jid is not None:
        # The two-step variant reads the schedules written by the DA sweep.
        lines.append("#$ -hold_jid " + hold_jid)

    lines += [
        "conda activate " + CONDA_ENV,
        "export LD_LIBRARY_PATH=~/.conda/envs/" + CONDA_ENV
        + "/lib:$LD_LIBRARY_PATH ",
        "module load gurobi",
        "module load ipopt/3.14.2 ",
        # Results are written relative to the working directory, so move to the
        # folder holding the driver before running it.
        "cd " + this_file_path,
        "python ./" + DRIVER
        + " --market " + market
        + " --pem_capex " + fmt(PEM_CAP_COST)
        + " --bids " + bids
        + " --delivery_cost " + repr(DELIVERY_COST)
        + " --delivery_loss " + repr(DELIVERY_LOSS)
        + " --delivery_fixed " + repr(DELIVERY_FIXED),
    ]

    with open(file_name, "w") as f:
        f.write("\n".join(lines) + "\n")

    if DRY_RUN:
        print("  [dry run] qsub {0}".format(file_name))
    else:
        subprocess.call(["qsub", file_name])

    return job_name


def parse_args(argv=None):
    """Command-line options for this submit script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Submit the NE price-taker sweep with hydrogen conditioning, "
                    "delivery and PEM stack replacement costs.",
        epilog="Examples:\n"
               "  python %(prog)s --markets RT\n"
               "  python %(prog)s --markets RT --dry-run\n"
               "  python %(prog)s --markets DA DA-RT\n"
               "  python %(prog)s --markets Max-DA-RT --bids 15 25 35\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--markets", nargs="+", default=["DA", "DA-RT"],
        choices=["DA", "RT", "Max-DA-RT", "DA-RT"],
        help="Price-taker variants to submit, one job each "
             "(default: DA DA-RT). DA is always submitted first when present, "
             "because DA-RT reads its schedules.",
    )
    parser.add_argument(
        "--bids", type=float, nargs="+", default=[15, 20, 25, 30, 35, 40],
        help="PEM bids to sweep [$/MWh] (default: 15 20 25 30 35 40)",
    )
    parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=DRY_RUN,
        help="Print the qsub commands and write the job scripts without submitting",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli = parse_args()
    DRY_RUN = cli.dry_run

    # The driver sweeps the capacity ratios internally: [0.05, 0.1, ..., 0.5]
    pem_bid_range = list(cli.bids)                      # $/MWh, the PCM grid

    # Submit "DA" before "DA-RT": the two-step variant reads the day-ahead schedules.
    markets = ([m for m in cli.markets if m == "DA"]
               + [m for m in cli.markets if m != "DA"])

    # delivered hydrogen price for each bid, for the table below only
    h2_price_range = [round(h2_price_for_bid(b), 4) for b in pem_bid_range]

    print("=" * 78)
    print("Affine bid relation  B = slope * rho^H2 + intercept")
    print("  psi^pem   = {0:9.4f} kg/MWh".format(PSI_PEM))
    print("  psi^eff   = {0:9.4f} kg/MWh  (theta^cond = {1:g} kWh/kg)".format(
        PSI_EFF, COND_ENERGY))
    print("  phi^rep   = {0:9.4f} $/MWh  (sigma={1:g}, Lambda={2:g} h, "
          "gamma^pem={3:g} $/kW)".format(
              PHI_REP, STACK_REPL_FRAC, STACK_LIFE, PEM_CAP_COST))
    print("  chi^del   = {0:9.4f} $/kg,  eps^del = {1:g},  kappa^del = {2:g} $/yr".format(
        DELIVERY_COST, DELIVERY_LOSS, DELIVERY_FIXED))
    print("  slope     = {0:9.4f} ($/MWh)/($/kg)".format(SLOPE))
    print("  intercept = {0:9.4f} $/MWh".format(INTERCEPT))
    print("  rho_min   = {0:9.4f} $/kg  (below this the PEM never runs)".format(
        -INTERCEPT / SLOPE))
    print("-" * 78)
    print("{0:>10} {1:>18} {2:>10} {3:>12}".format(
        "PEM bid B", "rho^H2 delivered", "netback", "B realized"))
    print("-" * 78)
    for b, p in zip(pem_bid_range, h2_price_range):
        netback = (1.0 - DELIVERY_LOSS) * (p - DELIVERY_COST)
        print("{0:10.1f} {1:18.4f} {2:10.4f} {3:12.4f}".format(
            b, p, netback, bid_for_h2_price(p)))
    print("=" * 78)

    check_driver_has_cli()
    check_against_driver()

    tag = cost_tag()
    print("\nResults will be written to:")
    for market in markets:
        folder = market + "_" + tag
        results_json = "price_taker_pem_{0}_{1}_{2}.json".format(
            int(PEM_CAP_COST), market, tag)
        print("  " + os.path.join(this_file_path, folder, results_json))
    print("\nSubmitting {0} job{1} ({2} cases each){3}\n".format(
        len(markets), "" if len(markets) == 1 else "s",
        len(pem_bid_range) * 10, " (DRY RUN)" if DRY_RUN else ""))

    da_job = None
    for market in markets:
        hold = da_job if market == "DA-RT" else None

        if market == "DA-RT" and hold is None:
            # DA is not part of this submission, so its schedules must already exist.
            step1 = os.path.join(this_file_path, "DA_" + tag)
            if not os.path.isdir(step1):
                raise SystemExit(
                    "DA-RT needs the step-1 schedules in\n  " + step1
                    + "\nand that folder does not exist. Submit DA first:\n"
                    "  python " + os.path.basename(__file__) + " --markets DA\n"
                    "or submit both together:\n"
                    "  python " + os.path.basename(__file__)
                    + " --markets DA DA-RT"
                )
            print("  [note] DA-RT will read the existing schedules in " + step1)

        print("Submitting {0}{1}".format(
            market, " (held until " + hold + " finishes)" if hold else ""))
        name = submit_job(market, pem_bid_range, hold_jid=hold)
        if market == "DA":
            da_job = name
