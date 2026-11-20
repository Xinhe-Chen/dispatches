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
Wind + PEM price-taker model WITH hydrogen conditioning (liquefaction or
compression), road-transport costs, and PEM stack replacement.

MODEL EXTENSION -- PEM STACK REPLACEMENT
----------------------------------------
One stack replacement costs sigma^stack * gamma^pem * Omega^pem and is consumed
over Lambda^stack full-load-equivalent hours. Annual equivalent full-load hours
are sum_t P_t^pem / Omega^pem, so

   REPL = (sigma^stack * gamma^pem * Omega^pem / Lambda^stack)
          * sum_t P_t^pem / Omega^pem
        = phi^rep * sum_t P_t^pem ,     phi^rep = sigma^stack*gamma^pem/Lambda^stack

Omega^pem CANCELS, so replacement is a pure variable cost in $/kWh that adds
onto phi_2^pem (which is 0 in load_parameters.py). The model stays linear; no
binaries and no integer replacement counter. Manuscript Eq. (1h) becomes

   VOM = sum_t ( phi_2^gen P_t^gen + (phi_2^pem + phi^rep) P_t^pem )

and the KKT threshold gains the variable-cost term:

   rho_t^LMP  ><  psi^eff * rho^net - (phi_2^pem + phi^rep)*1000   [$/MWh]

So replacement LOWERS the optimal PEM bid. Unlike the delivery fee (a pure axis
shift) this is an axis REMAP of the saved PCM sweep,
   rho^H2 = (B + phi^rep*1000) / (psi^eff*(1-eps^del)) + chi^del,
which is still a reindexing of the existing (Omega^pem, B) grid -- no new
Prescient runs.

Default parameters are the DOE CURRENT STATUS case:
   sigma^stack = 0.11   (DOE Program Record 24005: 11% of total installed
                         capital, applied every 40,000 operating hours)
   Lambda^stack = 40,000 h  (DOE PEM technical targets, 2022 status;
                         80,000 h is the 2026 and ultimate target)
   gamma^pem = pem_cap_cost = 1200 $/kW  (load_parameters.py)
   -> phi^rep = 0.11*1200/40000 = 0.0033 $/kWh = 3.30 $/MWh

The delivery defaults are the LIQUID PATH WITH BEHIND-THE-METER LIQUEFACTION:
the plant owns the liquefier, so it pays in electricity rather than a service
fee. Splitting Argonne IN025 slide 18 (liquefaction 2.69 + terminal 0.21 +
truck 0.29 = 3.19 $/kg):

   chi^del    = 0.50 $/kg   terminal 0.21 + road transport 0.29
   theta^cond = 11 kWh/kg   liquefaction (NREL 88818 gives 9-13)
   eps^del    = 0.005       0.5% mass loss (NREL 88818, 99.5% mass efficiency)

Do NOT combine chi^del = 3.2 (the all-in EXTERNAL service, which already pays
for the liquefier and its power) with theta^cond > 0; that double-counts
liquefaction, and the script raises ValueError if you try.

Pass --stack_life 0 to disable replacement entirely.

Because phi^rep is levelized with no discounting, it overstates the true cost of
future outlays. The script reports the discount correction

   kappa(T) = (T / PA) * sum_{j=1..n} (1+r)^(-jT),   T = Lambda^stack / EFLH

so a second pass can be run with --stack_repl_frac scaled by kappa. kappa
depends on EFLH, which is an output, hence the two-pass structure.

NOTE: replacement sits inside VOM and is therefore deducted before tax wherever
tax is applied. This is LESS conservative than the current SI, which states
"include no extra replacement tax shield". (This script has no tax term.)

MODEL EXTENSION -- HYDROGEN DELIVERY
------------------------------------
Let  rho^H2    = hydrogen price AT THE CUSTOMER DELIVERY POINT   [$/kg]   --H2_price
     chi^del   = all-in delivery charge per delivered kg          [$/kg]   --delivery_cost
     eps^del   = delivery loss fraction not covered by chi^del    [-]      --delivery_loss
     kappa^del = fixed annual delivery service expense            [$/yr]   --delivery_fixed
     theta^cond= behind-the-meter conditioning electricity        [kWh/kg] --cond_energy

The manuscript equations (1e)-(1e''), (1o) are

     HPROD = sum_t psi^eff * P_t^pem
     HDEL  = (1 - eps^del) * HPROD
     HREV  = HDEL * rho^H2
     DCOST = HDEL * chi^del + kappa^del

with the effective yield (conditioning drawn from the same diverted power)

     psi^eff = psi^pem / (1 + theta^cond * psi^pem / 1000)

Because HREV - DCOST = HPROD * rho^net - kappa^del, where the NETBACK is

     rho^net = (1 - eps^del) * (rho^H2 - chi^del),

the extended problem is the ORIGINAL problem with
     rho^H2    -> rho^net
     psi^pem   -> psi^eff
minus the constant kappa^del. The constant does not change the argmax, so it is
applied after the solve. This is an exact reformulation, not an approximation:
dispatch, LMP response and the implied PEM bid are those of the original model
evaluated at the netback price.

Equivalently, since the flowsheet's yield psi^pem is fixed by
RE_flowsheet.py (m.fs.pem.electricity_to_mol), the conditioning derate is
folded into the price handed to the solver:

     price_to_model = rho^net * (psi^eff / psi^pem)

so that the dispatch threshold seen by the optimizer is
psi^pem * price_to_model = psi^eff * rho^net, which is the correct KKT
threshold for the extended model.

NOTE ON Omega^pem: when theta^cond > 0, `pem_mw` is the capacity of the whole
hydrogen train (electrolyzer + conditioning), not the electrolyzer alone. This
is what keeps the bid-curve geometry identical to the existing PCM sweep. If
instead the plant OWNS the conditioning equipment, add gamma^cond to
`pem_cap_cost` in load_parameters.py and set chi^del to trucking only.

NOTE ON TAX: this price-taker script has no corporate tax term (NPV = -CAPEX +
PA * annual_revenue). Tax is applied in post-processing elsewhere. DCOST is an
operating expense, so when tax IS applied it must be deducted before tax, as in
manuscript Eq. (1l).

BACKWARD COMPATIBILITY: with the default delivery arguments (all zero) this
script reproduces run_pricetaker_wind_PEM_new.py exactly, and writes to the
same filenames.
"""
import multiprocessing as mp
from itertools import product
import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import json
from dispatches.case_studies.renewables_case.wind_battery_PEM_LMP_new import wind_battery_pem_optimize
from dispatches.case_studies.renewables_case.RE_flowsheet import default_input_params
from dispatches.case_studies.renewables_case.load_parameters import (
    market, h2_mols_per_kg, PA, pem_cap_cost, discount_rate, N,
)

# ---------------------------------------------------------------------------
# PEM conversion factor. MUST match m.fs.pem.electricity_to_mol in
# RE_flowsheet.py (currently fixed at 0.00275984 mol/s per kW).
# psi^pem = electricity_to_mol * 3600 / h2_mols_per_kg * 1000  [kg/MWh]
# ---------------------------------------------------------------------------
PEM_ELECTRICITY_TO_MOL = 0.00275984                       # mol/s per kW
PSI_PEM = PEM_ELECTRICITY_TO_MOL * 3600 / h2_mols_per_kg * 1000   # ~19.871 kg/MWh

usage = "Solve wind_PEM price-taker model with H2 conditioning and delivery costs."
parser = ArgumentParser(usage)
parser.add_argument(
    "--PEM_ratio",
    dest="PEM_ratio",
    help="Set PEM power as a ratio to the wind farm",
    action="store",
    type=float,
    default=0.1,
)

parser.add_argument(
    "--H2_price",
    dest="H2_price",
    help="Hydrogen price at the CUSTOMER DELIVERY POINT in $/kg. "
         "With the delivery arguments left at zero this is the plant-gate "
         "price and the behaviour is identical to the original script.",
    action="store",
    type=float,
    default=1.0,
)

# ----------------------- delivery / conditioning arguments -----------------
parser.add_argument(
    "--delivery_cost",
    dest="delivery_cost",
    help="chi^del: delivery charge per DELIVERED kg [$/kg], covering only the "
         "stages the plant BUYS. Default 0.50 = terminal 0.21 + road transport "
         "0.29 (Argonne IN025 slide 18), i.e. liquefaction is done behind the "
         "meter via --cond_energy. Other anchors: 0.2 LH2 trucking only "
         "(NREL 88818, 200 km round trip); 0.9 GH2 trucking only (Argonne, "
         "400 km); 1.9 complete compressed-gas path; 3.2 complete liquid path "
         "as an EXTERNAL service -- use 3.2 only with --cond_energy 0, or "
         "liquefaction is double-counted.",
    action="store",
    type=float,
    default=0.50,
)
parser.add_argument(
    "--delivery_loss",
    dest="delivery_loss",
    help="eps^del: delivery loss fraction NOT already covered by "
         "--delivery_cost. Default 0.005 is the 0.5%% mass loss implied by the "
         "99.5%% liquefaction mass efficiency in NREL 88818. Use 0 for the gas "
         "path.",
    action="store",
    type=float,
    default=0.005,
)
parser.add_argument(
    "--delivery_fixed",
    dest="delivery_fixed",
    help="kappa^del: fixed annual delivery service expense [$/yr]. Constant, "
         "so it shifts NPV without changing the optimal dispatch.",
    action="store",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--cond_energy",
    dest="cond_energy",
    help="theta^cond: behind-the-meter conditioning electricity [kWh/kg]. "
         "Default 11 = liquefaction done by the plant (NREL 88818 gives 9-13). "
         "2.87 = 350-bar compression. Set 0 when conditioning is an off-taker "
         "service, and then raise --delivery_cost to the all-in fee. When > 0, "
         "pem_mw denotes the whole hydrogen train.",
    action="store",
    type=float,
    default=11.0,
)

# ----------------------- PEM stack replacement arguments -------------------
parser.add_argument(
    "--stack_life",
    dest="stack_life",
    help="Lambda^stack: PEM stack life in OPERATING HOURS. DOE PEM technical "
         "targets: 40000 (2022 status, default), 80000 (2026 and ultimate "
         "target). Set 0 to disable stack replacement.",
    action="store",
    type=float,
    default=40000.0,
)
parser.add_argument(
    "--stack_repl_frac",
    dest="stack_repl_frac",
    help="sigma^stack: stack replacement cost as a fraction of INSTALLED PEM "
         "capital. DOE Program Record 24005: 0.11 (default). Record 19009: "
         "0.15. Current SI stresses: 0.30, 0.60.",
    action="store",
    type=float,
    default=0.11,
)
options = parser.parse_args()

PEM_ratio = options.PEM_ratio
H2_price = options.H2_price                 # delivered price, $/kg
delivery_cost = options.delivery_cost       # chi^del,   $/kg delivered
delivery_loss = options.delivery_loss       # eps^del,   fraction
delivery_fixed = options.delivery_fixed     # kappa^del, $/yr
cond_energy = options.cond_energy           # theta^cond, kWh/kg
stack_life = options.stack_life             # Lambda^stack, operating hours
stack_repl_frac = options.stack_repl_frac   # sigma^stack, fraction of PEM capital

# ---------------------------------------------------------------------------
# Derived delivery economics
# ---------------------------------------------------------------------------
if not (0.0 <= delivery_loss < 1.0):
    raise ValueError(f"--delivery_loss must be in [0, 1); got {delivery_loss}")
if cond_energy < 0.0:
    raise ValueError(f"--cond_energy must be >= 0; got {cond_energy}")
if stack_life < 0.0:
    raise ValueError(f"--stack_life must be >= 0; got {stack_life}")
if stack_repl_frac < 0.0:
    raise ValueError(f"--stack_repl_frac must be >= 0; got {stack_repl_frac}")
# Guard against double-counting liquefaction: an all-in liquid service fee
# (~3.2 $/kg) already pays for the liquefier AND its electricity, so it must
# not be combined with behind-the-meter conditioning energy.
if cond_energy > 0.0 and delivery_cost > 1.5:
    raise ValueError(
        f"--delivery_cost {delivery_cost} with --cond_energy {cond_energy} "
        "double-counts liquefaction. Either buy the service "
        "(--delivery_cost 3.2 --cond_energy 0) or run the liquefier behind "
        "the meter (--delivery_cost 0.50 --cond_energy 11)."
    )

# ---------------------------------------------------------------------------
# PEM stack replacement reserve
#   phi^rep = sigma^stack * gamma^pem / Lambda^stack   [$/kWh]
# gamma^pem is pem_cap_cost from load_parameters.py, an INSTALLED cost, and the
# DOE sigma^stack values are fractions of installed capital, so the bases match.
# With theta^cond > 0, Omega^pem is the hydrogen-train capacity and sigma^stack
# is then the stack's share of the train's capital.
# ---------------------------------------------------------------------------
PHI_REP = 0.0                                          # $/kWh
if stack_life > 0.0:
    PHI_REP = stack_repl_frac * pem_cap_cost / stack_life

# psi^eff: effective yield after behind-the-meter conditioning [kg/MWh]
PSI_EFF = PSI_PEM / (1.0 + cond_energy * PSI_PEM / 1000.0)
YIELD_DERATE = PSI_EFF / PSI_PEM                      # = 1.0 when cond_energy = 0

# rho^net: netback received by the plant per kg PRODUCED [$/kg]
NETBACK = (1.0 - delivery_loss) * (H2_price - delivery_cost)

# price handed to the solver so the KKT threshold is psi^eff * rho^net
PRICE_TO_MODEL = NETBACK * YIELD_DERATE

# Tag so that runs with different delivery assumptions do not overwrite each
# other. Empty when no delivery costs -> original filenames are preserved.
_has_delivery = any(
    abs(x) > 0.0 for x in (delivery_cost, delivery_loss, delivery_fixed, cond_energy)
)
TAG = ""
if _has_delivery:
    TAG = (f"_cd{delivery_cost:g}_loss{delivery_loss:g}"
           f"_cond{cond_energy:g}_kfix{delivery_fixed:g}")
if PHI_REP > 0.0:
    TAG += f"_rep{PHI_REP * 1000:g}"

print("=" * 72)
print("PEM stack replacement")
print(f"  stack life        Lambda^stack = {stack_life:9.0f} operating h")
print(f"  replacement frac  sigma^stack  = {stack_repl_frac:9.4f} of installed capital")
print(f"  PEM capital       gamma^pem    = {pem_cap_cost:9.2f} $/kW")
print(f"  reserve           phi^rep      = {PHI_REP:9.6f} $/kWh "
      f"({PHI_REP * 1000:.2f} $/MWh)")
if PHI_REP == 0.0:
    print("  (stack replacement DISABLED)")
print("=" * 72)
print("Hydrogen delivery economics")
print(f"  delivered price   rho^H2     = {H2_price:9.4f} $/kg")
print(f"  delivery charge   chi^del    = {delivery_cost:9.4f} $/kg delivered")
print(f"  delivery loss     eps^del    = {delivery_loss:9.4f} -")
print(f"  fixed service     kappa^del  = {delivery_fixed:9.2f} $/yr")
print(f"  conditioning      theta^cond = {cond_energy:9.4f} kWh/kg")
print(f"  flowsheet yield   psi^pem    = {PSI_PEM:9.4f} kg/MWh")
print(f"  effective yield   psi^eff    = {PSI_EFF:9.4f} kg/MWh "
      f"(derate {YIELD_DERATE:.4f})")
print(f"  netback           rho^net    = {NETBACK:9.4f} $/kg produced")
print(f"  price to solver              = {PRICE_TO_MODEL:9.4f} $/kg")
print(f"  gross PEM bid                = {PSI_EFF * NETBACK:9.4f} $/MWh "
      "(before replacement)")
print(f"  implied PEM bid   B          = {PSI_EFF * NETBACK - PHI_REP * 1000:9.4f} $/MWh "
      "(after replacement)")
print("=" * 72)

if NETBACK <= 0.0:
    print("WARNING: netback <= 0. The delivery charge meets or exceeds the "
          "delivered price, so the optimal PEM dispatch is zero. The solve "
          "will still run.")


def delivery_summary(h2_kg_model):
    """
    Translate the model-internal hydrogen mass into physical produced,
    delivered, revenue and delivery-cost quantities.

    `h2_kg_model` is the mass the flowsheet reports, i.e. psi^pem * P^pem.
    The physical mass after behind-the-meter conditioning is that value times
    the yield derate.
    """
    hprod = h2_kg_model * YIELD_DERATE                   # HPROD  [kg/yr]
    hdel = hprod * (1.0 - delivery_loss)                 # HDEL   [kg/yr]
    hrev_gross = hdel * H2_price                         # HREV   [$/yr]
    dcost = hdel * delivery_cost + delivery_fixed        # DCOST  [$/yr]
    return {
        "h2_kg_model": h2_kg_model,
        "h2_produced_kg": hprod,
        "h2_delivered_kg": hdel,
        "h2_revenue_gross": hrev_gross,
        "delivery_cost_variable": hdel * delivery_cost,
        "delivery_cost_fixed": delivery_fixed,
        "delivery_cost_total": dcost,
        "h2_revenue_net": hrev_gross - dcost,
    }


def replacement_summary(pem_mw, pem_energy_kwh):
    """
    Derive the stack replacement schedule implied by the optimal dispatch.

    EFLH = E^pem / Omega^pem is an OUTPUT of the optimization, so the number of
    replacements over the project life is derived rather than assumed. This is
    the quantity Reviewer 1 asked for.

    Also returns the discount correction

        kappa(T) = (T / PA) * sum_{j=1..n} (1+r)^(-jT),  T = Lambda^stack/EFLH

    The levelized reserve phi^rep spreads one replacement evenly over T years
    with no discounting, so it overstates the present cost of future outlays by
    1/kappa. Re-run with --stack_repl_frac multiplied by kappa for a
    discount-consistent second pass.
    """
    out = {
        "stack_life_hours": stack_life,
        "stack_repl_frac": stack_repl_frac,
        "phi_rep_per_kwh": PHI_REP,
        "phi_rep_per_mwh": PHI_REP * 1000.0,
        "pem_energy_kwh": pem_energy_kwh,
        "annual_replacement_cost": PHI_REP * pem_energy_kwh,
    }
    if pem_mw <= 0.0 or pem_energy_kwh <= 0.0 or stack_life <= 0.0:
        out.update({"eflh_hours": 0.0, "pem_capacity_factor": 0.0,
                    "replacement_interval_years": None,
                    "n_replacements": 0, "kappa_discount": None,
                    "discounted_annual_replacement_cost": 0.0})
        return out

    eflh = pem_energy_kwh / (pem_mw * 1000.0)              # h/yr
    T = stack_life / eflh                                   # yr between replacements
    years, t = [], T
    while t < N - 1e-9:
        years.append(t)
        t += T
    disc_sum = sum((1.0 + discount_rate) ** (-y) for y in years)
    kappa = (T / PA) * disc_sum if years else 0.0

    out.update({
        "eflh_hours": eflh,
        "pem_capacity_factor": eflh / 8760.0,
        "replacement_interval_years": T,
        "n_replacements": len(years),
        "replacement_years": years,
        "kappa_discount": kappa,
        "discounted_annual_replacement_cost": kappa * PHI_REP * pem_energy_kwh,
    })
    return out


def run_design(h2_price, pem_ratio):
    """
    Runs the Wind + PEM pricetaker optimization for the given hydrogen selling price ($/kg) and optionally,
    a fixed PEM size that is the `pem_ratio` of the wind capacity. If `pem_ratio` is not provided, the PEM
    capacity is free to be optimized.

    `h2_price` is the DELIVERED price; the solver receives the netback
    PRICE_TO_MODEL derived above.
    """
    input_params = default_input_params.copy()
    # The solver sees the netback, not the delivered price.
    input_params['h2_price_per_kg'] = PRICE_TO_MODEL
    # Stack replacement reserve, added to the PEM variable cost inside the model.
    input_params['pem_repl_var_cost'] = PHI_REP
    input_params['extant_wind'] = True
    if pem_ratio == None:
        input_params['design_opt'] = "PEM"
    else:
        input_params["pem_mw"] = pem_ratio * input_params["wind_mw"]
        input_params['design_opt'] = False
    input_params["batt_mw"] = 0
    input_params["tank_size"] = 0

    result_file = file_dir / f"result_{market}_{h2_price}_{pem_ratio}{TAG}.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            res = json.load(f)
        res = {**input_params, **res}
        res.pop("DA_LMPs")
        res.pop("design_opt")
        res.pop("extant_wind")
        res.pop("wind_resource")
        print(f"Already complete: {h2_price} {pem_ratio}{TAG}")
        return res
    print(f"Running: {h2_price} {pem_ratio}{TAG}")
    des_res, solver_res, df_res = wind_battery_pem_optimize(
        # time_points=24 * 7,
        time_points=len(wind_cfs),
        input_params=input_params, market=market, verbose=False, plot=False)

    # ----------------------------------------------------------------------
    # Delivery post-processing.
    # df_res["h2_prod"] is in mol/h (flow_mol * 3600), so divide by
    # h2_mols_per_kg to get kg/h, then sum over the year.
    # ----------------------------------------------------------------------
    h2_kg_model = float(np.sum(df_res["h2_prod"].values)) / h2_mols_per_kg
    dsum = delivery_summary(h2_kg_model)

    # Consistency check: the model's own hydrogen revenue is evaluated at the
    # netback, so it must equal (gross revenue - variable delivery cost).
    expected = dsum["h2_revenue_gross"] - dsum["delivery_cost_variable"]
    if not np.isclose(des_res["annual_rev_h2"], expected, rtol=1e-6, atol=1.0):
        print("WARNING: delivery accounting mismatch. "
              f"model annual_rev_h2={des_res['annual_rev_h2']:.2f}, "
              f"expected={expected:.2f}. Check PEM_ELECTRICITY_TO_MOL against "
              "RE_flowsheet.py.")

    # kappa^del is a constant annual expense: it shifts NPV but not the optimum.
    des_res["NPV_with_delivery"] = des_res["NPV"] - PA * delivery_fixed
    des_res["NPV_ann_with_delivery"] = des_res["NPV_ann"] - delivery_fixed
    des_res["profit_with_delivery"] = des_res["profit"] - delivery_fixed

    # ----------------------------------------------------------------------
    # Stack replacement post-processing. phi^rep is already INSIDE the solve
    # (it changed the bid and the dispatch); what follows only derives the
    # implied schedule and the discount correction for a second pass.
    # ----------------------------------------------------------------------
    rsum = replacement_summary(
        des_res["pem_mw"], des_res.get("pem_energy_kwh", 0.0)
    )
    des_res["replacement"] = rsum
    # NPV if future outlays are discounted rather than levelized (pass 2 target).
    _repl_credit = (rsum["annual_replacement_cost"]
                    - rsum["discounted_annual_replacement_cost"])
    des_res["NPV_ann_discounted_repl"] = (
        des_res["NPV_ann_with_delivery"] + _repl_credit
    )

    des_res["delivery"] = {
        "h2_price_delivered": H2_price,
        "delivery_cost_per_kg": delivery_cost,
        "delivery_loss_fraction": delivery_loss,
        "delivery_fixed_annual": delivery_fixed,
        "cond_energy_kwh_per_kg": cond_energy,
        "psi_pem_kg_per_mwh": PSI_PEM,
        "psi_eff_kg_per_mwh": PSI_EFF,
        "yield_derate": YIELD_DERATE,
        "netback_per_kg": NETBACK,
        "price_to_model": PRICE_TO_MODEL,
        "implied_pem_bid_per_mwh": PSI_EFF * NETBACK,
        **dsum,
    }

    res = {**input_params, **des_res}
    res.pop("DA_LMPs")
    res.pop("design_opt")
    res.pop("extant_wind")
    res.pop("wind_resource")
    res.pop("pyo_model")
    if market =="DA-RT":
        res.pop("DA_dispatch")
    with open(result_file, 'w') as f:
        json.dump(res, f)
    df_res.to_csv(
        file_dir
        / f"wind_pem_{market}_{np.round(input_params['pem_mw'], 2)}_{H2_price}{TAG}.csv"
    )

    print(f"  H2 produced   {dsum['h2_produced_kg']/1e6:10.3f} kt/yr")
    print(f"  H2 delivered  {dsum['h2_delivered_kg']/1e6:10.3f} kt/yr")
    print(f"  H2 revenue    {dsum['h2_revenue_gross']/1e6:10.3f} M$/yr (gross, at delivered price)")
    print(f"  delivery cost {dsum['delivery_cost_total']/1e6:10.3f} M$/yr")
    print(f"  H2 revenue    {dsum['h2_revenue_net']/1e6:10.3f} M$/yr (net)")
    if PHI_REP > 0.0 and rsum["replacement_interval_years"]:
        print(f"  PEM EFLH      {rsum['eflh_hours']:10.1f} h/yr "
              f"(CF {rsum['pem_capacity_factor']*100:.1f}%)")
        print(f"  stack life    {rsum['replacement_interval_years']:10.2f} yr "
              f"-> {rsum['n_replacements']} replacement(s) over {N:.0f} yr")
        print(f"  REPL          {rsum['annual_replacement_cost']/1e6:10.3f} M$/yr (levelized, in-model)")
        print(f"  REPL          {rsum['discounted_annual_replacement_cost']/1e6:10.3f} M$/yr "
              f"(discounted, kappa={rsum['kappa_discount']:.3f})")
        print(f"  -> second pass: --stack_repl_frac "
              f"{stack_repl_frac * rsum['kappa_discount']:.5f}")
    print(f"  NPV_ann       {des_res['NPV_ann_with_delivery']/1e6:10.3f} M$/yr")
    print(f"Finished: {h2_price} {pem_ratio}{TAG}")
    return res

if __name__ == "__main__":
    shortfall = 1000
    wind_df = pd.read_parquet(Path(__file__).parent / "data" / f"303_LMPs_15_reserve_{shortfall}_shortfall.parquet")

    if market == "DA":
        default_input_params['DA_LMPs'] = wind_df['LMP DA'].values
        wind_cfs = wind_df[f"303_WIND_1-DACF"].values
    elif market == "Both":
        default_input_params['DA_LMPs'] = np.max((wind_df['LMP DA'].values, wind_df['LMP'].values), axis=0)
        wind_cfs = wind_df[f"303_WIND_1-RTCF"].values
    elif market == "RT":
        default_input_params['DA_LMPs'] =  wind_df['LMP'].values
        wind_cfs = wind_df[f"303_WIND_1-RTCF"].values
    else:
        # Two-settlement (PT4): the day-ahead stage must have been run with the
        # SAME delivery arguments, so the lookup carries the same TAG.
        da_file = (Path(__file__).parent / "wind_PEM" / "DA"
                   / f"wind_pem_DA_{np.round(PEM_ratio*default_input_params['wind_mw'], 2)}_{H2_price}{TAG}.csv")
        if not da_file.exists():
            raise FileNotFoundError(
                f"Day-ahead dispatch file not found:\n  {da_file}\n"
                "Run the DA stage first with identical --H2_price and delivery "
                "arguments (market='DA' in load_parameters.py)."
            )
        da_res_df = pd.read_csv(da_file)
        DA_dispatch = da_res_df["wind_out"].values
        default_input_params['DA_LMPs'] = {"DA": wind_df['LMP DA'].values, "RT": wind_df['LMP'].values}
        default_input_params["DA_dispatch"] = DA_dispatch
        wind_cfs = wind_df[f"303_WIND_1-RTCF"].values

    wind_capacity_factors = {t:
                                {'wind_resource_config': {
                                    'capacity_factor':
                                        [wind_cf]}} for t, wind_cf in enumerate(wind_cfs)}
    default_input_params["wind_resource"] = wind_capacity_factors


    # TempfileManager.tempdir = '/tmp/scratch'
    file_name = f'wind_PEM_PT_{H2_price}_{PEM_ratio}{TAG}.json'
    file_dir = Path(__file__).parent / "wind_PEM" / f"{market}"
    if not file_dir.exists():
        os.mkdir(file_dir)

    res = run_design(H2_price, PEM_ratio)
