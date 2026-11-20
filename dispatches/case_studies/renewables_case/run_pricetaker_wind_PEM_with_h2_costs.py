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
compression) and road-transport costs.

MODEL EXTENSION
---------------
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
from dispatches.case_studies.renewables_case.load_parameters import market, h2_mols_per_kg, PA

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
    help="chi^del: all-in delivery charge per DELIVERED kg [$/kg]. "
         "Literature anchors: 0.2 LH2 trucking only (NREL 88818, 200 km round "
         "trip); 0.9 GH2 trucking only (Argonne IN025, 400 km); 1.9 complete "
         "compressed-gas path; 3.2 complete liquid path.",
    action="store",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--delivery_loss",
    dest="delivery_loss",
    help="eps^del: delivery loss fraction NOT already covered by "
         "--delivery_cost (e.g. boil-off). Use 0 for the gas path.",
    action="store",
    type=float,
    default=0.0,
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
         "0 = conditioning is an off-taker service (use --delivery_cost only); "
         "2.87 = 350-bar compression; 9-13 = liquefaction (NREL 88818). "
         "When > 0, pem_mw denotes the whole hydrogen train.",
    action="store",
    type=float,
    default=0.0,
)
options = parser.parse_args()

PEM_ratio = options.PEM_ratio
H2_price = options.H2_price                 # delivered price, $/kg
delivery_cost = options.delivery_cost       # chi^del,   $/kg delivered
delivery_loss = options.delivery_loss       # eps^del,   fraction
delivery_fixed = options.delivery_fixed     # kappa^del, $/yr
cond_energy = options.cond_energy           # theta^cond, kWh/kg

# ---------------------------------------------------------------------------
# Derived delivery economics
# ---------------------------------------------------------------------------
if not (0.0 <= delivery_loss < 1.0):
    raise ValueError(f"--delivery_loss must be in [0, 1); got {delivery_loss}")
if cond_energy < 0.0:
    raise ValueError(f"--cond_energy must be >= 0; got {cond_energy}")

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
print(f"  implied PEM bid   B          = {PSI_EFF * NETBACK:9.4f} $/MWh")
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
