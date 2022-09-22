import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np


def calc_npv(wind_mw, batt_mw, annual_revenue):
    wind_batt_cap = (wind_mw * 1550 + batt_mw * 300 * 4) * 1e3
    discount_rate = 0.05
    lifetime = 30
    PA = ((1 + discount_rate) ** lifetime - 1) / (
        discount_rate * (1 + discount_rate) ** lifetime
    )
    NPV = wind_batt_cap + PA * annual_revenue

    return NPV


def plot_contours(
    df,
    x_axis_name,
    y_axis_name,
    z_axis_name,
    num_levels,
    x_label,
    y_label,
    vmin=None,
    vmax=None,
    levels=None,
):

    df.sort_values(by=[x_axis_name, y_axis_name], inplace=True)
    Z = df.pivot_table(
        index=x_axis_name, columns=y_axis_name, values=z_axis_name
    ).T.values
    X_unique = df[x_axis_name].unique()
    Y_unique = df[y_axis_name].unique()
    X, Y = np.meshgrid(X_unique, Y_unique)

    # make the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    if levels is None:
        levels = list(
            df[z_axis_name]
            .quantile([l / num_levels for l in range(1, num_levels + 1)])
            .round(2)
        )
    levels.sort()
    line_colors = ["black" for _ in levels]

    cp = ax.contour(X, Y, Z, levels=levels, colors=line_colors)
    cpf = ax.contourf(X, Y, Z, num_levels, extend="both", vmin=vmin, vmax=vmax)
    ax.clabel(cp, fontsize="medium", colors=line_colors)

    ax.set_xticks(X_unique)
    ax.set_yticks(Y_unique)
    ax.tick_params(axis="x", labelsize="x-large")
    ax.tick_params(axis="y", labelsize="x-large")

    ax.set_xlabel(x_label, fontsize="xx-large")
    ax.set_ylabel(y_label, fontsize="xx-large")

    return ax
