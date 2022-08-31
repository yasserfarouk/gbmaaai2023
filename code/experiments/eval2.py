from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from utils import (
    PRECISION,
    do_all_tests,
    get_max_util,
    get_rational_fraction,
    get_rationality,
    make_latex_table,
    read_data,
)

matplotlib.use("TkAgg")
# print(matplotlib.rcsetup.interactive_bk)
# print(matplotlib.rcsetup.non_interactive_bk)
# print(matplotlib.rcsetup.all_backends)
# exit()
plt.rcParams["figure.figsize"] = (20, 20)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:.2}".format
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
plt.rcParams.update({"font.size": 28})
matplotlib.rcParams["legend.fontsize"] = 28

RESERVED = ["reserved0", "reserved1"]
# STATS = ["Utility", "Welfare", "Rounds", "P. Dist", "N. Dist", "Time", "AR"]
STATS = [
    "Utility",
    "Agreement Rate",
    "P. Optimality",
    "N. Optimality",
    "Time",
    "Rounds",
]
NONSTATS = ["Condition", "Domain"]


def plot_lines(data, save, output, agreements_only, show):
    reserved_label = "Rational Fraction"
    fig = plt.figure()
    figM = plt.get_current_fig_manager()
    figM.resize(*figM.window.maxsize())
    axs = fig.subplots(3, 2, sharex=True)
    g = sns.lineplot(
        ax=axs[0, 0],
        data=data,
        x=reserved_label,
        y="Utility",
        hue="Condition",
        ci=95,
    )
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    g.legend_.remove()
    g = sns.lineplot(
        ax=axs[0, 1],
        data=data,
        x=reserved_label,
        y="Agreement Rate",
        hue="Condition",
        ci=95,
    )
    g.legend_.remove()
    g = sns.lineplot(
        ax=axs[1, 0],
        data=data,
        x=reserved_label,
        y="Pareto Optimality",
        hue="Condition",
        ci=95,
    )
    g.legend_.remove()
    g = sns.lineplot(
        ax=axs[1, 1],
        data=data,
        x=reserved_label,
        y="Nash Optimality",
        hue="Condition",
        ci=95,
    )
    g.legend_.remove()
    g = sns.lineplot(
        ax=axs[2, 0],
        data=data,
        x=reserved_label,
        y="Rounds",
        hue="Condition",
        ci=95,
    )
    g.legend_.remove()
    g = sns.lineplot(
        ax=axs[2, 1],
        data=data,
        x=reserved_label,
        y="Time",
        hue="Condition",
        ci=95,
    )
    g.legend_.remove()
    axs[2, 0].set(yscale="log")
    axs[2, 1].set(yscale="log")
    if save:
        for ext in ("pdf", "png"):
            plt.savefig(
                f"./figs/{output}-{reserved_label.replace(' ', '')}{'-agreements' if agreements_only else ''}.{ext}",
                bbox_inches="tight",
            )

    if show:
        plt.show()


def plot_bars(data, reserved_label, save, output, agreements_only, show):
    fig = plt.figure()
    figM = plt.get_current_fig_manager()
    figM.resize(*figM.window.maxsize())
    axs = fig.subplots(3, 2, sharex=True)
    g = sns.barplot(
        ax=axs[0, 0],
        data=data,
        x=reserved_label,
        y="Utility",
        hue="Condition",
        ci=95,
    )
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    g.legend_.remove()
    g = sns.barplot(
        ax=axs[0, 1],
        data=data,
        x=reserved_label,
        y="Agreement Rate",
        hue="Condition",
        ci=95,
    )
    g.legend_.remove()
    g = sns.barplot(
        ax=axs[1, 0],
        data=data,
        x=reserved_label,
        y="Pareto Optimality",
        hue="Condition",
        ci=95,
    )
    g.legend_.remove()
    g = sns.barplot(
        ax=axs[1, 1],
        data=data,
        x=reserved_label,
        y="Nash Optimality",
        hue="Condition",
        ci=95,
    )
    g.legend_.remove()
    g = sns.barplot(
        ax=axs[2, 0],
        data=data,
        x=reserved_label,
        y="Rounds",
        hue="Condition",
        ci=95,
    )
    g.legend_.remove()
    g = sns.barplot(
        ax=axs[2, 1],
        data=data,
        x=reserved_label,
        y="Time",
        hue="Condition",
        ci=95,
    )
    g.legend_.remove()
    axs[2, 0].set(yscale="log")
    axs[2, 1].set(yscale="log")
    if save:
        for ext in ("pdf", "png"):
            plt.savefig(
                f"./figs/{output}-{reserved_label.replace(' ', '')}{'-agreements' if agreements_only else ''}.{ext}",
                bbox_inches="tight",
            )

    if show:
        plt.show()


def main(
    files: list[Path] = [],
    failures: bool = True,
    rounds: bool = True,
    timelimit: bool = True,
    agreements_only: bool = False,
    output: str = "fig2",
    max_trials: int = 3,
    insignificant: bool = False,
    significant: bool = True,
    allstats: bool = False,
    show: bool = True,
    save: bool = True,
    rational_only: bool = False,
    lineplots: bool = False,
    barplots: bool = True,
    exceptions: bool = False,
    precision: int = PRECISION,
):
    rationality = get_rationality()
    max_utils = get_max_util()
    rational_fraction = get_rational_fraction()
    (Path() / "figs").mkdir(exist_ok=True, parents=True)
    data = read_data(
        files, failures, rounds, timelimit, agreements_only, max_trials=max_trials
    )
    assert (
        agreements_only
        and len(data["succeeded"].unique()) == 1
        or not agreements_only
        and len(data["succeeded"].unique()) == 2
    )
    # remove the original runs that have no controlled reserved values (some are not 0,0)
    data = data.loc[(data["reserved0"] > 0.09) | (data["reserved1"] > 0.09), :]
    data = data[
        [
            "Welfare",
            "Domain",
            "P. Optimality",
            "N. Optimality",
            "Time",
            "Rounds",
            "AR",
            "Condition",
            "agreement_utils",
        ]
        + RESERVED
    ]
    data["Agreement Rate"] = data["AR"]
    data["Nash Optimality"] = data["N. Optimality"]
    data["Pareto Optimality"] = data["P. Optimality"]
    data["Has Rational"] = data.apply(
        lambda x: rationality[
            (
                x["Domain"],
                int(x["reserved0"] * 10 + 0.5),
                int(x["reserved1"] * 10 + 0.5),
            )
        ],
        axis=1,
    )
    data["Rational Fraction"] = data.apply(
        lambda x: rational_fraction[
            (
                x["Domain"],
                int(x["reserved0"] * 10 + 0.5),
                int(x["reserved1"] * 10 + 0.5),
            )
        ],
        axis=1,
    )
    data["max1"] = data.apply(
        lambda x: max_utils[
            (
                x["Domain"],
                int(x["reserved0"] * 10 + 0.5),
                int(x["reserved1"] * 10 + 0.5),
            )
        ][1],
        axis=1,
    )
    data["max0"] = data.apply(
        lambda x: max_utils[
            (
                x["Domain"],
                int(x["reserved0"] * 10 + 0.5),
                int(x["reserved1"] * 10 + 0.5),
            )
        ][0],
        axis=1,
    )
    data["util0"] = data["agreement_utils"].apply(lambda x: eval(x)[0])
    data["util1"] = data["agreement_utils"].apply(lambda x: eval(x)[1])
    data["Utility (0)"] = (data["util0"] - data["reserved0"]) / (
        data["max0"] - data["reserved0"]
    )
    data["Utility (1)"] = (data["util1"] - data["reserved1"]) / (
        data["max1"] - data["reserved1"]
    )
    if rational_only:
        data = data.loc[data["Has Rational"], :]
    # print(data.columns)
    print("Generating Figures ... ", flush=True, end="")
    for self_util in (False, True):
        reserved_label = (
            "Self Reserved Value" if self_util else "Opponent Reserved Value"
        )
        data["Utility"] = data["Utility (0)"] if not self_util else data["Utility (1)"]
        data[reserved_label] = data["reserved1"]
        if barplots:
            plot_bars(data, reserved_label, save, output, agreements_only, show)
        if lineplots:
            plot_lines(data, save, output, agreements_only, show)
        do_all_tests(
            data,
            insignificant,
            allstats,
            basename=f"exp2{reserved_label.replace(' ', '')}",
            significant=significant,
            exceptions=exceptions,
            stats=STATS + ["Welfare"],
            precision=precision,
        )
        basename = f"exp2{reserved_label.replace(' ', '')}-all"
        make_latex_table(
            data, f"tables/{basename}", stats=STATS + ["Welfare"], count=True
        )
        for r in (1, 5, 9):
            basename = f"exp2{reserved_label.replace(' ', '')}-{r}"
            x = data.loc[
                ((data[reserved_label] * 10).astype(int) == r),
                :,
            ]
            do_all_tests(
                x,
                insignificant,
                allstats,
                basename=basename,
                significant=significant,
                exceptions=exceptions,
                stats=STATS + ["Welfare"],
                precision=precision,
            )
            make_latex_table(
                x,
                f"tables/{basename}",
                stats=STATS + ["Welfare"],
                count=True,
                perdomain=False,
            )
            make_latex_table(
                x,
                f"tables/{basename}-perdomain",
                stats=STATS + ["Welfare"],
                count=True,
                perdomain=True,
            )

    print("DONE")


if __name__ == "__main__":
    typer.run(main)
