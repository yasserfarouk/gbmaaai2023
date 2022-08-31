from itertools import product
from math import sqrt
from pathlib import Path
from typing import Iterable

from negmas.helpers.inout import dump
from negmas.helpers.inout import load
import numpy as np
import pandas as pd
from scipy.stats import median_test, ttest_rel, wilcoxon

STATS = ["Welfare", "AR", "P. Optimality", "N. Optimality", "Rounds", "Time"]
NONSTATS = ["Condition", "Domain"]
PRECISION = 5
MAX_TRIALS = 3

########################
# Evaluation Functions #
########################
def dist(x: tuple[float, ...], y: tuple[float, ...]) -> float:
    if x is None or y is None:
        return float("nan")
    return sqrt(sum((a - b) * (a - b) for a, b in zip(x, y)))


def nash_dist(w: tuple[float, ...], n: tuple[float, ...]) -> float:
    return dist(w, n)


def pareto_dist(w: tuple[float, ...], p: Iterable[tuple[float, ...]]) -> float:
    return min(dist(w, x) for x in p)

##################
# Simple Helpers #
##################
def full_name(x: str):
    if x in ("AspirationNegotiator", "MiCRONegotiator"):
        return f"negmas.sao.{x}"
    return f"negmas.genius.gnegotiators.{x}"


################
# Reading Data #
################
def read_data(
    files: list[Path] = [],
    failures: bool = False,
    rounds: bool = True,
    timelimit: bool = True,
    agreements_only: bool = False,
    max_trials: int = MAX_TRIALS,
) -> pd.DataFrame:
    x = []
    for file_path in files:
        clean_results(file_path)
        x.append(pd.read_csv(file_path))  # type: ignore
    data = pd.concat(x, ignore_index=True)
    data["succeeded"] = data["succeeded"].fillna(False).astype(bool)
    if not failures:
        data = data.loc[~data["failed"], :]
    if agreements_only:
        data = data.loc[data["succeeded"], :]
    data: pd.DataFrame
    data.rename(
        columns={
            "mechanism_name": "Mechanism",
            "strategy_name": "Strategy",
            "domain_name": "Domain",
            "relative_rounds": "Rounds",
            "pdist": "P. Optimality",
            "ndist": "N. Optimality",
            "time": "Time",
        },
        inplace=True,
    )
    data["Welfare"] = (data["welfare"] - (data["reserved0"] + data["reserved1"])) / (
        data["max_welfare"] - (data["reserved0"] + data["reserved1"])
    )
    data["P. Optimality"] = 1 - data["P. Optimality"]
    data["N. Optimality"] = 1 - data["N. Optimality"]
    data["Strategy"] = (
        data["Strategy"]
        .str.replace("Negotiator", "")
        .str.replace("CUHKAgent", "CUHK")
        .replace("Aspiration", "Boulware")
        .replace("NiceTitForTat", "NiceTfT")
    )
    data["Mechanism"] = (
        data["Mechanism"]
        .str.replace("AOr", "AOP($2 n_o$)")
        .replace("AOt", "AOP(3min)")
        .replace("TAU0", "TAU$(\\infty, 0)$")
        .replace("TAUinf", "TAU$(\\infty, \\infty)$")
    )
    data["Condition"] = data["Mechanism"] + "+" + data["Strategy"]
    data.drop(columns=["Mechanism", "Strategy"], inplace=True)
    if max_trials > 0:
        data = (
            data.groupby(["Condition", "Domain", "reserved0", "reserved1"])
            .sample(max_trials, replace=True)
            .reset_index(drop=True)
        )
    print(f"Will use {len(data)} data points")
    data["Rounds"] = 2 * data["steps"] / data["n_rounds"]
    data["AR"] = data["succeeded"].astype(int)
    return data

########################
# Data cleaning chores #
########################
def clean_data(
    files: list[Path],
    output: Path,
    max_trials: int,
    failures: bool = False,
) -> pd.DataFrame:
    x = []
    for file_path in files:
        clean_results(file_path)
        x.append(pd.read_csv(file_path))  # type: ignore
    data = pd.concat(x, ignore_index=True)
    if not failures:
        data = data.loc[~data["failed"], :]
    if max_trials > 0:
        data = (
            data.groupby(["Condition", "Domain", "reserved0", "reserved1"])
            .head(max_trials)
            .reset_index(drop=True)
        )
    data.to_csv(output, index=False)
    return data

def clean_results(path: Path, verbose: bool = False):
    if not path.exists():
        return
    with open(path, "r") as f:
        lines = f.readlines()
    n_lines = len(lines)
    if verbose:
        print(f"Found {n_lines} lines", flush=True, end="")
    if not lines:
        return
    first = lines[0]
    corrected = [first]
    changed = False
    for l in lines[1:]:
        if l[:20] == first[:20]:
            changed = True
            continue
        corrected.append(l)
    if not changed:
        if verbose:
            print(f" [green]all clean[/green]", flush=True)
        return
    if verbose:
        print(
            f" ... removed [red]{n_lines - len(corrected)}[/red] lines keeping {len(corrected) - 1} records",
            flush=True,
        )
    with open(path, "w") as f:
        f.writelines(corrected)


#####################
# Domain Properties #
#####################
def get_max_util():
    results = dict()
    for f in (Path() / "domains").glob("*"):
        if not f.is_dir():
            continue
        infos = load(f / "info.json")
        for info in infos:
            results[(f.name, *info["reserved"])] = info["max"]
    return results


def get_rationality():
    results = dict()
    for f in (Path() / "domains").glob("*"):
        if not f.is_dir():
            continue
        infos = load(f / "info.json")
        for info in infos:
            results[(f.name, *info["reserved"])] = info["has_rational"]
    return results


def get_rational_fraction():
    results = dict()
    for f in (Path() / "domains").glob("*"):
        if not f.is_dir():
            continue
        infos = load(f / "info.json")
        for info in infos:
            results[(f.name, *info["reserved"])] = info["f_rational"]
    return results

########################
# TeX output functions #
########################
def adjust_tex(path: Path):
    path = Path(path)
    map = {
        "lrrrrrrrrrrr": "lrr|r|rr|rr|rr|rr",
        "multicolumn{2}{l}": "multicolumn{2}{c}",
    }
    with open(path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        for k, v in map.items():
            line = line.replace(k, v)
        lines[i] = line
    with open(path, "w") as f:
        f.writelines(lines)


def make_latex_table(
    data: pd.DataFrame,
    file_name: str,
    count: bool,
    stats: list[str],
    perdomain: bool = False,
    precision: int = PRECISION,
) -> pd.DataFrame:
    groups = ["Condition"]
    if perdomain:
        groups.append("Domain")
    calculations = {k: ["mean", "std"] for k in stats}
    calculations["AR"] = ["mean"]
    if count:
        calculations[stats[0]] = ["size"] + calculations[stats[0]]
    results = data.groupby(groups).agg(calculations).reset_index()
    results.to_latex(
        file_name,
        index=False,
        escape=False,
        float_format=f"{{:0.{precision}f}}".format if precision > 0 else None,
    )
    adjust_tex(Path(file_name))
    return results


#####################
# Statistical Tests #
#####################
def do_all_tests(
    data: pd.DataFrame,
    insignificant: bool,
    allstats: bool,
    basename: str = "exp1",
    significant: bool = True,
    exceptions: bool = True,
    stats: list[str] = STATS,
    precision: int = PRECISION,
):
    for type in ("ttest", "wilcoxon"):
        results = dict()
        file_name = f"stats/tbl-{basename}-{type}{'-all' if allstats else ''}{'-insig' if insignificant else ''}"
        for stat in stats:
            results[stat] = factorial_test(
                data,
                stat,
                type,
                insignificant,
                allstats,
                significant=significant,
                exceptions=exceptions,
                tbl_name=file_name,
                precision=precision,
            )
        file_name = f"stats/stats-{basename}-{type}{'-all' if allstats else ''}{'-insig' if insignificant else ''}.json"
        dump(results, file_name)


def factorial_test(
    data: pd.DataFrame,
    stat: str,
    type: str,
    insignificant: bool = True,
    allstats: bool = False,
    significant: bool = True,
    exceptions: bool = True,
    tbl_name: str | None = None,
    ignore_nan_ps: bool = True,
    precision: int = PRECISION,
):
    method = dict(ttest=ttest_rel, wilcoxon=wilcoxon, median=median_test)[type]
    alternative = "two-sided"
    results = dict()
    data = data.loc[:, NONSTATS + [stat]]
    tbl = pd.pivot_table(data, index=["Domain"], columns="Condition", values=stat)
    default = {
        "Welfare": 0,
        "Utility": 0,
        "P. Optimality": 0,
        "N. Optimality": 0,
        "AR": 0,
        "Agreement Rate": 0,
        "Rounds": 0,
        "Time": np.nanmax(tbl.values),
    }
    tbl = tbl.fillna(default[stat])
    if tbl_name is not None:
        if tbl_name.endswith(".tex"):
            tbl_name = tbl_name[:-4]
        tbl_name = f"{tbl_name}-{stat.replace(' ','').replace('.', '')}"
        tbl_name += ".tex"
        tbl.to_latex(
            tbl_name,
            index=False,
            escape=False,
            float_format=f"{{:0.{precision}f}}".format if precision > 0 else None,
        )
    correction = len(tbl.columns) - 1
    combinations = (
        (a, b)
        for a, b in product(tbl.columns, tbl.columns)
        if a < b and (allstats or "TAU" in a or "TAU" in b)
    )
    combinations = [(b, a) if b > a else (a, b) for a, b in combinations]
    for a, b in combinations:
        if a == b:
            continue
        try:
            t, p = method(tbl[a], tbl[b], nan_policy="omit", alternative=alternative)  # type: ignore
            if ignore_nan_ps and np.isnan(p):
                p = float("inf")
            if not insignificant and p >= 0.05 / correction:
                continue
            if not significant and p < 0.05 / correction:
                continue
            results[f"{a}-{b}"] = (t, p)
        except Exception as e:
            if exceptions:
                results[f"{a}-{b}"] = str(e)
    return results


