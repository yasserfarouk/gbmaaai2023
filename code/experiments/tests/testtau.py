from pathlib import Path
from functools import partial
import sys
from time import perf_counter
from enum import Enum
from negmas.gb.evaluators.tau import INFINITE
from negmas.gb.mechanisms.mechanisms import TAUMechanism
from negmas.gb.negotiators.scs import SCSNegotiator
from negmas.inout import Scenario
from negmas.outcomes import issues_from_outcomes
from negmas import pareto_frontier, nash_point
from negmas.outcomes import outcome_space
from negmas.outcomes.base_issue import make_issue
from negmas.outcomes.common import Outcome
from negmas.outcomes.outcome_space import make_os
from negmas.preferences.crisp.mapping import MappingUtilityFunction
from experiment import pareto_dist, nash_dist
from rich import print
import matplotlib
import matplotlib.pyplot as plt

import typer

DOMAINS_BASE_FOLDER = "."


class Experiments(Enum):
    E1 = "e1"
    E2 = "e2"
    E3 = "e3"
    E4 = "e4"
    E5 = "e5"
    E6 = "e6"
    Complete = "complete"
    AllRational = "all-rational"
    NoneRational = "none-rational"
    Large = "large"
    Small = "small"
    Difficult = "difficult"
    Free = "free"


def report_results(
    m: TAUMechanism,
    outcome_space,
    ufuns,
    title=None,
    show=True,
    save=True,
    verbose=True,
):
    print(f"[cyan][bold]{title}[/bold][/cyan]")
    outcomes = list(outcome_space.enumerate_or_sample())
    pareto, _ = pareto_frontier(ufuns, outcomes=outcomes)
    nash, _ = nash_point(ufuns, pareto)
    valid_outcomes = [
        set(_ for _ in outcomes if u(_) > u.reserved_value) for u in ufuns
    ]
    if verbose:
        print(valid_outcomes)
    rational_outcomes = list(valid_outcomes[0].intersection(valid_outcomes[1]))
    if not rational_outcomes:
        print("No rational outcomes are found. will not converge to an agreement")
    elif verbose:
        print(f"The set of rational outcome is: {rational_outcomes}")
    # print("Negotiation Trace:\n", m.full_trace)
    print("Agreement: ", m.agreement)
    agreement_utils = tuple(u(m.agreement) for u in ufuns)
    print("Pareto Distance: ", pareto_dist(agreement_utils, pareto))
    print("Nash Distance: ", nash_dist(agreement_utils, nash))  # type: ignore
    for item in m.full_trace:
        indx = int(item.negotiator[-1])
        valid = valid_outcomes[indx]
        if item.offer is not None and item.offer not in valid:
            print(f"[red]{item.offer} not in {indx}'s valid outcomes: {valid}[/red]")
    assert (
        m.agreement is None or m.agreement in rational_outcomes
    ), f"Agreement {agreement_utils} is [red]IRRATIONAL[/red]"
    if show:
        matplotlib.use("tkAgg")
        m.plot()
        if title:
            plt.suptitle(title)
        plt.show()


def run_negotiation(
    outcome_space,
    ufuns,
    min_unique=0,
    verbose=False,
    title="TAU+SCS",
    show=True,
    save=True,
):
    if min_unique < 0:
        min_unique = INFINITE
    _start = perf_counter()
    m = TAUMechanism(outcome_space=outcome_space, min_unique=min_unique)
    for i, u in enumerate(ufuns):
        if u is None:
            sys.exit(1)
        neg = SCSNegotiator(name=f"n{i}", id=f"n{i}")
        neg.id = f"n{i}"
        m.add(neg, ufun=u)  # type: ignore
    m.run()
    t = perf_counter() - _start
    report_results(m, outcome_space, ufuns, title, show, save, verbose)
    return t


def example_1_completeness(show, save, verbose, beta: int = 0):
    domain = Scenario.from_genius_folder(f"{DOMAINS_BASE_FOLDER}/0000003NiceOrDie/", ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain: Scenario
    domain.ufuns[0].reserved_value = 0
    domain.ufuns[1].reserved_value = 0.9
    return run_negotiation(
        domain.agenda,
        domain.ufuns,
        title=f"TAU($\infty, {beta}$) + SCS: {'Complete' if 0 <= beta<2 else 'Incomplete'}",
        show=show,
        verbose=verbose,
        save=save,
        min_unique=beta,
    )


def free_run(show, save, verbose, folder: Path | None, beta: int = 0):
    if folder is None:
        print(f"[red]Error: Must pass folder for free runs[/red]")
        sys.exit(1)
    domain: Scenario = Scenario.from_genius_folder(folder, ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain.ufuns[0].reserved_value = 0.3
    domain.ufuns[1].reserved_value = 0.4
    ufuns, outcome_space = domain.ufuns, domain.agenda
    return run_negotiation(
        outcome_space,
        ufuns,
        show=show,
        save=save,
        verbose=verbose,
        title=f"TAU($\infty, {beta}$) + SCS: {str(folder.name)}",
        min_unique=beta,
    )


def example_5_large(show, save, verbose, beta: int = 0):
    folder = f"{DOMAINS_BASE_FOLDER}/0015625Kitchen/"
    domain: Scenario = Scenario.from_genius_folder(folder, ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain.ufuns[0].reserved_value = 0.95
    domain.ufuns[1].reserved_value = 0.95
    ufuns, outcome_space = domain.ufuns, domain.agenda
    return run_negotiation(
        outcome_space,
        ufuns,
        show=show,
        save=save,
        verbose=verbose,
        title=f"TAU($\infty, {beta}$) + SCS: Example (Kitchen)",
        min_unique=beta,
    )


def example_6_difficult(show, save, verbose, beta: int = 0):
    folder = f"{DOMAINS_BASE_FOLDER}/0000128Outfit/"
    domain: Scenario = Scenario.from_genius_folder(folder, ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain.ufuns[0].reserved_value = 0.8
    domain.ufuns[1].reserved_value = 0.9
    ufuns, outcome_space = domain.ufuns, domain.agenda
    return run_negotiation(
        outcome_space,
        ufuns,
        show=show,
        save=save,
        verbose=verbose,
        title=f"TAU($\infty, {beta}$) + SCS: Example (Outfit)",
        min_unique=beta,
    )


def example_4_small(show, save, verbose, beta: int = 0):
    folder = f"{DOMAINS_BASE_FOLDER}/0000128Outfit/"
    domain: Scenario = Scenario.from_genius_folder(folder, ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain.ufuns[0].reserved_value = 0.5
    domain.ufuns[1].reserved_value = 0.6
    ufuns, outcome_space = domain.ufuns, domain.agenda
    return run_negotiation(
        outcome_space,
        ufuns,
        show=show,
        save=save,
        verbose=verbose,
        title=f"TAU($\infty, {beta}$) + SCS: Example (Outfit)",
        min_unique=beta,
    )


def example_2_all_rational(show, save, verbose, beta: int = 0):
    domain = Scenario.from_genius_folder(f"{DOMAINS_BASE_FOLDER}/0000003NiceOrDie/", ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain: Scenario
    domain.ufuns[0].reserved_value = 0
    domain.ufuns[1].reserved_value = 0
    return run_negotiation(
        domain.agenda,
        domain.ufuns,
        title="TAU($\infty$, 0) + SCS: All Rational (Success)",
        show=show,
        verbose=verbose,
        save=save,
        min_unique=beta,
    )


def example_3_none_rational(show, save, verbose, beta: int = 0):
    domain = Scenario.from_genius_folder(f"{DOMAINS_BASE_FOLDER}/0000003NiceOrDie/", ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain: Scenario
    domain.ufuns[0].reserved_value = 0.9
    domain.ufuns[1].reserved_value = 0.9
    return run_negotiation(
        domain.agenda,
        domain.ufuns,
        title="TAU($\infty$, 0) + SCS: No Rational (Expected Failure)",
        show=show,
        verbose=verbose,
        save=save,
        min_unique=beta,
    )


def main(
    experiment: Experiments,
    show: bool = True,
    save: bool = True,
    verbose: bool = True,
    beta: int = 0,
    domain: Path | None = None,
):
    {
        Experiments.E1: example_1_completeness,
        Experiments.E2: example_2_all_rational,
        Experiments.E3: example_3_none_rational,
        Experiments.E4: example_4_small,
        Experiments.E5: example_5_large,
        Experiments.E6: example_6_difficult,
        Experiments.Complete: example_1_completeness,
        Experiments.AllRational: example_2_all_rational,
        Experiments.NoneRational: example_3_none_rational,
        Experiments.Small: example_4_small,
        Experiments.Large: example_5_large,
        Experiments.Difficult: example_6_difficult,
        Experiments.Free: partial(free_run, folder=domain),
    }[experiment](show, save, verbose, beta=beta)


if __name__ == "__main__":
    typer.run(main)
