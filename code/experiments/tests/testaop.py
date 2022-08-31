from pathlib import Path
import sys
from functools import partial
from time import perf_counter
from negmas.inout import Scenario
from negmas.outcomes.common import Outcome
from negmas.sao.mechanism import SAOMechanism
from negmas import pareto_frontier, nash_point
from negmas.helpers import instantiate, humanize_time
from experiment import pareto_dist, nash_dist
from rich import print
import matplotlib
import matplotlib.pyplot as plt
from enum import Enum
import typer

DOMAINS_BASE_FOLDER = "."


def full_name(x: str):
    x = dict(micro="MiCRONegotiator", boulware="AspirationNegotiator").get(x, x)
    if x in ("AspirationNegotiator", "MiCRONegotiator"):
        return f"negmas.sao.{x}"
    return f"negmas.genius.gnegotiators.{x}"


class Experiments(Enum):
    E1 = "e1"
    E2 = "e2"
    E3 = "e3"
    E4 = "e4"
    E5 = "e5"
    E6 = "e6"
    Incomplete = "incomplete"
    AllRational = "all-rational"
    NoneRational = "none-rational"
    Small = "small"
    Difficult = "difficult"
    Large = "large"
    Free = "free"


def report_results(
    m: SAOMechanism,
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
        assert (
            item.offer in valid
        ), f"{item.offer} not in {indx}'s valid outcomes: {valid}"
    assert (
        m.agreement is None or m.agreement in rational_outcomes
    ), f"Agreement {agreement_utils} is [red]IRRATIONAL[/red]"
    if show:
        matplotlib.use("tkAgg")
        m.plot(xdim="step")
        if title:
            plt.suptitle(title)
        plt.show()


def run_negotiation(
    outcome_space,
    ufuns,
    timelimit,
    n_steps,
    verbose=False,
    title="MiCRO",
    show=True,
    save=True,
    strategy="MiCRONegotiator",
):
    # print("Making Negotiation")
    _start = perf_counter()
    m = SAOMechanism(
        outcome_space=outcome_space,
        time_limit=timelimit if timelimit and timelimit > 0 else None,
        n_steps=n_steps if n_steps and n_steps > 0 else None,
        extra_callbacks=True,
    )
    for i, u in enumerate(ufuns):
        if u is None:
            print("[red]No ufus. Cannot run a negotiation[/red]")
            sys.exit(1)
        neg = instantiate(full_name(strategy), name=f"n{i}", id=f"n{i}")
        neg.id = f"n{i}"
        m.add(neg, ufun=u)  # type: ignore
    # print("Starting Negotiation")
    m.run()
    t = perf_counter() - _start
    report_results(m, outcome_space, ufuns, title, show, save, verbose)
    return t


def example_1_incompleteness(show, save, verbose, strategy: str = "MiCRONegotiator"):
    domain = Scenario.from_genius_folder(f"{DOMAINS_BASE_FOLDER}/0000003NiceOrDie/", ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain: Scenario
    domain.ufuns[0].reserved_value = 0
    domain.ufuns[1].reserved_value = 0.9
    return run_negotiation(
        domain.agenda,
        domain.ufuns,
        None,
        2 * domain.agenda.cardinality,
        title=f"AOP + {strategy}: Incomplete",
        show=show,
        verbose=verbose,
        save=save,
        strategy=strategy,
    )


def example_2_all_rational(show, save, verbose, strategy: str = "MiCRONegotiator"):
    domain = Scenario.from_genius_folder(f"{DOMAINS_BASE_FOLDER}/0000003NiceOrDie/", ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain: Scenario
    domain.ufuns[0].reserved_value = 0
    domain.ufuns[1].reserved_value = 0
    return run_negotiation(
        domain.agenda,
        domain.ufuns,
        None,
        2 * domain.agenda.cardinality,
        title=f"AOP + {strategy}: All Rational (Success)",
        show=show,
        verbose=verbose,
        save=save,
        strategy=strategy,
    )


def example_3_none_rational(show, save, verbose, strategy: str = "MiCRONegotiator"):
    domain = Scenario.from_genius_folder(f"{DOMAINS_BASE_FOLDER}/0000003NiceOrDie/", ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain: Scenario
    domain.ufuns[0].reserved_value = 0.9
    domain.ufuns[1].reserved_value = 0.9
    return run_negotiation(
        domain.agenda,
        domain.ufuns,
        None,
        2 * domain.agenda.cardinality,
        title=f"AOP + {strategy}: None Rational (Should Fail)",
        show=show,
        verbose=verbose,
        save=save,
        strategy=strategy,
    )


def free_run(
    show, save, verbose, strategy: str = "MiCRONegotiator", folder: Path | None = None
):
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
        None,
        2 * domain.agenda.cardinality,
        show=show,
        save=save,
        verbose=verbose,
        title=f"AOP + {strategy}: Example",
        strategy=strategy,
    )


def example_5_large(show, save, verbose, strategy: str = "MiCRONegotiator"):
    folder = f"{DOMAINS_BASE_FOLDER}/0015625Kitchen/"
    domain: Scenario = Scenario.from_genius_folder(folder, ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain.ufuns[0].reserved_value = 0.95
    domain.ufuns[1].reserved_value = 0.95
    ufuns, outcome_space = domain.ufuns, domain.agenda
    return run_negotiation(
        outcome_space,
        ufuns,
        None,
        2 * domain.agenda.cardinality,
        show=show,
        save=save,
        verbose=verbose,
        title=f"AOP + {strategy}: Example",
        strategy=strategy,
    )


def example_4_difficult(show, save, verbose, strategy: str = "MiCRONegotiator"):
    folder = f"{DOMAINS_BASE_FOLDER}/0000128Outfit/"
    domain: Scenario = Scenario.from_genius_folder(folder, ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain.ufuns[0].reserved_value = 0.8
    domain.ufuns[1].reserved_value = 0.9
    ufuns, outcome_space = domain.ufuns, domain.agenda
    return run_negotiation(
        outcome_space,
        ufuns,
        None,
        2 * domain.agenda.cardinality,
        show=show,
        save=save,
        verbose=verbose,
        title=f"AOP + {strategy}: Example",
        strategy=strategy,
    )


def example_6_difficult(show, save, verbose, strategy: str = "MiCRONegotiator"):
    folder = f"{DOMAINS_BASE_FOLDER}/0000128Outfit/"
    domain: Scenario = Scenario.from_genius_folder(folder, ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain.ufuns[0].reserved_value = 0.8
    domain.ufuns[1].reserved_value = 0.9
    ufuns, outcome_space = domain.ufuns, domain.agenda
    return run_negotiation(
        outcome_space,
        ufuns,
        None,
        2 * domain.agenda.cardinality,
        show=show,
        save=save,
        verbose=verbose,
        title=f"AOP + {strategy}: Example",
        strategy=strategy,
    )


def example_4_small(show, save, verbose, strategy: str = "MiCRONegotiator"):
    folder = f"{DOMAINS_BASE_FOLDER}/0000128Outfit/"
    domain: Scenario = Scenario.from_genius_folder(folder, ignore_discount=True, ignore_reserved=False)  # type: ignore
    domain.ufuns[0].reserved_value = 0.5
    domain.ufuns[1].reserved_value = 0.6
    ufuns, outcome_space = domain.ufuns, domain.agenda
    return run_negotiation(
        outcome_space,
        ufuns,
        None,
        2 * domain.agenda.cardinality,
        show=show,
        save=save,
        verbose=verbose,
        title=f"AOP + {strategy}: Example",
        strategy=strategy,
    )


def main(
    experiment: Experiments,
    strategy: str = "MiCRONegotiator",
    show: bool = True,
    save: bool = True,
    verbose: bool = True,
    domain: Path | None = None,
):
    time = {
        Experiments.E1: example_1_incompleteness,
        Experiments.E2: example_2_all_rational,
        Experiments.E3: example_3_none_rational,
        Experiments.E4: example_4_small,
        Experiments.E5: example_5_large,
        Experiments.E6: example_6_difficult,
        Experiments.Difficult: example_6_difficult,
        Experiments.Incomplete: example_1_incompleteness,
        Experiments.AllRational: example_2_all_rational,
        Experiments.NoneRational: example_2_all_rational,
        Experiments.Small: example_4_small,
        Experiments.Large: example_5_large,
        Experiments.Free: partial(free_run, folder=domain),
    }[experiment](show, save, verbose, strategy=strategy)
    print(f"Done in {humanize_time(time, show_us=True)}")


if __name__ == "__main__":
    typer.run(main)
